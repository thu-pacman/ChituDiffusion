import json
import os
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chitu_diffusion.evaluation.eval_manager import EvalStrategy
from chitu_diffusion.evaluation.utils.get_eval_videos import collect_videos_and_prompts
from chitu_diffusion.runtime.output_naming import parse_video_name, slugify_prompt
from chitu_diffusion.runtime.output_layout import write_json

logger = getLogger(__name__)


class ReferenceMetricStrategy(EvalStrategy):
    def __init__(self, metric_name: str, output_dir: str = "./eval_out"):
        super().__init__()
        self.type = metric_name
        self.requires_reference = True
        self.output_dir = output_dir
        self.run_name = f"{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _reference_path(self, args: Any) -> str:
        ref_path = getattr(args.eval, "reference_path", None)
        if ref_path is None:
            return ""
        return str(ref_path).strip()

    def _build_video_pairs(
        self,
        video_prompt: Dict[str, str],
        generated_dir: str,
        reference_dir: str,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        generated_base = Path(generated_dir)
        reference_base = Path(reference_dir)

        def _normalize_model(value: Any) -> str:
            return str(value or "").strip().lower()

        def _match_key(model: Any, prompt: str, seed: Any) -> Tuple[str, str, str]:
            return (
                _normalize_model(model),
                slugify_prompt(prompt),
                "none" if seed is None else str(seed),
            )

        def _read_json(path: Path) -> Optional[Dict[str, Any]]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                return None
            return data if isinstance(data, dict) else None

        def _reference_run_model(base_dir: Path) -> str:
            data = _read_json(base_dir / "system_params.json")
            if data is None:
                return ""
            return _normalize_model(data.get("model_name"))

        def _sidecar_for_video(video_path: Path) -> Optional[Dict[str, Any]]:
            return _read_json(video_path.with_suffix(".json"))

        def _load_reference_index(base_dir: Path) -> Dict[Tuple[str, str, str], Path]:
            mapping: Dict[Tuple[str, str, str], Path] = {}
            run_model = _reference_run_model(base_dir)

            for reference_video in base_dir.rglob("*.mp4"):
                sidecar = _sidecar_for_video(reference_video)
                if sidecar is not None and sidecar.get("prompt") is not None:
                    ref_model = sidecar.get("model_name") or sidecar.get("model") or run_model
                    mapping[_match_key(ref_model, str(sidecar["prompt"]), sidecar.get("seed"))] = reference_video
                    continue

                parsed = parse_video_name(reference_video.name)
                if parsed is not None and run_model:
                    prompt_slug, seed, _step = parsed
                    mapping[(run_model, prompt_slug, seed)] = reference_video

            # Also support sidecars that name a file when the mp4 name is not canonical.
            for sidecar_path in base_dir.rglob("*.json"):
                if sidecar_path.name in {"system_params.json", "request_params.json"}:
                    continue
                data = _read_json(sidecar_path)
                if data is None:
                    continue

                filename = data.get("filename")
                prompt = data.get("prompt")
                seed = data.get("seed")
                if not filename or prompt is None:
                    continue
                candidate = sidecar_path.parent / str(filename)
                if not candidate.exists():
                    candidate = base_dir / str(filename)
                if not candidate.exists():
                    continue
                ref_model = data.get("model_name") or data.get("model") or run_model
                mapping[_match_key(ref_model, str(prompt), seed)] = candidate
            return mapping

        def _generated_key(video_path: Path, prompt: str) -> Tuple[str, str, str]:
            sidecar = _sidecar_for_video(video_path)
            if sidecar is not None:
                gen_model = sidecar.get("model_name") or sidecar.get("model") or model_name
                gen_prompt = sidecar.get("prompt", prompt)
                return _match_key(gen_model, str(gen_prompt), sidecar.get("seed"))

            parsed = parse_video_name(video_path.name)
            if parsed is not None:
                prompt_slug, seed, _step = parsed
                return (_normalize_model(model_name), prompt_slug, seed)

            return _match_key(model_name, prompt, None)

        ref_match_map = _load_reference_index(reference_base)

        pairs: List[Dict[str, str]] = []
        for video_key, prompt in video_prompt.items():
            generated_path = generated_base / video_key
            video_name = generated_path.name
            task_id = generated_path.parent.name if generated_path.parent != generated_base else None
            if not generated_path.exists():
                continue

            match_key = _generated_key(generated_path, prompt)
            matched_reference = ref_match_map.get(match_key)

            if matched_reference is not None:
                pairs.append(
                    {
                        "task_id": task_id,
                        "video_name": video_name,
                        "video_key": video_key,
                        "generated": str(generated_path.resolve()),
                        "reference": str(matched_reference.resolve()),
                        "match_model": match_key[0],
                        "match_prompt": match_key[1],
                        "match_seed": match_key[2],
                    }
                )

        if not pairs:
            logger.warning(
                "No reference videos matched in reference_path by (model, prompt, seed). "
                "reference_path=%s generated_dir=%s",
                reference_base,
                generated_base,
            )
        return pairs

    def get_eval_videos(self, args, **kwargs):
        video_prompt, videos_dir = collect_videos_and_prompts(args)
        reference_dir = self._reference_path(args)
        if not reference_dir:
            payload = {
                "name": self.run_name,
                "metric_type": self.type,
                "video_prompt": video_prompt,
                "generated_dir": videos_dir,
                "reference_dir": None,
                "pairs": [],
                "num_eval_items": 0,
                "skip_reason": "reference_path is empty",
            }
            return payload

        reference_path = Path(reference_dir).resolve()
        if not reference_path.exists() or not reference_path.is_dir():
            payload = {
                "name": self.run_name,
                "metric_type": self.type,
                "video_prompt": video_prompt,
                "generated_dir": videos_dir,
                "reference_dir": str(reference_path),
                "pairs": [],
                "num_eval_items": 0,
                "skip_reason": f"invalid reference_path: {reference_path}",
            }
            return payload

        model_name = getattr(getattr(args, "models", None), "name", None)
        pairs = self._build_video_pairs(
            video_prompt,
            videos_dir,
            str(reference_path),
            model_name=model_name,
        )
        payload = {
            "name": self.run_name,
            "metric_type": self.type,
            "video_prompt": video_prompt,
            "generated_dir": videos_dir,
            "reference_dir": str(reference_path),
            "pairs": pairs,
            "num_eval_items": len(pairs),
        }
        return payload

    def save_result(self, result: Dict[str, Any]):
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, f"{self.run_name}_eval_results.json")
        write_json(out_path, self._with_task_groups(result))
        return out_path

    def _with_task_groups(self, result: Dict[str, Any]) -> Dict[str, Any]:
        per_video = result.get("per_video")
        if not isinstance(per_video, list):
            pairs = result.get("pairs")
            if isinstance(pairs, list):
                grouped: Dict[str, Dict[str, Any]] = {}
                for pair in pairs:
                    if not isinstance(pair, dict):
                        continue
                    task_id = str(pair.get("task_id") or "unknown")
                    grouped.setdefault(task_id, {"pairs": []})["pairs"].append(pair)
                for group in grouped.values():
                    group["num_pairs"] = len(group["pairs"])
                    if "score" in result:
                        group["score"] = result["score"]
                result["by_task_id"] = grouped
            return result

        grouped: Dict[str, Dict[str, Any]] = {}
        for item in per_video:
            if not isinstance(item, dict):
                continue
            task_id = str(item.get("task_id") or "unknown")
            grouped.setdefault(task_id, {"items": []})["items"].append(item)

        for task_id, group in grouped.items():
            scores = [
                float(item["score"])
                for item in group["items"]
                if isinstance(item, dict) and "score" in item
            ]
            group["num_videos"] = len(group["items"])
            if scores:
                group["mean_score"] = sum(scores) / len(scores)

        result["by_task_id"] = grouped
        return result
