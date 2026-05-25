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
    ) -> List[Dict[str, str]]:
        generated_base = Path(generated_dir)
        reference_base = Path(reference_dir)

        def _triplet_key(prompt: str, seed: Any, step: Any) -> Tuple[str, str, str]:
            return (
                slugify_prompt(prompt),
                "none" if seed is None else str(seed),
                "none" if step is None else str(step),
            )

        def _load_sidecar_triplets(base_dir: Path) -> Dict[Tuple[str, str, str], Path]:
            mapping: Dict[Tuple[str, str, str], Path] = {}
            for sidecar in base_dir.glob("*.json"):
                try:
                    with open(sidecar, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue

                filename = data.get("filename")
                prompt = data.get("prompt")
                seed = data.get("seed")
                step = data.get("step")
                if not filename or prompt is None:
                    continue
                candidate = base_dir / str(filename)
                if not candidate.exists():
                    continue
                mapping[_triplet_key(str(prompt), seed, step)] = candidate
            return mapping

        ref_triplet_map: Dict[Tuple[str, str, str], Path] = {}
        for reference_video in reference_base.glob("*.mp4"):
            parsed = parse_video_name(reference_video.name)
            if parsed is not None:
                ref_triplet_map[parsed] = reference_video

        # Sidecar metadata can capture raw prompt/seed/step and helps when names are not canonical.
        ref_triplet_map.update(_load_sidecar_triplets(reference_base))

        pairs: List[Dict[str, str]] = []
        for video_key in video_prompt.keys():
            generated_path = generated_base / video_key
            video_name = generated_path.name
            task_id = generated_path.parent.name if generated_path.parent != generated_base else None
            reference_path = reference_base / video_key
            if not reference_path.exists():
                reference_path = reference_base / video_name
            if not generated_path.exists():
                continue

            matched_reference: Optional[Path] = None
            if reference_path.exists():
                matched_reference = reference_path
            else:
                triplet = parse_video_name(video_name)
                if triplet is not None:
                    matched_reference = ref_triplet_map.get(triplet)

            if matched_reference is not None:
                pairs.append(
                    {
                        "task_id": task_id,
                        "video_name": video_name,
                        "video_key": video_key,
                        "generated": str(generated_path.resolve()),
                        "reference": str(matched_reference.resolve()),
                    }
                )

        if pairs:
            return pairs

        generated_files = sorted(generated_base.rglob("*.mp4"))
        reference_files = sorted(reference_base.glob("*.mp4"))
        n = min(len(generated_files), len(reference_files))
        if n == 0:
            return []

        logger.warning(
            "No same-name match found in reference_path, fallback to sorted pairing by index."
        )
        for idx in range(n):
            task_id = generated_files[idx].parent.name if generated_files[idx].parent != generated_base else None
            pairs.append(
                {
                    "task_id": task_id,
                    "video_name": generated_files[idx].name,
                    "video_key": str(generated_files[idx].relative_to(generated_base)),
                    "generated": str(generated_files[idx].resolve()),
                    "reference": str(reference_files[idx].resolve()),
                }
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

        pairs = self._build_video_pairs(video_prompt, videos_dir, str(reference_path))
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
