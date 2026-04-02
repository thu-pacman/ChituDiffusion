from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Dict, List, Optional, Sequence

from .utils.distributed import dist_run

logger = getLogger(__name__)

class EvalStrategy(ABC):
    def __init__(self):
        self.type = None
        self.requires_reference = False
        
    @abstractmethod
    def get_eval_videos(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass



class EvalManager():
    def __init__(self):
        self.strategy: Optional[EvalStrategy] = None
        self.eval_result: Dict[str, Dict[str, Any]] = {}
        self._strategy_registry = {
            "vbench": self._build_vbench,
            "fid": self._build_fid,
            "fvd": self._build_fvd,
            "psnr": self._build_psnr,
            "ssim": self._build_ssim,
            "lpips": self._build_lpips,
        }

    def set_strategy(self, strategy: EvalStrategy):
        self.strategy = strategy

    def normalize_eval_types(self, eval_type: Any) -> List[str]:
        if eval_type is None:
            return []
        if isinstance(eval_type, str):
            value = eval_type.strip().lower()
            if value in {"", "none", "null"}:
                return []
            return [value]
        if isinstance(eval_type, Sequence):
            normalized: List[str] = []
            for item in eval_type:
                if item is None:
                    continue
                value = str(item).strip().lower()
                if value in {"", "none", "null"}:
                    continue
                normalized.append(value)
            return normalized
        value = str(eval_type).strip().lower()
        if value in {"", "none", "null"}:
            return []
        return [value]

    def _build_vbench(self, output_dir: Optional[str] = None) -> EvalStrategy:
        from chitu_diffusion.eval.strategy.Vbench import VbenchStrategy

        if output_dir:
            return VbenchStrategy(output_dir=output_dir)
        return VbenchStrategy()

    def _build_fid(self, output_dir: Optional[str] = None) -> EvalStrategy:
        from chitu_diffusion.eval.strategy.Fid import FidStrategy

        if output_dir:
            return FidStrategy(output_dir=output_dir)
        return FidStrategy()

    def _build_fvd(self, output_dir: Optional[str] = None) -> EvalStrategy:
        from chitu_diffusion.eval.strategy.Fvd import FvdStrategy

        if output_dir:
            return FvdStrategy(output_dir=output_dir)
        return FvdStrategy()

    def _build_psnr(self, output_dir: Optional[str] = None) -> EvalStrategy:
        from chitu_diffusion.eval.strategy.Psnr import PsnrStrategy

        if output_dir:
            return PsnrStrategy(output_dir=output_dir)
        return PsnrStrategy()

    def _build_ssim(self, output_dir: Optional[str] = None) -> EvalStrategy:
        from chitu_diffusion.eval.strategy.Ssim import SsimStrategy

        if output_dir:
            return SsimStrategy(output_dir=output_dir)
        return SsimStrategy()

    def _build_lpips(self, output_dir: Optional[str] = None) -> EvalStrategy:
        from chitu_diffusion.eval.strategy.Lpips import LpipsStrategy

        if output_dir:
            return LpipsStrategy(output_dir=output_dir)
        return LpipsStrategy()

    def create_strategy(self, eval_type: str, output_dir: Optional[str] = None) -> Optional[EvalStrategy]:
        builder = self._strategy_registry.get(eval_type)
        if builder is None:
            return None
        return builder(output_dir=output_dir)

    def _append_skipped(self, eval_type: str, message: str):
        self.eval_result[eval_type] = {
            "type": eval_type,
            "status": "skipped",
            "message": message,
            "result": None,
        }

    def _reference_path(self, args: Any) -> Optional[str]:
        eval_cfg = getattr(args, "eval", None)
        if eval_cfg is None:
            return None
        path = getattr(eval_cfg, "reference_path", None)
        if path is None:
            return None
        path = str(path).strip()
        if not path:
            return None
        return path
    
    def run(self, args, eval_types: Optional[List[str]] = None, output_dir: Optional[str] = None, **kwargs):
        if eval_types is None:
            eval_types = self.normalize_eval_types(getattr(args.eval, "eval_type", None))

        if not eval_types:
            logger.info("No eval strategy configured, skip evaluation.")
            return {}

        reference_path = self._reference_path(args)

        for eval_type in eval_types:
            strategy = self.create_strategy(eval_type, output_dir=output_dir)
            if strategy is None:
                self._append_skipped(eval_type, f"Unsupported eval type: {eval_type}")
                logger.warning("Unsupported eval type: %s, skipped.", eval_type)
                continue

            if strategy.requires_reference and reference_path is None:
                self._append_skipped(
                    eval_type,
                    "reference_path is required for this metric but not configured",
                )
                logger.warning(
                    "Eval strategy %s requires reference_path, skipped.",
                    eval_type,
                )
                continue

            self.set_strategy(strategy)
            try:
                result = dist_run(self.strategy, args, **kwargs)
                if result is None:
                    self.eval_result[eval_type] = {
                        "type": eval_type,
                        "status": "skipped",
                        "message": "no eval items",
                        "result": None,
                    }
                elif isinstance(result, dict) and result.get("status") in {
                    "skipped",
                    "failed",
                }:
                    logger.warning(
                            "Eval strategy '%s' %s: %s",
                            eval_type,
                            result.get("status"),
                            result.get("message", ""),
                        )
                    self.eval_result[eval_type] = {
                        "type": eval_type,
                        "status": result.get("status"),
                        "message": result.get("message", ""),
                        "result": result,
                    }
                else:
                    self.eval_result[eval_type] = {
                        "type": eval_type,
                        "status": "success",
                        "message": "ok",
                        "result": result,
                    }
            except Exception as exc:
                logger.exception("Eval strategy '%s' failed: %s", eval_type, exc)
                self.eval_result[eval_type] = {
                    "type": eval_type,
                    "status": "failed",
                    "message": str(exc),
                    "result": None,
                }

        return self.eval_result


