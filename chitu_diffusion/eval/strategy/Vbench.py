import torch
import numpy as np
from typing import Optional, Any, Dict, List
import torch.distributed as dist
import functools
from logging import getLogger
from chitu_diffusion.eval.eval_manager import EvalStrategy
import sys
import chitu_diffusion.eval.utils.distributed as my_dist
sys.modules["vbench.distributed"] = my_dist
from vbench.utils import init_submodules, save_json
from ..utils.get_eval_videos import collect_videos_and_prompts
from ..utils.model_adapt import _vbench_fp32_guard, _force_fp32_recursive
from ..utils.distributed import get_world_size, barrier, print0, get_rank, distribute_list_to_rank, gather_list_of_dict, all_gather
import importlib
from pathlib import Path
import os
from datetime import datetime
logger = getLogger(__name__)


def _ensure_numpy_sctypes_compat() -> None:
    """
    Compatibility shim for old deps (e.g. imgaug) that still access np.sctypes.
    NumPy 2.0 removed this attribute.
    """
    if hasattr(np, "sctypes"):
        return

    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, np.object_, np.str_, np.bytes_],
    }


class VbenchStrategy(EvalStrategy):
    def __init__(
        self,
        output_dir: str = './vbench_out',
        local: bool = True,
        dimension_list: Optional[List[str]] = None,
        name: str="vbench"
    ):

        super().__init__()
        self.type = "vbench"
        self.output_path = output_dir
        if dimension_list is None:
            self.dimension_list = [
            "subject_consistency",
            "background_consistency",
            "motion_smoothness",
            "dynamic_degree",
            "aesthetic_quality",
            "imaging_quality",
            ]
        else:
            self.dimension_list = dimension_list
        self.name = "vbench_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        

    def get_eval_videos(self, *args, **kwargs):
        rank=get_rank()
        video_prompt, videos_dir = collect_videos_and_prompts(args)
        cur_full_info_list = []
        base = Path(videos_dir)
        if rank==0:
            payload = {
            "name": self.name,
            "videos_dir": videos_dir,
            "video_prompt": video_prompt,          # key: filename(with suffix), value: prompt
            "dimension_list": self.dimension_list, # 让所有rank一致
            "output_path": self.output_path,       # rank0写json/写结果要用
            "num_eval_items": len(video_prompt),
            }  
            for video_name, prompt in video_prompt.items():
                video_path = (base / video_name).resolve()
                cur_full_info_list.append(
                {
                    "prompt_en": prompt,
                    "dimension": self.dimension_list,
                    "video_list": [str(video_path)],
                }
                ) 
        else:
            payload = None
        os.makedirs(self.output_path, exist_ok=True)
        out_json = os.path.join(self.output_path, f"{self.name}_full_info.json")
        save_json(cur_full_info_list, out_json)

        return payload

    def evaluate(self, 
        local: bool = True,
        read_frame: bool = False,
        *args, 
        **kwargs):

        _ensure_numpy_sctypes_compat()

        with _vbench_fp32_guard(), torch.autocast(device_type="cuda", enabled=False):
            submodules_dict = init_submodules(self.dimension_list, local=local, read_frame=read_frame)
            _force_fp32_recursive(submodules_dict, device='cuda')

            if get_world_size() > 1:
                barrier()
            meta_json = os.path.join(self.output_path, f"{self.name}_full_info.json")
            
            results_dict = {}
            for dim in self.dimension_list:
                try:
                    dim_module = importlib.import_module(f"vbench.{dim}")
                    compute_fn = getattr(dim_module, f"compute_{dim}")
                except Exception as e:
                    raise NotImplementedError(f"Dimension {dim} not implemented/importable: {e}")

                submods = submodules_dict[dim]
                _force_fp32_recursive(submods, device='cuda')

                print0(f"[VBench custom] running {dim} ...")

                results = compute_fn(meta_json, 'cuda', submods, **kwargs)
                results_dict[dim] = results
        out_path = os.path.join(self.output_path, f"{self.name}_eval_results.json")
        if get_rank() == 0:
            os.makedirs(self.output_path, exist_ok=True)
            save_json(results_dict, out_path)
            print0(f"Evaluation results saved to {out_path}")

        return results_dict


