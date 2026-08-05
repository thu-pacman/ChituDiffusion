"""
评测子进程组（subgroup），可用于 num_videos < world_size 的情况
"""

import torch
import torch.distributed as dist
import pickle
from logging import getLogger

logger = getLogger(__name__)


_CURRENT_GROUP = None  # None 表示使用 dist.group.WORLD（默认全局组）

def set_group(group):
    global _CURRENT_GROUP
    _CURRENT_GROUP = group

def clear_group():
    global _CURRENT_GROUP
    _CURRENT_GROUP = None

def _resolve_group(group=None):
    if group is not None:
        return group
    if _CURRENT_GROUP is not None:
        return _CURRENT_GROUP
    return dist.group.WORLD


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank(group=None):
    if not is_dist_initialized():
        return 0
    g = _resolve_group(group)
    return dist.get_rank(group=g)


def get_world_size(group=None):
    if not is_dist_initialized():
        return 1
    g = _resolve_group(group)
    return dist.get_world_size(group=g)


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


def distribute_list_to_rank(data_list):
    rank = get_rank()
    world_size = get_world_size()
    return data_list[rank::world_size]



def all_gather(data, group=None, device=None):
    g = _resolve_group(group)
    world_size = get_world_size(group=g)
    if world_size == 1:
        return [data]


    origin_size = None
    if not isinstance(data, torch.Tensor):
        buffer = pickle.dumps(data)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).to("cuda")
    else:
        origin_size = data.size()
        tensor = data.reshape(-1)

    tensor_type = tensor.dtype


    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size, group=g)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.FloatTensor(size=(max_size,)).cuda().to(tensor_type))
    if local_size != max_size:
        padding = torch.FloatTensor(size=(max_size - local_size,)).cuda().to(tensor_type)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor, group=g)


    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        if origin_size is None:
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        else:
            buffer = tensor[:size]
            data_list.append(buffer)

    if origin_size is not None:
        new_shape = [-1] + list(origin_size[1:])
        resized_list = []
        for data in data_list:
            data = data.reshape(new_shape)
            resized_list.append(data)

        return resized_list
    else:
        return data_list


def merge_list_of_list(results):  
    results= [item for sublist in results for item in sublist]
    return results


def gather_list_of_dict(results):
    results = all_gather(results)
    results = merge_list_of_list(results)
    return results


def barrier(group=None):
    if is_dist_initialized():
        g = _resolve_group(group)
        dist.barrier(group=g)


def barrier_world():
    if is_dist_initialized():
        dist.barrier(group=dist.group.WORLD)


def dist_run(strategy, args, **kwargs):
    rank = get_rank()
    world_size = get_world_size()


    barrier_world()

    if rank == 0:
        payload = strategy.get_eval_videos(args, **kwargs)
    else:
        payload = None

    if world_size > 1 and dist.is_available() and dist.is_initialized():
        obj = [payload]
        dist.broadcast_object_list(obj, src=0)
        payload = obj[0]


    if payload is None:
        if rank == 0:
            logger.warning("Eval payload is empty, skip.")
        barrier_world()
        return None

    n = payload.get("num_eval_items")
    if n is None:
        video_prompt = payload.get("video_prompt", {})
        n = len(video_prompt)
    if not n or n <= 0:
        if rank == 0:
            logger.warning("No eval items, skip.")
        barrier_world()
        return None

    eval_world_size = min(world_size, n)


    eval_group = None
    if world_size > 1 and eval_world_size < world_size:
        eval_ranks = list(range(eval_world_size))
        eval_group = dist.new_group(ranks=eval_ranks)

    if rank >= eval_world_size:
        logger.info(f"[Rank {rank}] Skip eval (eval_world_size={eval_world_size}).")
        logger.info(f"eval: {payload}")
        return None

    if eval_group is not None:
        set_group(eval_group)
    else:
        clear_group()  

    try:
        result = strategy.evaluate(payload=payload, args=args, **kwargs)
        return result
    finally:
        clear_group()
