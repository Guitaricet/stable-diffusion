import torch
import torch.distributed as dist


def is_distributed():
    return dist.is_available() and dist.is_initialized()

def barrier():
    """Barrier for all processes."""
    if is_distributed(): dist.barrier()


def get_world_size():
    """Get the number of processes."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank():
    """Get the rank of the current process."""
    if is_distributed():
        return dist.get_rank()
    return 0

def gather_tensor(x: torch.Tensor):
    if not is_distributed(): return x

    gather_list = None
    if get_rank() == 0:
        gather_list = [torch.zeros_like(x) for _ in range(get_world_size())]
    dist.gather(x, gather_list=gather_list, dst=0)

    if get_rank() == 0:
        return torch.cat(gather_list, dim=0)


def gather_object(x):
    if not is_distributed(): return [x]
    gather_list = None
    if get_rank() == 0:
        gather_list = [None for _ in range(get_world_size())]

    dist.gather_object(x, object_gather_list=gather_list, dst=0)

    if get_rank() == 0:
        if isinstance(gather_list[0], list):
            return [item for sublist in gather_list for item in sublist]
        return gather_list


def all_gather_object(x):
    if not is_distributed(): return [x]
    gather_list = [None for _ in range(get_world_size())]
    dist.all_gather_object(obj=x, object_list=gather_list)

    gather_list = [item for sublist in gather_list for item in sublist]
    return gather_list
