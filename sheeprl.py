from sheeprl.cli import run
import torch
import multiprocessing.shared_memory as shm

if __name__ == "__main__":
    shared_memory = shm.SharedMemory(name='my_shared_memory', create=True, size=1)
    shared_memory.buf[0] = 0
    # https://github.com/pytorch/pytorch/issues/40403
    # Cannot re-initialize CUDA in forked subprocess
    torch.multiprocessing.set_start_method('spawn')
    run()
