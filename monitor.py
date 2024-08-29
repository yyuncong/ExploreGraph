# monitor the usage of GPU within the system
import pynvml

def get_gpu_memory_usage(rank):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return mem_info.used

def log_gpu_memory_usage(rank, stage):
    usage = get_gpu_memory_usage(rank)
    print(f"[Rank {rank} - {stage}] GPU Memory Usage: {usage / (1024 ** 2):.2f} MB")
