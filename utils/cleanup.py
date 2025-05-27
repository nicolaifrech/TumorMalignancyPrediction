import gc
import torch

from utils.printing import print_verbose

def clear_gpu_cache(verbose=True):
    """
    Frees up any cached GPU memory and runs IPC cleanup.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Cleans up inter-process handles (useful if you fork processes)
        torch.cuda.ipc_collect()
        print_verbose("🧹 Cleared GPU cache.")
    else: 
        print_verbose("⚠️ No CUDA GPU detected; skipping cache clear.")

def run_gc(verbose=True):
    """
    Runs Python garbage collection to free up CPU RAM.
    """
    collected = gc.collect() 
    print_verbose(f"🗑️  Garbage collector freed {collected} objects.")
