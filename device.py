import os
import torch
import subprocess

def setup_device(use_gpu_1_only=False, enable_dataparallel=True):
    """
    Set up GPU device(s) for training.

    Args:
        use_gpu_1_only (bool): If True, restricts visible GPUs to only GPU 1.
        enable_dataparallel (bool): If True and multiple GPUs are available, wraps model in DataParallel.

    Returns:
        device (torch.device): The torch device ("cuda" or "cpu")
        dataparallel_fn (Callable): A function to wrap your model in nn.DataParallel if needed
    """
    if use_gpu_1_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 1 becomes cuda:0
        print("🔧 Using only GPU 1")

    # Check actual availability after applying env restrictions
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✅ CUDA is available: {device_count} GPU(s) visible")

        def dataparallel_fn(model):
            if enable_dataparallel and device_count > 1:
                print(f"🚀 Wrapping model in DataParallel over {device_count} GPUs")
                return torch.nn.DataParallel(model)
            return model

        return torch.device("cuda"), dataparallel_fn

    else:
        print("⚠️ No CUDA device available, using CPU")
        def identity(model): return model
        return torch.device("cpu"), identity
