import torch

def get_best_device():
    if torch.cuda.is_available():   # NVIDIA
        return "cuda"
    if torch.backends.mps.is_available():  # Mac M1/M2 GPU
        return "mps"
    return "cpu"