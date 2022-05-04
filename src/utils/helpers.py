import torch

def get_device(gpuID):
    if not gpuID:
        return "cpu"

    if torch.cuda.is_available():
        device = "cuda:" + str(gpuID)
    else:
        device = "cpu"
    return device
