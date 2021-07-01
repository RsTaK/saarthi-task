import torch

class Config:
    """This class has all the parameters that we require to set"""
    globals_={
        "model": r"model\sst\checkpoint-4200",
        "processor": r"model\sst\processor",
        "base_dir": r"data",
        'device' : "cuda" if torch.cuda.is_available() else "cpu",
        'output_path': r"SST.csv"
    }