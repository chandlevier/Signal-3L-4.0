import os
import gc
import torch
import pickle
import logging


def setup_logger(old_epoch, output_dir: str = None):
    logger = logging.getLogger(__name__)
    
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        
        c_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%y/%m/%d %H:%M:%S",
        )
        c_handler.setFormatter(formatter)
        logger.addHandler(c_handler)

        if output_dir is not None:
            f_handler = logging.FileHandler(os.path.join(output_dir, f"log_{old_epoch}.log"))
            f_handler.setFormatter(formatter)
            logger.addHandler(f_handler)
    
    return logger

def set_randomseed(seed: int = None):
  if seed is not None:
    torch.manual_seed(seed)
    return seed
  else:
    return torch.seed()

def save_mydict(dict, name):
    f_save = open(name + '.pkl', 'wb')
    pickle.dump(dict, f_save)
    f_save.close()

def load_mydict(name):
    f_read = open(name + '.pkl', 'rb')
    dict2 = pickle.load(f_read)
    f_read.close()
    return dict2

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()




