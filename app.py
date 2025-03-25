import os 
import sys
import torch
from src.ui import build_interface
from src.utils import logger

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.backends.cudnn.benchmark = True 
    interface = build_interface()
    interface.launch()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application stopped by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        sys.exit(1)