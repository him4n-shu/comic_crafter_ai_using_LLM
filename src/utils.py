import os
import logging
import torch  

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (512, 512)

# Model Paths
DISTILGPT2_PATH = r"C:\Users\himan\OneDrive\Desktop\comic_crafter_ai\models\distilgpt2".replace("\\", "/")
STABLE_DIFFUSION_PATH = r"C:\Users\himan\OneDrive\Desktop\comic_crafter_ai\models\stable-diffusion-2-1-base".replace("\\", "/")
CONTROLNET_PATH = r"C:\Users\himan\OneDrive\Desktop\comic_crafter_ai\models\controlnet-canny".replace("\\", "/")

def validate_path(path: str, model_type: str = "transformers") -> None:
    """Validate that a given path has the necessary files for the specified model type."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    if model_type == "transformers":
        required_files = ["config.json", "pytorch_model.bin", "vocab.json", "merges.txt"]
    elif model_type == "diffusers":
        required_files = ["model_index.json"]
        required_dirs = ["feature_extractor", "scheduler", "text_encoder", "tokenizer", "unet", "vae"]
    elif model_type == "controlnet":
        required_files = ["config.json", "diffusion_pytorch_model.bin"]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    for f in required_files:
        if not os.path.exists(os.path.join(path, f)):
            raise FileNotFoundError(f"Required file {f} not found in {path}")
    
    if model_type == "diffusers":
        for d in required_dirs:
            if not os.path.exists(os.path.join(path, d)):
                raise FileNotFoundError(f"Required directory {d} not found in {path}")

    logger.info(f"Path validation successful for {path}")