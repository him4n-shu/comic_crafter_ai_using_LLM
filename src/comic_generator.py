import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
from src.utils import DEVICE, IMAGE_SIZE, DISTILGPT2_PATH, STABLE_DIFFUSION_PATH, CONTROLNET_PATH, validate_path, logger

class ComicGenerator:
    def __init__(self, character_library: Optional[Dict[str, str]] = None):
        self.tokenizer = None
        self.story_model = None
        self.image_pipe = None
        self.controlnet = None
        self.visual_elements = []
        self.character_library = character_library if character_library else {}
        self._setup_models()

    def _setup_models(self):
        """Load GPT-2 and ControlNet-based Stable Diffusion from local paths."""
        try:
            # GPU / CPU detection
            if DEVICE == "cuda":
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
            else:
                logger.warning("No GPU detected, using CPU fallback.")

            # Validate & load DistilGPT2
            validate_path(DISTILGPT2_PATH, "transformers")
            logger.info("Loading DistilGPT-2 from local path...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(DISTILGPT2_PATH)
            self.story_model = GPT2LMHeadModel.from_pretrained(DISTILGPT2_PATH).to(DEVICE)

            # Validate & load ControlNet
            validate_path(CONTROLNET_PATH, "controlnet")
            logger.info("Loading ControlNet (Canny)...")
            self.controlnet = ControlNetModel.from_pretrained(
                CONTROLNET_PATH,
                torch_dtype=torch.float16,
                local_files_only=True,
                use_safetensors=False
            )

            # Validate & load Stable Diffusion
            validate_path(STABLE_DIFFUSION_PATH, "diffusers")
            logger.info("Loading Stable Diffusion 2-1-base with ControlNet...")
            self.image_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                STABLE_DIFFUSION_PATH,
                controlnet=self.controlnet,
                local_files_only=True,
                use_safetensors=False,
                torch_dtype=torch.float16,
                safety_checker=None
            )

            # Memory optimizations
            self.image_pipe.enable_model_cpu_offload()
            try:
                self.image_pipe.enable_xformers_memory_efficient_attention()
                logger.info("xFormers memory efficient attention enabled.")
            except Exception as e:
                logger.warning(f"Could not enable xFormers: {str(e)}")

            logger.info("All models loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def generate_story(self, prompt: str) -> Tuple[List[str], List[str]]:
        story_prompts = [
            f"Introduction of {prompt}: Describe the main characters visually and the setting in detail.",
            f"Development of {prompt}: Describe the key action scenes and emotional expressions vividly.",
            f"Climax of {prompt}: Describe the most dramatic moment with detailed visual descriptions.",
            f"Conclusion of {prompt}: Describe the final scene with visual details and atmosphere."
        ]
        story_parts, highlights = [], []
        self.visual_elements = []
        running_context = ""

        for sp in story_prompts:
            full_prompt = f"{running_context}\n{sp}" if running_context else sp
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
            outputs = self.story_model.generate(
                inputs["input_ids"],
                max_new_tokens=100,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.1
            )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = text.split(sp)[-1].strip()
            running_context += " " + new_text
            story_parts.append(new_text)

            # Generate a short summary
            summary = self._summarize_part(new_text)
            highlights.append(summary)

            # For visuals
            sentences = [s.strip() for s in new_text.split(".") if s.strip()]
            visual = sentences[0] if sentences else new_text
            self.visual_elements.append(visual)

            logger.info(f"Generated story part -> {new_text[:60]}...")

        return story_parts, highlights

    def _summarize_part(self, text: str) -> str:
        """Generate a short summary (one sentence)."""
        try:
            prompt = f"Summarize in one short sentence: {text}"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
            outputs = self.story_model.generate(
                inputs["input_ids"],
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary.split("Summarize in one short sentence:")[-1].strip()
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return text[:40] + "..."

    def generate_image(self,
                       prompt: str,
                       user_sketch: Image.Image,
                       visual_element: str = None,
                       num_steps: int = 25,
                       guidance: float = 7.5,
                       controlnet_scale: float = 0.8) -> Image.Image:
        """Generate an image from the story part using ControlNet-based pipeline."""
        logger.info(f"Generating image for prompt: {prompt[:60]}...")
        if user_sketch is None:
            user_sketch = Image.new("RGB", IMAGE_SIZE, color="white")

        user_sketch = user_sketch.resize(IMAGE_SIZE)
        edges = cv2.Canny(np.array(user_sketch), 100, 200)
        canny_image = Image.fromarray(edges)

        # Build the prompt
        key_elements = self._extract_key_visuals(prompt)
        visual_context = visual_element if visual_element else prompt[:100]

        positive_prompt = (
            f"{key_elements}, {visual_context}, comic book panel, detailed illustration, vibrant colors, "
            "professional comic art style, digital art, crisp, high quality, cinematic lighting, no text, no lettering"
        )
        negative_prompt = (
            "text, letters, words, random characters, watermark, signature, blurry, low quality, distorted, "
            "bad anatomy, extra limbs, poorly drawn, cropped, worst quality, lowres"
        )
        logger.info(f"Enhanced prompt: {positive_prompt[:80]}...")

        try:
            image = self.image_pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                image=canny_image,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                controlnet_conditioning_scale=controlnet_scale
            ).images[0]
            return image
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return Image.new("RGB", IMAGE_SIZE, "red")

    def _extract_key_visuals(self, text: str) -> str:
        """Extract basic keywords for visuals from the text."""
        visual_terms = []
        known_visuals = [
            "hero", "villain", "castle", "forest", "city", "space", "dragon",
            "princess", "battle", "magic", "alien", "spaceship", "monster",
            "knight", "robot", "superhero", "flying", "running", "wizard"
        ]
        tokens = text.lower().split()
        for tok in tokens:
            if tok in known_visuals:
                visual_terms.append(tok)
        if visual_terms:
            return ", ".join(list(set(visual_terms)))
        return "character, scene"