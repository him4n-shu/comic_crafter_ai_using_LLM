import json
from typing import Tuple
from PIL import Image
from src.comic_generator import ComicGenerator
from src.utils import logger

def generate_comic(prompt: str,
                   user_sketch: Image.Image,
                   num_steps: int,
                   guidance: float,
                   controlnet_scale: float,
                   char_lib_text: str) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    logger.info(f"Starting comic generation for prompt: {prompt}")
    try:
        char_lib = json.loads(char_lib_text) if char_lib_text.strip() else {}
    except Exception as e:
        logger.warning(f"Failed to parse character library, ignoring. Error: {e}")
        char_lib = {}

    generator = ComicGenerator(character_library=char_lib)
    story_parts, highlights = generator.generate_story(prompt)

    panels = []
    for i, part in enumerate(story_parts):
        visual_element = generator.visual_elements[i] if i < len(generator.visual_elements) else None
        image = generator.generate_image(
            part, user_sketch, visual_element,
            num_steps=num_steps, guidance=guidance, controlnet_scale=controlnet_scale
        )
        panels.append(image)

    logger.info("Comic generation completed.")
    return panels[0], panels[1], panels[2], panels[3]