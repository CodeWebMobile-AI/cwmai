import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from io import BytesIO

from PIL import Image

__description__ = "Generates thumbnail images from uploaded files with various sizes and formats."

__parameters__ = {
    "input_file": {
        "type": "str",
        "description": "Path to the input image file.",
        "required": True,
    },
    "output_dir": {
        "type": "str",
        "description": "Path to the output directory for thumbnails.",
        "required": True,
    },
    "sizes": {
        "type": "list",
        "description": "List of thumbnail sizes (width, height) as tuples. e.g., [(128, 128), (64, 64)]",
        "required": True,
    },
    "format": {
        "type": "str",
        "description": "Output image format (e.g., JPEG, PNG).",
        "default": "JPEG",
        "required": False,
    },
    "quality": {
        "type": "int",
        "description": "Quality of the output image (for JPEG). Range: 1-100",
        "default": 85,
        "required": False,
    },
}

__examples__ = [
    {
        "description": "Generate thumbnails of sizes 128x128 and 64x64 in JPEG format from input.png",
        "input_file": "input.png",
        "output_dir": "thumbnails",
        "sizes": "[(128, 128), (64, 64)]",
        "format": "JPEG",
        "quality": "85",
    },
]


async def image_thumbnail_generator(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates thumbnail images from an uploaded file with various sizes and formats.
    """
    try:
        input_file = kwargs["input_file"]
        output_dir = kwargs["output_dir"]
        sizes: List[tuple[int, int]] = eval(kwargs["sizes"])  #  eval necessary to convert from string to list of tuples
        format = kwargs.get("format", "JPEG")
        quality = kwargs.get("quality", 85)

        # Validate input
        if not os.path.exists(input_file):
            return {"status": "error", "message": f"Input file not found: {input_file}"}

        Path(output_dir).mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

        results = []
        with Image.open(input_file) as img:
            for width, height in sizes:
                try:
                    img_copy = img.copy()  # Create a copy for each size to avoid issues
                    img_copy.thumbnail((width, height))
                    output_filename = os.path.join(output_dir, f"thumbnail_{width}x{height}.{format.lower()}")

                    if format.upper() == "JPEG":
                         img_copy.save(output_filename, format=format.upper(), quality=quality)
                    else:
                        img_copy.save(output_filename, format=format.upper())
                    results.append(output_filename)
                except Exception as e:
                    return {"status": "error", "message": f"Error creating thumbnail {width}x{height}: {str(e)}"}

        return {"status": "success", "message": "Thumbnails generated successfully.", "thumbnails": results}

    except Exception as e:
        return {"status": "error", "message": f"An error occurred: {str(e)}"}