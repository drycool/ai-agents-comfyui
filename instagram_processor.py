"""Instagram Photo Processor for Vintage Clothing - Professional Edition.

Поддержка NEF (Nikon RAW), точные параметры Photoshop.
"""
import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

# Image processing
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import numpy as np

# RAW processing
try:
    import rawpy
    RAW_AVAILABLE = True
except ImportError:
    RAW_AVAILABLE = False

# OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Configuration
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output/instagram")
INSTAGRAM_WIDTH = 2160  # Higher resolution
INSTAGRAM_HEIGHT = 2700  # 4:5 ratio

# Quality settings
DEFAULT_JPEG_QUALITY = 100


class InstagramProcessor:
    """Процессор фото для Instagram винтажной одежды."""

    # Пресет магазина (параметры из Photoshop)
    SHOP_PRESET = {
        "brightness_offset": -2,  # Less dark (was -5)
        "contrast_factor": 1.30,  # 30% increase
        "color_balance_shadows": (0, 0, -8),
        "color_balance_midtones": (0, 0, -5),
        "color_balance_highlights": (0, 0, -3),
    }

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_image(self, image_path: str) -> Image.Image:
        """Загрузить изображение (NEF, TIFF, JPEG, PNG)."""
        path = Path(image_path)
        ext = path.suffix.upper()

        if ext == ".NEF":
            return self._load_nef(image_path)
        else:
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img

    def _load_nef(self, nef_path: str) -> Image.Image:
        """Загрузить NEF файл с максимальным качеством."""
        if not RAW_AVAILABLE:
            raise ImportError("rawpy not installed. Run: pip install rawpy")

        with rawpy.imread(nef_path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=16,
                gamma=(2.222, 4.5),
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            )

        rgb_8 = (rgb / 256).astype(np.uint8)
        return Image.fromarray(rgb_8)

    def _auto_fix_dark_edges(self, img: Image.Image) -> Image.Image:
        """Автоматически обнаружить и исправить тёмные края."""
        if not CV2_AVAILABLE:
            return img

        img_array = np.array(img)
        h, w = img_array.shape[:2]

        brightness_per_col = img_array.mean(axis=(0, 2))
        avg_brightness = brightness_per_col.mean()
        threshold = avg_brightness * 0.5

        dark_left = 0
        dark_right = 0

        for i in range(min(w // 10, 150)):
            if brightness_per_col[i] < threshold:
                dark_left = i + 1
            else:
                break

        for i in range(w - 1, max(w - w // 10 - 1, w - 151), -1):
            if brightness_per_col[i] < threshold:
                dark_right = w - i
            else:
                break

        if dark_left > 10 or dark_right > 10:
            new_left = dark_left
            new_right = w - dark_right
            if new_left < new_right and (new_right - new_left) > w * 0.5:
                img_array = img_array[:, new_left:new_right]

        return Image.fromarray(img_array)

    def _center_crop_to_target(self, img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Центрировать и обрезать до 4:5."""
        target_w, target_h = target_size
        img_w, img_h = img.size

        scale = max(target_w / img_w, target_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        return img.crop((left, top, left + target_w, top + target_h))

    def process_image(
        self,
        image_path: str,
        preset: str = "shop_vintage",
        jpeg_quality: int = 100,
        target_size: Tuple[int, int] = (INSTAGRAM_WIDTH, INSTAGRAM_HEIGHT),
        center_crop: bool = True,
        auto_fix_edges: bool = True,
        output_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """Обработать изображение для Instagram."""
        start_time = datetime.now()
        original_path = Path(image_path)

        if not original_path.exists():
            return {"error": f"File not found: {image_path}"}

        # Load image
        img = self.load_image(str(original_path))
        steps = []

        # Auto fix edges
        if auto_fix_edges:
            img = self._auto_fix_dark_edges(img)
            steps.append("auto_fix_edges")

        # Center crop to 4:5
        if center_crop:
            img = self._center_crop_to_target(img, target_size)
            steps.append("center_crop")

        # Apply shop preset
        if preset == "shop_vintage":
            brightness = self.SHOP_PRESET["brightness_offset"]
            if brightness != 0:
                img_array = np.array(img).astype(np.float32)
                img_array = img_array - abs(brightness)
                img_array = np.clip(img_array, 0, 255)
                img = Image.fromarray(img_array.astype(np.uint8))
                steps.append("brightness")

            contrast = self.SHOP_PRESET["contrast_factor"]
            if contrast != 1.0:
                img_array = np.array(img).astype(np.float32)
                img_array = ((img_array - 128) * contrast) + 128
                img_array = np.clip(img_array, 0, 255)
                img = Image.fromarray(img_array.astype(np.uint8))
                steps.append("contrast")

            # Color Balance
            img_array = np.array(img).astype(np.float32)
            img_array[:, :, 2] -= 8
            img_array[:, :, 2] -= 5
            img_array[:, :, 2] -= 3
            img_array = np.clip(img_array, 0, 255)
            img = Image.fromarray(img_array.astype(np.uint8))
            steps.append("color_balance")

        # Generate output filename
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"instagram_{original_path.stem}_{preset}_{timestamp}.jpg"

        output_path = self.output_dir / output_filename

        # Save with max quality
        img.save(output_path, "JPEG", quality=jpeg_quality, optimize=True)

        processing_time = (datetime.now() - start_time).total_seconds()

        return {
            "status": "success",
            "input": str(original_path),
            "output": str(output_path),
            "preset": preset,
            "quality": jpeg_quality,
            "size": target_size,
            "steps": steps,
            "processing_time": processing_time,
            "file_size": output_path.stat().st_size
        }


def process_single(
    image_path: str,
    preset: str = "shop_vintage",
    jpeg_quality: int = 100
) -> Dict[str, Any]:
    """Удобная функция для обработки одного изображения."""
    processor = InstagramProcessor()
    return processor.process_image(
        image_path,
        preset=preset,
        jpeg_quality=jpeg_quality
    )


if __name__ == "__main__":
    print("Instagram Vintage Photo Processor - Professional")
    print(f"RAW support: {RAW_AVAILABLE}")
    print(f"OpenCV: {CV2_AVAILABLE}")
    print(f"Default size: {INSTAGRAM_WIDTH}x{INSTAGRAM_HEIGHT}")
    print(f"Default quality: {DEFAULT_JPEG_QUALITY}%")
