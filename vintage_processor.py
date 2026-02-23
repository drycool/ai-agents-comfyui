"""Classic Image Processor for Instagram Vintage Clothing.

Классическая обработка изображений для Instagram магазина винтажной одежды.
Использует Pillow + OpenCV для быстрой и предсказуемой обработки.
"""
import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

# Image processing libraries
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import numpy as np

# Try to import OpenCV (optional)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Some features limited.")

# Configuration
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output/instagram")
INSTAGRAM_WIDTH = 1080
INSTAGRAM_HEIGHT = 1350  # 4:5 ratio

# Color presets for vintage clothing
COLOR_PRESETS = {
    "warm_vintage": {
        "brightness": 1.05,
        "contrast": 1.1,
        "saturation": 0.9,
        "sharpness": 1.2,
        "color_temp": "warm",
        "sepia": 0.05
    },
    "cool_vintage": {
        "brightness": 1.0,
        "contrast": 1.05,
        "saturation": 0.85,
        "sharpness": 1.15,
        "color_temp": "cool",
        "sepia": 0.0
    },
    "sepia_vintage": {
        "brightness": 1.02,
        "contrast": 1.15,
        "saturation": 0.7,
        "sharpness": 1.1,
        "color_temp": "warm",
        "sepia": 0.15
    },
    "neutral": {
        "brightness": 1.0,
        "contrast": 1.0,
        "saturation": 1.0,
        "sharpness": 1.0,
        "color_temp": "neutral",
        "sepia": 0.0
    },
    "minimal": {
        "brightness": 1.0,
        "contrast": 1.02,
        "saturation": 0.95,
        "sharpness": 1.05,
        "color_temp": "neutral",
        "sepia": 0.0
    }
}

# Background colors
BACKGROUND_COLORS = {
    "white": (255, 255, 255),
    "cream": (245, 240, 230),
    "light_gray": (240, 240, 240),
    "warm_white": (255, 250, 245)
}


class VintagePhotoProcessor:
    """Процессор фото для Instagram винтажной одежды."""

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_image(
        self,
        image_path: str,
        preset: str = "warm_vintage",
        background_color: str = "cream",
        target_size: Tuple[int, int] = (INSTAGRAM_WIDTH, INSTAGRAM_HEIGHT),
        center_crop: bool = True,
        remove_background: bool = False,
        denoise: bool = True,
        retouch: bool = False,
        auto_fix_edges: bool = True,
        output_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Обработать изображение для Instagram.

        Args:
            image_path: Путь к исходному изображению
            preset: Цветовой пресет
            background_color: Цвет фона
            target_size: Целевой размер (по умолчанию 1080x1350)
            center_crop: Центрировать и обрезать
            remove_background: Удалить фон (упрощённый метод)
            denoise: Уменьшить шум
            retouch: Применить лёгкую ретушь
            auto_fix_edges: Автоудаление тёмных краёв/артефактов
            output_filename: Имя выходного файла

        Returns:
            Dict с результатами обработки
        """
        start_time = datetime.now()
        original_path = Path(image_path)

        if not original_path.exists():
            return {"error": f"File not found: {image_path}"}

        # Load image
        img = Image.open(original_path)
        original_mode = img.mode
        if img.mode != "RGB":
            img = img.convert("RGB")

        steps = []

        # Step 1: Auto fix edges (dark bands, vignette)
        if auto_fix_edges:
            img = self._auto_fix_dark_edges(img)
            steps.append("auto_fix_edges")

        # Step 2: Denoise (if enabled)
        if denoise:
            img = self._apply_denoise(img)
            steps.append("denoise")

        # Step 3: Retouch (if enabled)
        if retouch:
            img = self._apply_retouch(img)
            steps.append("retouch")

        # Step 4: Remove background (if enabled)
        if remove_background:
            img = self._remove_background_simple(img, BACKGROUND_COLORS.get(background_color, (245, 240, 230)))
            steps.append("remove_background")

        # Step 5: Resize and crop to 4:5
        if center_crop:
            img = self._center_crop_to_target(img, target_size)
            steps.append("center_crop")

        # Step 6: Apply color preset
        img = self._apply_preset(img, preset)
        steps.append(f"preset_{preset}")

        # Step 7: Final resize to exact target
        img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Generate output filename
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"instagram_{original_path.stem}_{preset}_{timestamp}.jpg"

        output_path = self.output_dir / output_filename

        # Save with optimal quality
        img.save(output_path, "JPEG", quality=95, optimize=True)

        processing_time = (datetime.now() - start_time).total_seconds()

        return {
            "status": "success",
            "input": str(original_path),
            "output": str(output_path),
            "preset": preset,
            "background": background_color,
            "size": target_size,
            "steps": steps,
            "processing_time": processing_time
        }

    def _auto_fix_dark_edges(self, img: Image.Image) -> Image.Image:
        """Автоматически обнаружить и исправить тёмные края/полосы по бокам."""
        if not CV2_AVAILABLE:
            return img

        img_array = np.array(img)
        h, w = img_array.shape[:2]

        # Calculate average brightness per column
        brightness_per_col = img_array.mean(axis=(0, 2))

        # Calculate overall average brightness
        avg_brightness = brightness_per_col.mean()
        threshold = avg_brightness * 0.5  # Dark if < 50% of average

        # Find dark columns on left and right
        dark_left_cols = 0
        dark_right_cols = 0

        # Check left side
        for i in range(min(w // 10, 200)):  # Check up to 10% or 200 cols
            if brightness_per_col[i] < threshold:
                dark_left_cols = i + 1
            else:
                break

        # Check right side
        for i in range(w - 1, max(w - w // 10 - 1, w - 201), -1):
            if brightness_per_col[i] < threshold:
                dark_right_cols = w - i
            else:
                break

        # Crop if dark edges detected
        if dark_left_cols > 10 or dark_right_cols > 10:
            new_left = dark_left_cols
            new_right = w - dark_right_cols
            if new_left < new_right and (new_right - new_left) > w * 0.5:
                img_array = img_array[:, new_left:new_right]
                img = Image.fromarray(img_array)

        # Also check for dark bands in the middle (like the one we see)
        # Check if there's a vertical dark band
        center_region = w // 4
        left_brightness = brightness_per_col[:center_region].mean()
        center_brightness = brightness_per_col[center_region:w-center_region].mean()
        right_brightness = brightness_per_col[w-center_region:].mean()

        # If center is significantly darker, might be a band
        if center_brightness < avg_brightness * 0.7:
            # Find the dark region
            dark_band_start = 0
            dark_band_end = w
            for i in range(center_region, w - center_region):
                if brightness_per_col[i] < avg_brightness * 0.5:
                    if dark_band_start == 0:
                        dark_band_start = i
                    dark_band_end = i

            # If it's a narrow band (< 15% of width), try to fix it
            if 0 < dark_band_start < dark_band_end and (dark_band_end - dark_band_start) < w * 0.15:
                # Replace dark band with interpolated pixels from edges
                band_width = dark_band_end - dark_band_start
                for i in range(band_width):
                    alpha = i / band_width
                    # Interpolate between left and right edge
                    left_pixel = img_array[:, dark_band_start - 1]
                    right_pixel = img_array[:, dark_band_end]
                    img_array[:, dark_band_start + i] = (
                        (1 - alpha) * left_pixel + alpha * right_pixel
                    ).astype(np.uint8)
                img = Image.fromarray(img_array)

        return img

    def _apply_denoise(self, img: Image.Image) -> Image.Image:
        """Применить шумоподавление."""
        if CV2_AVAILABLE:
            # Convert to numpy array
            img_array = np.array(img)

            # Apply denoising
            denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)

            return Image.fromarray(denoised)
        else:
            # Fallback to PIL filter
            return img.filter(ImageFilter.SMOOTH)

    def _apply_retouch(self, img: Image.Image) -> Image.Image:
        """Применить лёгкую ретушь."""
        # Apply unsharp mask for slight sharpening
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=3))

        # Slight smoothing
        img = img.filter(ImageFilter.SMOOTH)

        return img

    def _remove_background_simple(self, img: Image.Image, bg_color: Tuple[int, int, int]) -> Image.Image:
        """Упрощённое удаление фона (пороговая обработка)."""
        if not CV2_AVAILABLE:
            # Just add background color if no OpenCV
            background = Image.new("RGB", img.size, bg_color)
            return Image.composite(img, background, img.convert("L").point(lambda x: 255 if x > 240 else 0))

        img_array = np.array(img)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

        # Calculate brightness
        brightness = hsv[:, :, 2]

        # Create mask for very bright/white pixels (likely background)
        _, bright_mask = cv2.threshold(brightness, 240, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations to clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Invert mask
        mask = 255 - mask

        # Create output with new background
        result = img_array.copy()
        result[mask == 0] = bg_color

        return Image.fromarray(result)

    def _center_crop_to_target(self, img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Центрировать и обрезать изображение."""
        target_width, target_height = target_size
        img_width, img_height = img.size

        # Calculate scaling factor (cover the target)
        scale = max(target_width / img_width, target_height / img_height)

        # Resize while maintaining aspect ratio
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Calculate crop box (center)
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height

        return img.crop((left, top, right, bottom))

    def _apply_preset(self, img: Image.Image, preset: str) -> Image.Image:
        """Применить цветовой пресет."""
        settings = COLOR_PRESETS.get(preset, COLOR_PRESETS["neutral"])

        # Brightness
        if settings.get("brightness", 1.0) != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(settings["brightness"])

        # Contrast
        if settings.get("contrast", 1.0) != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(settings["contrast"])

        # Saturation
        if settings.get("saturation", 1.0) != 1.0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(settings["saturation"])

        # Sharpness
        if settings.get("sharpness", 1.0) != 1.0:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(settings["sharpness"])

        # Sepia tone (warm vintage effect)
        sepia_amount = settings.get("sepia", 0.0)
        if sepia_amount > 0:
            img = self._apply_sepia(img, sepia_amount)

        return img

    def _apply_sepia(self, img: Image.Image, amount: float = 0.1) -> Image.Image:
        """Применить сепию."""
        # Convert to numpy
        img_array = np.array(img).astype(np.float32)

        # Sepia transformation matrix
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])

        # Apply transformation
        sepia = np.dot(img_array, sepia_matrix.T)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)

        # Blend with original
        sepia_img = Image.fromarray(sepia)
        return Image.blend(img, sepia_img, amount)

    def batch_process(
        self,
        image_paths: List[str],
        preset: str = "warm_vintage",
        background_color: str = "cream",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Пакетная обработка нескольких изображений."""
        results = []
        for path in image_paths:
            result = self.process_image(
                path,
                preset=preset,
                background_color=background_color,
                **kwargs
            )
            results.append(result)
        return results


def process_single(
    image_path: str,
    preset: str = "warm_vintage",
    background_color: str = "cream"
) -> Dict[str, Any]:
    """Удобная функция для обработки одного изображения."""
    processor = VintagePhotoProcessor()
    return processor.process_image(
        image_path,
        preset=preset,
        background_color=background_color
    )


def list_presets() -> List[str]:
    """Список доступных пресетов."""
    return list(COLOR_PRESETS.keys())


def list_backgrounds() -> List[str]:
    """Список доступных цветов фона."""
    return list(BACKGROUND_COLORS.keys())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Instagram Vintage Photo Processor")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--preset", "-p", default="warm_vintage", choices=list_presets(),
                        help="Color preset")
    parser.add_argument("--background", "-b", default="cream", choices=list_backgrounds(),
                        help="Background color")
    parser.add_argument("--output", "-o", help="Output filename")

    args = parser.parse_args()

    result = process_single(args.image, args.preset, args.background)
    print(f"Result: {result}")
