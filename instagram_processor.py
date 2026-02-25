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
INSTAGRAM_WIDTH = 2160  # Higher resolution (Instagram minimum 1080x1350)
INSTAGRAM_HEIGHT = 2700  # 4:5 ratio

# Quality settings (target ~2-3 MB)
DEFAULT_JPEG_QUALITY = 80  # ~80% gives 2-3 MB for 2160x2700


class InstagramProcessor:
    """Процессор фото для Instagram винтажной одежды."""

    # Пресет магазина - простая обработка
    SHOP_PRESET = {
        # Базовая коррекция
        "brightness": 10,        # Осветление
        "contrast": 1.15,        # 15% контраст
        # Тёплый винтаж
        "temperature": 5800,    # Тёплый оттенок
        # Кадрирование
        "vertical_offset_percent": 0.0,
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
        """Автоматически обрезать лишнее пространство вокруг объекта."""
        img_array = np.array(img)
        h, w = img_array.shape[:2]

        # Пробуем с OpenCV
        if CV2_AVAILABLE:
            try:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                blur = cv2.GaussianBlur(gray, (9, 9), 0)
                _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    x, y, cw, ch = cv2.boundingRect(largest)
                    margin_x = int(cw * 0.05)
                    margin_y = int(ch * 0.05)

                    new_x = max(0, x - margin_x)
                    new_y = max(0, y - margin_y)
                    new_w = min(w - new_x, cw + margin_x * 2)
                    new_h = min(h - new_y, ch + margin_y * 2)

                    if new_w > w * 0.3 and new_h > h * 0.3:
                        img_array = img_array[new_y:new_y+new_h, new_x:new_x+new_w]
                        return Image.fromarray(img_array)
            except Exception:
                pass

        # Запасной вариант - анализ яркости
        gray = np.mean(img_array, axis=2)
        brightness_per_row = gray.mean(axis=1)
        brightness_per_col = gray.mean(axis=0)
        avg = brightness_per_row.mean()

        # Находим границы где яркость резко меняется
        top = 0
        for i in range(h):
            if brightness_per_row[i] < avg * 0.5:
                top = i
                break

        bottom = h
        for i in range(h - 1, -1, -1):
            if brightness_per_row[i] < avg * 0.5:
                bottom = i + 1
                break

        left = 0
        for i in range(w):
            if brightness_per_col[i] < avg * 0.5:
                left = i
                break

        right = w
        for i in range(w - 1, -1, -1):
            if brightness_per_col[i] < avg * 0.5:
                right = i + 1
                break

        # Если есть что обрезать
        if (right - left) > w * 0.4 and (bottom - top) > h * 0.4:
            img_array = img_array[top:bottom, left:right]

        return Image.fromarray(img_array)

    def _center_crop_to_target(self, img: Image.Image, target_size: Tuple[int, int], vertical_offset_percent: float = 0.0) -> Image.Image:
        """Центрировать и обрезать до 4:5 с вертикальным смещением."""
        target_w, target_h = target_size
        img_w, img_h = img.size

        scale = max(target_w / img_w, target_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        left = (new_w - target_w) // 2
        # Calculate vertical offset in pixels
        offset_pixels = int(target_h * (vertical_offset_percent / 100.0))
        top = (new_h - target_h) // 2 + offset_pixels

        # Ensure top is within valid bounds
        top = max(0, min(top, new_h - target_h))

        return img.crop((left, top, left + target_w, top + target_h))

    def _apply_exposure(self, img: Image.Image, exposure: float) -> Image.Image:
        """Применить коррекцию экспозиции."""
        if exposure == 0:
            return img

        # Exposure в Photoshop примерно соответствует умножению значений
        # exposure=1.55 означает умножение на ~1.55
        factor = 2 ** exposure  # Конвертируем в линейный множитель
        img_array = np.array(img).astype(np.float32)
        img_array = img_array * factor
        img_array = np.clip(img_array, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))

    def _apply_highlights_shadows(self, img: Image.Image, highlights: int, shadows: int, blacks: int) -> Image.Image:
        """Применить коррекцию светов, теней и чёрных."""
        img_array = np.array(img).astype(np.float32)

        if len(img_array.shape) == 3:
            # Упрощённая коррекция - применяем ко всему изображению
            # с мягким коэффициентом
            factor = 0.3  # Мягкий коэффициент

            # Highlights: снижение ярких областей
            if highlights != 0:
                adjustment = highlights / 100.0 * factor
                img_array = img_array * (1 - adjustment * 0.3)

            # Shadows: усиление тёмных областей
            if shadows != 0:
                adjustment = shadows / 100.0 * factor
                img_array = img_array * (1 + adjustment * 0.3)

            # Blacks: сдвиг уровня чёрного
            if blacks != 0:
                adjustment = blacks / 100.0 * 20
                img_array = img_array - adjustment

        img_array = np.clip(img_array, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))

    def _apply_texture_clarity_dehaze(self, img: Image.Image, texture: int, clarity: int, dehaze: int) -> Image.Image:
        """Применить Texture, Clarity и Dehaze."""
        if not CV2_AVAILABLE:
            return img

        img_array = np.array(img)

        # Texture - усиление локального контраста в средних частотах
        if texture != 0:
            # Используем unsharp mask для имитации texture
            blur = cv2.GaussianBlur(img_array, (0, 0), 1.0)
            texture_amount = texture / 100.0 * 30
            img_array = np.clip(img_array + (img_array - blur) * texture_amount, 0, 255).astype(np.uint8)

        # Clarity - усиление контраста в средних тонах
        if clarity != 0:
            # Аналог: S-кривая для средних тонов
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0].astype(np.float32)

            # Применяем кривую
            clarity_factor = clarity / 100.0
            l_new = np.clip(l_channel * (1 + clarity_factor * 0.3), 0, 255).astype(np.uint8)
            lab[:, :, 0] = l_new
            img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Dehaze - устранение дымки
        if dehaze != 0:
            # Простой метод: увеличиваем контраст и насыщенность в зависимости от "дымки"
            # Находим среднюю яркость
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            avg_brightness = np.mean(gray)

            # Dehaze эффект
            dehaze_factor = dehaze / 100.0

            # Увеличиваем локальный контраст (CLAHE работает с grayscale)
            clahe = cv2.createCLAHE(clipLimit=2.0 + dehaze_factor * 3, tileGridSize=(8, 8))
            gray_enhanced = clahe.apply(gray)

            # Применяем к Lналу LAB для ка лучшего результата
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = gray_enhanced
            img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return Image.fromarray(img_array)

    def _apply_color_grading(self, img: Image.Image,
                              highlights_hue: int, highlights_sat: int,
                              midtones_hue: int, midtones_sat: int,
                              shadows_hue: int, shadows_sat: int,
                              temperature: int, tint: int) -> Image.Image:
        """Применить цветовую коррекцию: Temperature, Tint и Color Grading."""
        img_array = np.array(img).astype(np.float32)

        # Temperature - сдвиг от жёлтого к синему (в K)
        # 5500K означает тёплый оттенок (меньше = холоднее, больше = теплее)
        # Мы используем как сдвиг в жёлтый/оранжевый
        if temperature != 5500:  # baseline
            temp_adjust = (temperature - 5500) / 1000
            img_array[:, :, 0] += temp_adjust * 20  # Red
            img_array[:, :, 2] -= temp_adjust * 15  # Blue

        # Tint - сдвиг в зелёный/пурпурный
        if tint != 0:
            img_array[:, :, 1] += tint * 0.5  # Green

        # Color Grading - по областям (Highlights, Midtones, Shadows)
        # Для упрощения: применяем hue rotation к синему каналу для warm vintage эффекта

        # Вычисляем яркость для масок
        luminance = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        normalized = luminance / 255.0

        highlight_mask = np.clip(normalized - 0.5, 0, 1) * 2  # 0.5-1.0 -> 0-1
        shadow_mask = np.clip(0.5 - normalized, 0, 1) * 2      # 0.0-0.5 -> 0-1
        midtone_mask = 1 - highlight_mask - shadow_mask

        # Highlights: hue 45 = оранжевый, saturation 8
        if highlights_hue != 0 or highlights_sat != 0:
            warm_shift = highlights_hue / 100.0 * 0.3 + highlights_sat / 100.0 * 2
            img_array[:, :, 0] += highlight_mask * warm_shift * 20  # Red+
            img_array[:, :, 2] -= highlight_mask * warm_shift * 10  # Blue-

        # Midtones: hue 45 = оранжевый, saturation 6
        if midtones_hue != 0 or midtones_sat != 0:
            warm_shift = midtones_hue / 100.0 * 0.2 + midtones_sat / 100.0 * 1.5
            img_array[:, :, 0] += midtone_mask * warm_shift * 15
            img_array[:, :, 2] -= midtone_mask * warm_shift * 8

        # Shadows: hue 35 = более тёплый, saturation 10
        if shadows_hue != 0 or shadows_sat != 0:
            warm_shift = shadows_hue / 100.0 * 0.35 + shadows_sat / 100.0 * 2.5
            img_array[:, :, 0] += shadow_mask * warm_shift * 25
            img_array[:, :, 2] -= shadow_mask * warm_shift * 12

        img_array = np.clip(img_array, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))

    def process_image(
        self,
        image_path: str,
        preset: str = "shop_vintage",
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        target_size: Tuple[int, int] = (INSTAGRAM_WIDTH, INSTAGRAM_HEIGHT),
        center_crop: bool = True,
        vertical_offset_percent: float = 0.0,
        auto_fix_edges: bool = True,
        output_filename: Optional[str] = None,
        # Параметры коррекции (могут быть переопределены из UI)
        brightness: Optional[int] = None,
        contrast: Optional[float] = None,
        saturation: Optional[float] = None,
        temperature: Optional[int] = None
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
            # Всегда используем переданный vertical_offset_percent
            current_vertical_offset = vertical_offset_percent
            img = self._center_crop_to_target(img, target_size, current_vertical_offset)
            steps.append(f"center_crop_offset_{current_vertical_offset:.1f}%")

        # Apply коррекцию - используем переданные параметры
        # Конвертируем в numpy массив один раз
        img_array = np.array(img)

        # 1. Brightness (по умолчанию 20 для заметного эффекта)
        use_brightness = brightness if brightness is not None else 20
        if use_brightness != 0:
            img_array = img_array.astype(np.float32)
            img_array = img_array + use_brightness
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            steps.append(f"brightness_{use_brightness}")

        # 2. Contrast (по умолчанию 1.3)
        use_contrast = contrast if contrast is not None else 1.3
        if use_contrast != 1.0:
            img_array = img_array.astype(np.float32)
            img_array = ((img_array - 128) * use_contrast) + 128
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            steps.append(f"contrast_{use_contrast}")

        # 3. Saturation (по умолчанию 1.1)
        use_saturation = saturation if saturation is not None else 1.1
        if use_saturation != 1.0:
            # Работаем с img напрямую
            hsv = img.convert('HSV')
            hsv_array = np.array(hsv)
            # Умножаем S канал
            hsv_array[:, :, 1] = np.clip(hsv_array[:, :, 1] * use_saturation, 0, 255).astype(np.uint8)
            img = Image.fromarray(hsv_array).convert('RGB')
            steps.append(f"saturation_{use_saturation}")

        # 4. Temperature (по умолчанию 6000 - тёплый винтаж)
        use_temp = temperature if temperature is not None else 6000
        if use_temp != 5500:
            img_array = img_array.astype(np.float32)
            temp_adjust = (use_temp - 5500) / 1000
            img_array[:, :, 0] += temp_adjust * 20  # Red+
            img_array[:, :, 2] -= temp_adjust * 15  # Blue-
            img_array = np.clip(img_array, 0, 255)
            img = Image.fromarray(img_array.astype(np.uint8))
            steps.append(f"temperature_{use_temp}")

        # Generate output filename
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"instagram_{original_path.stem}_{preset}_{timestamp}.jpg"

        output_path = self.output_dir / output_filename

        # Save with quality
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
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    vertical_offset_percent: float = 0.0,
    output_filename: Optional[str] = None
) -> Dict[str, Any]:
    """Удобная функция для обработки одного изображения."""
    processor = InstagramProcessor()
    return processor.process_image(
        image_path,
        preset=preset,
        jpeg_quality=jpeg_quality,
        vertical_offset_percent=vertical_offset_percent,
        output_filename=output_filename
    )


if __name__ == "__main__":
    print("Instagram Vintage Photo Processor - Professional")
    print(f"RAW support: {RAW_AVAILABLE}")
    print(f"OpenCV: {CV2_AVAILABLE}")
    print(f"Default size: {INSTAGRAM_WIDTH}x{INSTAGRAM_HEIGHT}")
    print(f"Default quality: {DEFAULT_JPEG_QUALITY}%")
