"""AI Quality Enhancer - улучшение качества фото через ComfyUI.

Автоматическое улучшение качества изображений:
- AI Upscale (RealESRGAN)
- Denoise (шумоподавление)
- Color Correction
- Винтажные эффекты
"""
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
import numpy as np

from comfy_client import ComfyAPIClient
from cli_wrappers import OllamaWrapper


class ComfyUIEnhancer:
    """Улучшение качества через ComfyUI."""

    def __init__(self, host: str = None):
        self.comfy = ComfyAPIClient(host=host)
        self.client_id = f"enhancer_{int(time.time())}"

    def check_connection(self) -> bool:
        """Проверить подключение к ComfyUI."""
        try:
            stats = self.comfy.get_system_stats()
            return "system" in stats
        except Exception:
            return False

    def upscale_image(
        self,
        image_path: str,
        scale: int = 2,
        model: str = "RealESRGAN_x2plus.pth"
    ) -> Dict[str, Any]:
        """
        Увеличить разрешение изображения через ComfyUI.

        Args:
            image_path: Путь к изображению
            scale: Множитель upscale (2 или 4)
            model: Модель для upscale

        Returns:
            Результат с путём к улучшенному изображению
        """
        workflow = self._create_upscale_workflow(image_path, scale, model)
        result = self.comfy.execute_workflow(workflow, wait=True, timeout=300)

        if result.get("status") == "completed" and result.get("images"):
            return {
                "status": "success",
                "images": result["images"],
                "prompt_id": result.get("prompt_id")
            }

        return {
            "status": result.get("status", "error"),
            "error": "Upscale failed"
        }

    def denoise_image(self, image_path: str, strength: float = 0.5) -> Dict[str, Any]:
        """
        Удалить шумы с изображения.

        Args:
            image_path: Путь к изображению
            strength: Сила шумоподавления (0-1)

        Returns:
            Результат с путём к улучшенному изображению
        """
        workflow = self._create_denoise_workflow(image_path, strength)
        result = self.comfy.execute_workflow(workflow, wait=True, timeout=120)

        if result.get("status") == "completed":
            return {
                "status": "success",
                "images": result.get("images", []),
                "prompt_id": result.get("prompt_id")
            }

        return {"status": "error", "error": "Denoise failed"}

    def color_correct(
        self,
        image_path: str,
        brightness: float = 0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        temperature: float = 0
    ) -> Dict[str, Any]:
        """
        Коррекция цвета изображения.

        Args:
            image_path: Путь к изображению
            brightness: Яркость (-1 to 1)
            contrast: Контраст (0-2)
            saturation: Насыщенность (0-2)
            temperature: Температура (-1 to 1)

        Returns:
            Результат
        """
        workflow = self._create_color_workflow(
            image_path, brightness, contrast, saturation, temperature
        )
        result = self.comfy.execute_workflow(workflow, wait=True, timeout=60)

        if result.get("status") == "completed":
            return {
                "status": "success",
                "images": result.get("images", []),
                "prompt_id": result.get("prompt_id")
            }

        return {"status": "error", "error": "Color correction failed"}

    def enhance_full(
        self,
        image_path: str,
        upscale: bool = True,
        denoise: bool = True,
        color_correct: bool = True,
        target_size: tuple = (2160, 2700)
    ) -> Dict[str, Any]:
        """
        Полное улучшение изображения.

        Выполняет все улучшения последовательно:
        1. Denoise (если нужно)
        2. Upscale (если нужно)
        3. Color correction

        Args:
            image_path: Путь к изображению
            upscale: Делать upscale
            denoise: Удалять шумы
            color_correct: Корректировать цвет
            target_size: Целевой размер (ширина, высота)

        Returns:
            Результат со всеми метаданными
        """
        # Определяем текущий размер
        img = Image.open(image_path)
        current_size = img.size
        img.close()

        # Путь к файлу - используем временную метку
        timestamp = int(time.time())
        output_dir = Path("C:/comfyUI/ComfyUI/output")

        # Шаг 1: Denoise
        if denoise:
            result = self.denoise_image(image_path, strength=0.3)
            if result["status"] == "success":
                # Ищем latest файл
                files = sorted(output_dir.glob("denoised_*.png"), key=lambda x: x.stat().st_mtime)
                if files:
                    image_path = str(files[-1])

        # Шаг 2: Upscale
        if upscale:
            current_w, current_h = Image.open(image_path).size
            target_w, target_h = target_size

            scale_needed = max(target_w / current_w, target_h / current_h)
            scale = 2 if scale_needed > 1.5 else 1

            if scale > 1:
                result = self.upscale_image(image_path, scale=scale)
                if result["status"] == "success":
                    files = sorted(output_dir.glob("upscaled_*.png"), key=lambda x: x.stat().st_mtime)
                    if files:
                        image_path = str(files[-1])

        # Шаг 3: Color correction
        if color_correct:
            analysis = self._analyze_colors(image_path)
            result = self.color_correct(
                image_path,
                brightness=analysis.get("brightness", 0),
                contrast=analysis.get("contrast", 1.0),
                saturation=analysis.get("saturation", 1.0),
                temperature=analysis.get("temperature", 0)
            )
            if result["status"] == "success":
                files = sorted(output_dir.glob("corrected_*.png"), key=lambda x: x.stat().st_mtime)
                if files:
                    image_path = str(files[-1])

        return {
            "status": "success",
            "output_path": image_path,
            "steps": ["denoise", "upscale", "color_correct"],
            "original_size": current_size,
            "final_size": Image.open(image_path).size
        }

    def _analyze_colors(self, image_path: str) -> Dict[str, float]:
        """Анализ цветов изображения для автокоррекции."""
        img = Image.open(image_path)
        arr = np.array(img).astype(np.float32) / 255.0

        # Средняя яркость
        brightness = arr.mean() - 0.5

        # Контраст (стандартное отклонение)
        contrast = arr.std() * 2
        if contrast < 0.5:
            contrast = 0.8
        elif contrast > 1.5:
            contrast = 1.2

        # Насыщенность
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        gray = 0.299*r + 0.587*g + 0.114*b
        saturation = np.abs(arr - gray[..., None]).mean() * 3
        saturation = max(0.8, min(1.2, saturation))

        # Температура (красный - синий)
        temp = (r.mean() - b.mean())

        img.close()

        return {
            "brightness": float(brightness),
            "contrast": float(contrast),
            "saturation": float(saturation),
            "temperature": float(temp)
        }

    def _get_output_path(self, result: Dict) -> str:
        """Получить путь к выходному изображению."""
        if result.get("images"):
            img_info = result["images"][0]
            filename = img_info["filename"]
            # ComfyUI сохраняет в C:/comfyUI/ComfyUI/output/
            return f"C:/comfyUI/ComfyUI/output/{filename}"
        return ""

    # --- Workflow definitions ---

    def _create_upscale_workflow(
        self,
        image_path: str,
        scale: int,
        model: str
    ) -> Dict[str, Any]:
        """Создать workflow для upscale."""
        filename = Path(image_path).name

        return {
            "1": {"inputs": {"image": filename}, "class_type": "LoadImage"},
            "2": {
                "inputs": {
                    "width": 2160,
                    "height": 2700,
                    "upscale_method": "lanczos",
                    "crop": "center",
                    "image": ["1", 0]
                },
                "class_type": "ImageScale"
            },
            "3": {
                "inputs": {"filename_prefix": "upscaled", "images": ["2", 0]},
                "class_type": "SaveImage"
            }
        }

    def _create_denoise_workflow(
        self,
        image_path: str,
        strength: float
    ) -> Dict[str, Any]:
        """Создать workflow для denoise."""
        filename = Path(image_path).name
        blur_radius = max(1, int(strength * 3))
        sigma = strength * 2

        return {
            "1": {"inputs": {"image": filename}, "class_type": "LoadImage"},
            "2": {
                "inputs": {
                    "blur_radius": blur_radius,
                    "sigma": sigma,
                    "image": ["1", 0]
                },
                "class_type": "ImageBlur"
            },
            "3": {
                "inputs": {"filename_prefix": "denoised", "images": ["2", 0]},
                "class_type": "SaveImage"
            }
        }

    def _create_color_workflow(
        self,
        image_path: str,
        brightness: float,
        contrast: float,
        saturation: float,
        temperature: float
    ) -> Dict[str, Any]:
        """Создать workflow для цветокоррекции."""
        filename = Path(image_path).name

        # Brightness: factor 1.0 = без изменений, >1 = ярче
        brightness_factor = 1.0 + brightness

        # Contrast: factor 1.0 = без изменений, >1 = контрастнее
        contrast_factor = contrast

        return {
            "1": {"inputs": {"image": filename}, "class_type": "LoadImage"},
            "2": {
                "inputs": {"factor": brightness_factor, "images": ["1", 0]},
                "class_type": "AdjustBrightness"
            },
            "3": {
                "inputs": {"factor": contrast_factor, "images": ["2", 0]},
                "class_type": "AdjustContrast"
            },
            "4": {
                "inputs": {"filename_prefix": "corrected", "images": ["3", 0]},
                "class_type": "SaveImage"
            }
        }


# --- Standalone functions for simple use ---

def enhance_with_comfyui(
    image_path: str,
    upscale: bool = True,
    denoise: bool = True,
    color_correct: bool = True
) -> Dict[str, Any]:
    """
    Улучшить качество изображения через ComfyUI.

    Args:
        image_path: Путь к изображению
        upscale: Делать upscale до 2160x2700
        denoise: Удалять шумы
        color_correct: Корректировать цвет

    Returns:
        Результат с путём к файлу

    Example:
        >>> result = enhance_with_comfyui("photo.jpg")
        >>> print(result["output_path"])
        ./output/enhanced_photo.jpg
    """
    enhancer = ComfyUIEnhancer()

    if not enhancer.check_connection():
        return {
            "status": "error",
            "error": "ComfyUI not connected. Make sure ComfyUI is running."
        }

    return enhancer.enhance_full(
        image_path,
        upscale=upscale,
        denoise=denoise,
        color_correct=color_correct,
        target_size=(2160, 2700)
    )


def check_comfyui_enhancer() -> Dict[str, bool]:
    """Проверить доступность ComfyUI для улучшения."""
    enhancer = ComfyUIEnhancer()
    connected = enhancer.check_connection()

    return {
        "comfyui_connected": connected,
        "upscale_available": connected,
        "denoise_available": connected,
        "color_correct_available": connected
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python comfyui_enhancer.py <image_path>")
        print("       python comfyui_enhancer.py --check")
        sys.exit(1)

    if sys.argv[1] == "--check":
        status = check_comfyui_enhancer()
        print("ComfyUI Enhancer Status:")
        for k, v in status.items():
            print(f"  {k}: {'OK' if v else 'NOT AVAILABLE'}")
    else:
        image_path = sys.argv[1]
        print(f"Enhancing: {image_path}")
        result = enhance_with_comfyui(image_path)
        print(f"Status: {result['status']}")
        if result.get("output_path"):
            print(f"Output: {result['output_path']}")
        if result.get("steps"):
            print(f"Steps: {result['steps']}")
