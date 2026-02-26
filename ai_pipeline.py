"""AI Instagram Pipeline - полный автоматический пайплайн.

Объединяет все AI-модули в единый процесс:
1. AI-автоподбор пресета (Ollama)
2. Обработка изображения (InstagramProcessor)
3. AI-улучшение качества (ComfyUI)
4. Генерация описания (Claude/Gemini)
"""
import os
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from PIL import Image

from ai_preset_selector import ai_select_preset, AVAILABLE_PRESETS
from instagram_processor import InstagramProcessor
from comfyui_enhancer import ComfyUIEnhancer
from product_desc_generator import ProductDescriptionGenerator, generate_instagram_post


class InstaAutoPipeline:
    """
    Полный автоматический пайплайн для обработки фото для Instagram.

    Процесс:
    1. AI-анализ и автоподбор пресета (Ollama)
    2. Обработка (InstagramProcessor)
    3. AI-улучшение качества (ComfyUI)
    4. Генерация описания (Claude/Gemini)
    5. Сохранение с метаданными

    Example:
        >>> pipeline = InstaAutoPipeline()
        >>> result = pipeline.process("photo.tif")
        >>> print(result["description"])
    """

    def __init__(
        self,
        output_dir: str = None,
        comfyui_host: str = None,
        use_ai_preset: bool = True,
        use_comfyui_enhance: bool = True,
        use_ai_description: bool = True
    ):
        """
        Инициализация пайплайна.

        Args:
            output_dir: Директория для сохранения
            comfyui_host: Хост ComfyUI
            use_ai_preset: Использовать AI-подбор пресета
            use_comfyui_enhance: Использовать ComfyUI улучшение
            use_ai_description: Использовать AI-генерацию описания
        """
        self.output_dir = Path(output_dir or "./output/instagram_auto")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Подключаем модули
        self.processor = InstagramProcessor(output_dir=str(self.output_dir))
        self.comfy_enhancer = ComfyUIEnhancer(host=comfyui_host)
        self.desc_generator = ProductDescriptionGenerator()

        # Настройки
        self.use_ai_preset = use_ai_preset
        self.use_comfyui_enhance = use_comfyui_enhance
        self.use_ai_description = use_ai_description

    def process(
        self,
        image_path: str,
        category: str = "vintage_clothing",
        brand: str = None,
        price: str = None,
        target_size: tuple = (2160, 2700),
        preset: str = None,
        jpeg_quality: int = 90
    ) -> Dict[str, Any]:
        """
        Обработать изображение через полный пайплайн.

        Args:
            image_path: Путь к исходному изображению
            category: Категория товара
            brand: Бренд
            price: Цена
            target_size: Целевой размер
            preset: Пресет (если None - AI-подбор)
            jpeg_quality: Качество JPEG

        Returns:
            {
                "status": "success",
                "input_path": "...",
                "output_path": "...",
                "preset": {...},
                "description": {...},
                "processing_time": ...,
                "steps": [...]
            }

        Example:
            >>> pipeline = InstaAutoPipeline()
            >>> result = pipeline.process("DSC_4030.TIF", price="1500 руб")
            >>> print(result["output_path"])
        """
        start_time = time.time()
        steps = []

        input_path = Path(image_path)
        if not input_path.exists():
            return {"status": "error", "error": f"File not found: {image_path}"}

        # Шаг 1: AI-автоподбор пресета
        if preset:
            # Используем указанный пресет
            preset_result = {
                "preset": preset,
                "preset_name": AVAILABLE_PRESETS.get(preset, {}).get("name", preset),
                "parameters": AVAILABLE_PRESETS.get(preset, {})
            }
            steps.append("preset_manual")
        elif self.use_ai_preset:
            # AI-подбор пресета
            try:
                preset_result = ai_select_preset(image_path, use_ai=True)
                steps.append("preset_ai")
            except Exception as e:
                print(f"AI preset failed: {e}, using default")
                preset_result = {
                    "preset": "shop_vintage",
                    "preset_name": "Магазин (тёплый)",
                    "parameters": AVAILABLE_PRESETS["shop_vintage"]
                }
                steps.append("preset_fallback")
        else:
            # Дефолтный пресет
            preset_result = {
                "preset": "shop_vintage",
                "preset_name": "Магазин (тёплый)",
                "parameters": AVAILABLE_PRESETS["shop_vintage"]
            }
            steps.append("preset_default")

        # Шаг 2: Обработка через InstagramProcessor
        params = preset_result.get("parameters", {})
        try:
            process_result = self.processor.process_image(
                image_path=image_path,
                preset=preset_result["preset"],
                target_size=target_size,
                jpeg_quality=jpeg_quality,
                brightness=params.get("brightness", 0),
                contrast=params.get("contrast", 1.0),
                saturation=1.0,
                temperature=params.get("temperature", 5500),
                center_crop=True,
                auto_fix_edges=True
            )
            processed_path = process_result.get("output")
            steps.append("instagram_processed")
        except Exception as e:
            return {
                "status": "error",
                "error": f"Instagram processing failed: {e}",
                "step": "instagram_process"
            }

        # Шаг 3: AI-улучшение через ComfyUI
        enhanced_path = processed_path
        if self.use_comfyui_enhance:
            try:
                # Копируем в ComfyUI input
                comfy_input = Path("C:/comfyUI/ComfyUI/input")
                comfy_input.mkdir(exist_ok=True)

                # Нужно сохранить как PNG для ComfyUI
                img = Image.open(processed_path)
                temp_path = comfy_input / f"temp_{int(time.time())}.png"
                img.save(temp_path)
                img.close()

                # Выполняем улучшение
                enhance_result = self.comfy_enhancer.enhance_full(
                    str(temp_path),
                    upscale=True,
                    denoise=True,
                    color_correct=True,
                    target_size=target_size
                )

                if enhance_result.get("status") == "success":
                    enhanced_path = enhance_result.get("output_path", processed_path)
                    steps.append("comfyui_enhanced")
                else:
                    steps.append("comfyui_skipped")

                # Удаляем temp файл
                if temp_path.exists():
                    temp_path.unlink()

            except Exception as e:
                print(f"ComfyUI enhancement failed: {e}")
                steps.append("comfyui_failed")

        # Шаг 4: Генерация описания
        description_result = None
        if self.use_ai_description:
            try:
                description_result = self.desc_generator.generate_description(
                    enhanced_path,
                    category=category,
                    brand=brand,
                    price=price
                )
                steps.append("description_generated")
            except Exception as e:
                print(f"Description generation failed: {e}")
                description_result = {
                    "title": "Винтажная находка",
                    "description": "Отличный винтажный предмет",
                    "hashtags": "#винтаж #мода #стиль"
                }
                steps.append("description_fallback")
        else:
            steps.append("description_skipped")

        # Шаг 5: Сохранение результата
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_name = f"insta_{input_path.stem}_{timestamp}.jpg"
        final_path = self.output_dir / final_name

        # Копируем/перемещаем финальное изображение
        shutil.copy(enhanced_path, final_path)

        processing_time = time.time() - start_time

        return {
            "status": "success",
            "input_path": str(input_path),
            "output_path": str(final_path),
            "preset": preset_result,
            "description": description_result,
            "instagram_post": generate_instagram_post(
                final_path, category, brand, price
            ) if description_result else None,
            "processing_time": round(processing_time, 2),
            "steps": steps,
            "target_size": target_size,
            "jpeg_quality": jpeg_quality
        }

    def process_batch(
        self,
        image_paths: List[str],
        category: str = "vintage_clothing",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Пакетная обработка нескольких изображений.

        Args:
            image_paths: Список путей к изображениям
            category: Категория товара
            **kwargs: Дополнительные параметры для process()

        Returns:
            Список результатов
        """
        results = []

        for i, image_path in enumerate(image_paths):
            print(f"Processing {i+1}/{len(image_paths)}: {image_path}")

            try:
                result = self.process(image_path, category=category, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({
                    "status": "error",
                    "input_path": image_path,
                    "error": str(e)
                })

        return results


# --- Удобные функции ---

def process_instagram_photo(
    image_path: str,
    category: str = "vintage_clothing",
    brand: str = None,
    price: str = None,
    ai_preset: bool = True,
    enhance: bool = True
) -> Dict[str, Any]:
    """
    Удобная функция для обработки одного фото.

    Args:
        image_path: Путь к изображению
        category: Категория
        brand: Бренд
        price: Цена
        ai_preset: AI-подбор пресета
        enhance: ComfyUI улучшение

    Returns:
        Результат обработки

    Example:
        >>> result = process_instagram_photo("photo.tif", price="1500 руб")
        >>> print(result["instagram_post"])
    """
    pipeline = InstaAutoPipeline(
        use_ai_preset=ai_preset,
        use_comfyui_enhance=enhance,
        use_ai_description=True
    )

    return pipeline.process(
        image_path,
        category=category,
        brand=brand,
        price=price
    )


def process_folder(
    folder_path: str,
    category: str = "vintage_clothing",
    extensions: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Обработка всех изображений в папке.

    Args:
        folder_path: Путь к папке
        category: Категория товара
        extensions: Расширения файлов

    Returns:
        Список результатов
    """
    extensions = extensions or [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".nef"]

    folder = Path(folder_path)
    image_paths = [
        str(f) for f in folder.iterdir()
        if f.suffix.lower() in extensions
    ]

    if not image_paths:
        return []

    pipeline = InstaAutoPipeline()
    return pipeline.process_batch(image_paths, category=category)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python ai_pipeline.py <image_path>")
        print("  python ai_pipeline.py <image_path> --category vintage_clothing")
        print("  python ai_pipeline.py --folder D:/input")
        sys.exit(1)

    if sys.argv[1] == "--folder":
        folder = sys.argv[2] if len(sys.argv) > 2 else "D:/input"
        print(f"Processing folder: {folder}")
        results = process_folder(folder)
        print(f"Processed: {len(results)} images")
    else:
        image_path = sys.argv[1]
        category = "vintage_clothing"

        # Парсим аргументы
        for i, arg in enumerate(sys.argv):
            if arg == "--category" and i + 1 < len(sys.argv):
                category = sys.argv[i + 1]
            if arg == "--brand" and i + 1 < len(sys.argv):
                brand = sys.argv[i + 1]
            if arg == "--price" and i + 1 < len(sys.argv):
                price = sys.argv[i + 1]

        print(f"Processing: {image_path}")

        result = process_instagram_photo(
            image_path,
            category=category,
            brand=locals().get("brand"),
            price=locals().get("price")
        )

        print(f"Status: {result['status']}")
        print(f"Output: {result.get('output_path')}")
        print(f"Time: {result.get('processing_time')}s")
        print(f"Steps: {result.get('steps')}")

        if result.get("description"):
            print(f"\nDescription: {result['description']['title']}")
            print(result['description']['description'])
            print(f"\nHashtags: {result['description']['hashtags']}")
