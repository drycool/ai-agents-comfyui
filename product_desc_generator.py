"""AI Product Description Generator - генерация описаний товаров для Instagram.

Автоматическая генерация:
- Продающего описания товара
- Хештегов
- Метаданных (размер, состояние, категория)
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np

from cli_wrappers import ClaudeWrapper, GeminiWrapper, OllamaWrapper


class ProductDescriptionGenerator:
    """Генератор описаний товаров для Instagram."""

    # Категории товаров
    CATEGORIES = {
        "vintage_clothing": "Винтажная одежда",
        "modern_clothing": "Современная одежда",
        "accessories": "Аксессуары",
        "shoes": "Обувь",
        "bags": "Сумки",
        "jewelry": "Украшения"
    }

    # Состояние товара
    CONDITIONS = {
        "new": "Новое с биркой",
        "like_new": "Как новое",
        "excellent": "Отличное",
        "good": "Хорошее",
        "fair": "Удовлетворительное"
    }

    def __init__(self):
        self.claude = ClaudeWrapper()
        self.gemini = GeminiWrapper()
        self.ollama = OllamaWrapper()

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Анализ изображения для извлечения информации о товаре.

        Args:
            image_path: Путь к изображению

        Returns:
            Словарь с анализом
        """
        # Базовый анализ изображения
        img = Image.open(image_path)
        analysis = self._basic_image_analysis(img)
        img.close()

        # AI анализ через Gemini (если доступен)
        ai_analysis = self._ai_image_analysis(image_path)

        # Объединяем результаты
        return {
            **analysis,
            **ai_analysis
        }

    def _basic_image_analysis(self, img: Image.Image) -> Dict[str, Any]:
        """Базовый анализ изображения."""
        img_array = np.array(img)

        # Размеры
        width, height = img.size
        aspect_ratio = width / height

        # Основные характеристики
        avg_brightness = float(img_array.mean())
        is_dark = bool(avg_brightness < 100)
        is_bright = bool(avg_brightness > 180)

        # Определение типа фото по соотношению
        if aspect_ratio > 1.5:
            layout = "landscape"
        elif aspect_ratio < 0.7:
            layout = "portrait"
        else:
            layout = "square"

        # Определение доминирующего цвета
        if len(img_array.shape) == 3:
            r_mean = img_array[:, :, 0].mean()
            g_mean = img_array[:, :, 1].mean()
            b_mean = img_array[:, :, 2].mean()

            if r_mean > g_mean and r_mean > b_mean:
                dominant_color = "red"
            elif g_mean > r_mean and g_mean > b_mean:
                dominant_color = "green"
            elif b_mean > r_mean and b_mean > g_mean:
                dominant_color = "blue"
            elif r_mean > 150 and g_mean > 150 and b_mean < 100:
                dominant_color = "yellow"
            elif r_mean > 150 and g_mean < 100 and b_mean > 150:
                dominant_color = "purple"
            elif r_mean > 150 and g_mean > 100 and b_mean > 100:
                dominant_color = "brown"
            else:
                dominant_color = "neutral"
        else:
            dominant_color = "gray"

        return {
            "width": width,
            "height": height,
            "layout": layout,
            "aspect_ratio": round(aspect_ratio, 2),
            "dominant_color": dominant_color,
            "brightness": round(avg_brightness, 1),
            "is_dark": is_dark,
            "is_bright": is_bright,
            "needs_brightness_adjustment": is_dark or is_bright
        }

    def _ai_image_analysis(self, image_path: str) -> Dict[str, Any]:
        """AI-анализ через Gemini/Ollama."""
        result = {
            "ai_analysis": False,
            "description": "",
            "category": "vintage_clothing",
            "detected_items": [],
            "style": ""
        }

        # AI-анализ через Ollama (всегда доступен)
        try:
            basic_analysis = self._basic_image_analysis(Image.open(image_path))
            prompt = f"""Проанализируй это фото одежды.

Характеристики изображения:
- Доминирующий цвет: {basic_analysis.get('dominant_color', 'нейтральный')}
- Яркость: {basic_analysis.get('brightness', 0)}
- Освещение: {'тёмное' if basic_analysis.get('is_dark') else 'светлое' if basic_analysis.get('is_bright') else 'нормальное'}

Опиши:
1. Что изображено (одежда, аксессуар)
2. Стиль (винтажный, современный, спортивный и т.д.)
3. Цвет (основной)
4. Состояние (новое, б/у, как новое)

Верни краткое описание (2-3 предложения)."""

            ollama_result = self.ollama.run(prompt, max_tokens=300)
            result["ai_analysis"] = True
            result["ollama_analysis"] = ollama_result
            result["description"] = ollama_result[:200]
        except Exception as e:
            print(f"Ollama analysis failed: {e}")

        # Пробуем Gemini (опционально)
        try:
            gemini_result = self.gemini.analyze_image(image_path)
            result["gemini_analysis"] = gemini_result[:500]
        except Exception as e:
            print(f"Gemini analysis failed: {e}")

        return result

    def generate_description(
        self,
        image_path: str,
        category: str = "vintage_clothing",
        brand: str = None,
        price: str = None
    ) -> Dict[str, Any]:
        """
        Генерация полного описания товара.

        Args:
            image_path: Путь к изображению
            category: Категория товара
            brand: Бренд (опционально)
            price: Цена (опционально)

        Returns:
            {
                "title": "...",
                "description": "...",
                "hashtags": ["...", ...],
                "size": "...",
                "condition": "...",
                "category": "..."
            }

        Example:
            >>> generator = ProductDescriptionGenerator()
            >>> result = generator.generate_description("photo.jpg", "vintage_clothing")
            >>> print(result["description"])
        """
        # Анализ изображения
        analysis = self.analyze_image(image_path)

        # Формируем промпт для Claude
        category_name = self.CATEGORIES.get(category, category)

        prompt = f"""Создай продающее описание товара для Instagram-магазина винтажной одежды.

Информация о товаре:
- Категория: {category_name}
- Бренд: {brand or 'не указан'}
- Цена: {price or 'не указана'}
- Доминирующий цвет: {analysis.get('dominant_color', 'нейтральный')}
- Стиль: {analysis.get('style', 'винтажный')}
- AI анализ: {analysis.get('ollama_analysis', analysis.get('gemini_analysis', 'стандартный анализ'))[:300]}

Верни JSON с полями:
{{
    "title": "Короткое привлекательное название (до 50 символов)",
    "description": "Продающее описание (2-3 предложения, эмоциональное)",
    "hashtags": "Хештеги через пробел (10-15 штук, включая: #винтаж #одежда #мода)",
    "size": "Размер (если определён, иначе 'размер не указан')",
    "condition": "Состояние: new, like_new, excellent, good, fair",
    "color": "Основной цвет (1-2 слова)"
}}

Верни ТОЛЬКО JSON, без пояснений."""

        # Генерируем через Claude
        try:
            response = self.claude.run(prompt, max_tokens=1000)
            result = self._parse_json_response(response)
        except Exception as e:
            print(f"Claude generation failed: {e}")
            result = self._generate_fallback_description(analysis, category)

        # Добавляем анализ в результат
        result["analysis"] = analysis

        return result

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Парсинг JSON из ответа Claude."""
        try:
            # Ищем JSON в ответе
            start = response.find('{')
            end = response.rfind('}') + 1

            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Fallback
        return {
            "title": "Винтажная находка",
            "description": "Отличный винтажный предмет для вашего гардероба",
            "hashtags": "#винтаж #одежда #мода #стиль #красота",
            "size": "размер не указан",
            "condition": "good",
            "color": "нейтральный"
        }

    def _generate_fallback_description(
        self,
        analysis: Dict,
        category: str
    ) -> Dict[str, Any]:
        """Генерация описания без AI."""
        color = analysis.get("dominant_color", "стильный")
        category_name = self.CATEGORIES.get(category, "товар")

        return {
            "title": f"Винтажный {color} {category_name}",
            "description": f"Отличный {color} винтажный {category_name}. Идеальное дополнение к вашему гардеробу!",
            "hashtags": f"#винтаж #{category_name} #мода #стиль #{color}",
            "size": "размер не указан",
            "condition": "good",
            "color": color
        }

    def generate_hashtags(
        self,
        category: str = "vintage_clothing",
        color: str = None,
        style: str = "vintage"
    ) -> str:
        """
        Генерация хештегов.

        Args:
            category: Категория
            color: Цвет
            style: Стиль

        Returns:
            Строка с хештегами
        """
        base_tags = [
            "#винтаж", "#одежда", "#мода", "#стиль",
            "#шопинг", "#находка", "#уникальнаявещь"
        ]

        category_tags = {
            "vintage_clothing": ["#винтажнаяодежда", "#винтаж"],
            "modern_clothing": ["#современнаямода", "#новыйlook"],
            "accessories": ["#аксессуары", "#детали"],
            "shoes": ["#обувь", "#обувьвинтаж"],
            "bags": ["#сумки", "#сумкавинтаж"],
            "jewelry": ["#украшения", "#бижутерия"]
        }

        color_tags = {
            "red": ["#красный", "#бордо"],
            "blue": ["#синий", "#голубой"],
            "green": ["#зеленый", "#оливковый"],
            "yellow": ["#желтый", "#золотой"],
            "brown": ["#коричневый", "#бежевый"],
            "black": ["#черный", "# monochrome"],
            "white": ["#белый", "#минимализм"]
        }

        all_tags = base_tags + category_tags.get(category, [])

        if color:
            all_tags += color_tags.get(color, [])

        return " ".join(all_tags[:15])


# --- Удобные функции ---

def generate_product_description(
    image_path: str,
    category: str = "vintage_clothing",
    brand: str = None,
    price: str = None
) -> Dict[str, Any]:
    """
    Удобная функция для генерации описания товара.

    Args:
        image_path: Путь к изображению
        category: Категория товара
        brand: Бренд
        price: Цена

    Returns:
        Словарь с описанием

    Example:
        >>> result = generate_product_description("photo.jpg")
        >>> print(result["description"])
    """
    generator = ProductDescriptionGenerator()
    return generator.generate_description(image_path, category, brand, price)


def generate_instagram_post(
    image_path: str,
    category: str = "vintage_clothing",
    brand: str = None,
    price: str = None
) -> str:
    """
    Генерирует готовый текст поста для Instagram.

    Args:
        image_path: Путь к изображению
        category: Категория товара
        brand: Бренд
        price: Цена

    Returns:
        Готовый текст поста

    Example:
        >>> post = generate_instagram_post("photo.jpg", price="1500 руб")
        >>> print(post)
    """
    result = generate_product_description(image_path, category, brand, price)

    post = f"""✨ {result['title']}

{result['description']}

📏 Размер: {result['size']}
⭐ Состояние: {result.get('condition', 'отличное')}
{brand and f'🏷 Бренд: {brand}\n'}{price and f'💰 Цена: {price}\n'}

{result['hashtags']}

#instagram #магазин #купить #винтаж #aliexpress """
    return post


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python product_desc_generator.py <image_path>")
        print("  python product_desc_generator.py <image_path> --category vintage_clothing")
        sys.exit(1)

    image_path = sys.argv[1]
    category = "vintage_clothing"

    if len(sys.argv) > 2 and sys.argv[2] == "--category":
        category = sys.argv[3] if len(sys.argv) > 3 else "vintage_clothing"

    print(f"Генерация описания для: {image_path}")
    print("-" * 50)

    result = generate_product_description(image_path, category)

    print(f"Название: {result['title']}")
    print(f"Описание: {result['description']}")
    print(f"Хештеги: {result['hashtags']}")
    print(f"Размер: {result['size']}")
    print(f"Состояние: {result['condition']}")
    print(f"Цвет: {result.get('color', 'не указан')}")
