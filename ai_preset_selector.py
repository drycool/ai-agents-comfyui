"""AI Preset Selector - автоматический подбор пресета через Ollama.

Анализирует фото и выбирает оптимальный пресет обработки.
"""
import os
import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np

from cli_wrappers import OllamaWrapper

# Доступные пресеты
AVAILABLE_PRESETS = {
    "shop_vintage": {
        "name": "Магазин (тёплый)",
        "brightness": 20,
        "contrast": 1.15,
        "temperature": 6000,
        "description": "Для тёмных фото с тёплым оттенком"
    },
    "warm_vintage": {
        "name": "Тёплый винтаж",
        "brightness": 15,
        "contrast": 1.1,
        "temperature": 6500,
        "description": "Для холодных фото, добавляет теплоты"
    },
    "neutral": {
        "name": "Нейтральный",
        "brightness": 5,
        "contrast": 1.05,
        "temperature": 5500,
        "description": "Лёгкая коррекция, естественные цвета"
    },
    "minimal": {
        "name": "Минималистичный",
        "brightness": 0,
        "contrast": 1.0,
        "temperature": 5200,
        "description": "Без изменений, слегка холодный"
    }
}


def analyze_image_basic(image_path: str) -> Dict[str, Any]:
    """
    Базовый анализ изображения без AI.

    Возвращает характеристики изображения для помощи в выборе пресета.
    """
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    img_array = np.array(img)

    # Анализ яркости
    brightness = img_array.mean()

    # Анализ цветового баланса (R, G, B)
    r_mean = img_array[:, :, 0].mean()
    g_mean = img_array[:, :, 1].mean()
    b_mean = img_array[:, :, 2].mean()

    # Относительный цветовой баланс
    total = r_mean + g_mean + b_mean
    r_ratio = r_mean / total
    g_ratio = g_mean / total
    b_ratio = b_mean / total

    # Определение доминирующего оттенка
    if r_ratio > 0.36:
        color_cast = "warm"
    elif b_ratio > 0.36:
        color_cast = "cool"
    else:
        color_cast = "neutral"

    # Анализ качества (контрастность)
    contrast = img_array.std()

    # Определение качества изображения
    if contrast < 30:
        quality_issue = "low_contrast"
    elif brightness < 80:
        quality_issue = "dark"
    elif brightness > 200:
        quality_issue = "overexposed"
    else:
        quality_issue = "none"

    return {
        "brightness": float(brightness),
        "contrast": float(contrast),
        "color_cast": color_cast,
        "quality_issue": quality_issue,
        "r_ratio": float(r_ratio),
        "g_ratio": float(g_ratio),
        "b_ratio": float(b_ratio)
    }


def select_preset_from_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Выбор пресета на основе базового анализа.

    Args:
        analysis: Результат analyze_image_basic()

    Returns:
        Словарь с выбранным пресетом и параметрами
    """
    brightness = analysis["brightness"]
    color_cast = analysis["color_cast"]
    quality_issue = analysis["quality_issue"]

    # Логика выбора пресета
    if brightness < 80 or quality_issue == "dark":
        # Тёмное фото - нужен яркий пресет
        preset_name = "shop_vintage"
        adjustment = {
            "brightness": min(30, int(80 - brightness) + 10)
        }
    elif brightness > 200 or quality_issue == "overexposed":
        # Переэкспонированное - минимальная обработка
        preset_name = "minimal"
        adjustment = {
            "brightness": -10,
            "contrast": 1.1
        }
    elif color_cast == "cool":
        # Холодное фото - тёплый пресет
        preset_name = "warm_vintage"
        adjustment = {}
    elif color_cast == "warm":
        # Уже тёплое - нейтральный
        preset_name = "neutral"
        adjustment = {}
    else:
        # Нейтральное
        preset_name = "shop_vintage"
        adjustment = {}

    # Применяем базовые значения пресета + корректировки
    preset = AVAILABLE_PRESETS[preset_name].copy()
    preset.update(adjustment)

    return {
        "preset": preset_name,
        "preset_name": preset["name"],
        "parameters": {
            "brightness": preset["brightness"],
            "contrast": preset["contrast"],
            "temperature": preset["temperature"]
        },
        "reasoning": _build_reasoning(analysis, preset_name, adjustment),
        "analysis": analysis
    }


def _build_reasoning(analysis: Dict[str, Any], preset_name: str, adjustment: Dict) -> str:
    """Сформировать текстовое объяснение выбора."""
    reasons = []

    if analysis["brightness"] < 80:
        reasons.append("фото слишком тёмное")
    elif analysis["brightness"] > 200:
        reasons.append("фото переэкспонировано")

    if analysis["color_cast"] == "cool":
        reasons.append("холодный оттенок")
    elif analysis["color_cast"] == "warm":
        reasons.append("тёплый оттенок")

    if not reasons:
        reasons.append("нейтральное освещение")

    return f"Выбран пресет '{preset_name}': {', '.join(reasons)}"


def ai_select_preset(image_path: str, use_ai: bool = True) -> Dict[str, Any]:
    """
    AI-автоподбор пресета для изображения.

    Основная функция модуля. Использует Ollama для глубокого анализа,
    с fallback на базовый анализ при недоступности AI.

    Args:
        image_path: Путь к изображению
        use_ai: Использовать AI-анализ (True) или только базовый (False)

    Returns:
        {
            "preset": "shop_vintage",
            "parameters": {"brightness": 20, "contrast": 1.15, "temperature": 6000},
            "reasoning": "...",
            "ai_used": True/False,
            "analysis": {...}
        }

    Example:
        >>> result = ai_select_preset("photo.jpg")
        >>> print(result["preset"])
        shop_vintage
        >>> print(result["parameters"])
        {'brightness': 20, 'contrast': 1.15, 'temperature': 6000}
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Базовый анализ (всегда выполняем)
    basic_analysis = analyze_image_basic(image_path)

    if not use_ai:
        # Только базовый анализ
        result = select_preset_from_analysis(basic_analysis)
        result["ai_used"] = False
        return result

    # Пробуем AI-анализ через Ollama
    try:
        return _ai_analyze_with_ollama(image_path, basic_analysis)
    except Exception as e:
        print(f"AI analysis failed, using fallback: {e}")
        result = select_preset_from_analysis(basic_analysis)
        result["ai_used"] = False
        result["ai_error"] = str(e)
        return result


def _ai_analyze_with_ollama(image_path: str, basic_analysis: Dict) -> Dict[str, Any]:
    """
    Глубокий AI-анализ через Ollama.

    Анализирует фото с помощью vision-модели Ollama.
    """
    # Подготовка промпта для анализа
    prompt = f"""Проанализируй это фото одежды для Instagram-магазина винтажной одежды.

Характеристики изображения (автоматический анализ):
- Яркость: {basic_analysis['brightness']:.1f} (норма: 80-180)
- Контраст: {basic_analysis['contrast']:.1f} (норма: 40-80)
- Цветовой оттенок: {basic_analysis['color_cast']}
- Проблема качества: {basic_analysis['quality_issue']}

Верни JSON с полями:
{{
    "preset": "shop_vintage" | "warm_vintage" | "neutral" | "minimal",
    "brightness": число от -30 до +30 (корректировка яркости),
    "contrast": число от 0.8 до 1.5 (множитель контраста),
    "temperature": число от 4000 до 8000 (K - температура цвета),
    "reasoning": "краткое объяснение выбора (1-2 предложения)"
}}

Выбери наиболее подходящий пресет и параметры обработки."""

    # Пробуем использовать Ollama с поддержкой vision
    # Модель llama3.2-vision или аналог
    model = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision")

    try:
        # Пробуем vision-модель
        response = OllamaWrapper.run(
            prompt=f"Analyze this image. {prompt}",
            model=model,
            max_tokens=512
        )
    except Exception as e:
        # Если vision не работает, используем текстовую модель с описанием
        print(f"Vision model failed, using text analysis: {e}")

        # Генерируем описание на основе базового анализа
        description = f"""Фото одежды:
- Яркость: {basic_analysis['brightness']:.0f} ({'тёмное' if basic_analysis['brightness'] < 80 else 'нормальное' if basic_analysis['brightness'] < 180 else 'светлое'})
- Контраст: {basic_analysis['contrast']:.0f} ({'низкий' if basic_analysis['contrast'] < 40 else 'нормальный'})
- Оттенок: {basic_analysis['color_cast']}"""

        response = OllamaWrapper.run(
            prompt=f"{description}\n\n{prompt}",
            max_tokens=512
        )

    # Парсим ответ
    try:
        # Ищем JSON в ответе
        start = response.find('{')
        end = response.rfind('}') + 1

        if start >= 0 and end > start:
            json_str = response[start:end]
            ai_result = json.loads(json_str)
        else:
            raise ValueError("No JSON found in response")

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse AI response: {e}")
        # Fallback на базовый анализ
        return select_preset_from_analysis(basic_analysis)

    # Формируем результат
    preset_name = ai_result.get("preset", "neutral")

    # Проверяем что пресет существует
    if preset_name not in AVAILABLE_PRESETS:
        preset_name = "neutral"

    preset = AVAILABLE_PRESETS[preset_name]

    return {
        "preset": preset_name,
        "preset_name": preset["name"],
        "parameters": {
            "brightness": ai_result.get("brightness", preset["brightness"]),
            "contrast": ai_result.get("contrast", preset["contrast"]),
            "temperature": ai_result.get("temperature", preset["temperature"])
        },
        "reasoning": ai_result.get("reasoning", "AI-анализ"),
        "ai_used": True,
        "analysis": basic_analysis,
        "ai_raw_response": response[:200]  # Для отладки
    }


def get_available_presets() -> Dict[str, Dict]:
    """Вернуть список доступных пресетов."""
    return AVAILABLE_PRESETS.copy()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ai_preset_selector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"Анализ изображения: {image_path}")
    print("-" * 50)

    result = ai_select_preset(image_path, use_ai=True)

    print(f"AI использован: {result['ai_used']}")
    print(f"Выбран пресет: {result['preset_name']}")
    print(f"Параметры:")
    print(f"  - Яркость: {result['parameters']['brightness']:+d}")
    print(f"  - Контраст: {result['parameters']['contrast']:.2f}")
    print(f"  - Температура: {result['parameters']['temperature']}K")
    print(f"Обоснование: {result['reasoning']}")
