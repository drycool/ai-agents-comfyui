# AI Instagram Automation System

Система автоматизации обработки изображений для Instagram-магазинов винтажной одежды.

## Два направления

| Направление | Статус | Описание |
|-------------|--------|----------|
| **Classic** | ✅ Готов | Классическая обработка без AI |
| **AI Pipeline** | ⏸️ Незавершен | Полный AI-пайплайн |

---

## Направление 1: Classic (ПОЛНОСТЬЮ РАБОЧЕЕ)

Классическая обработка фото без использования AI и ComfyUI.

### Использование

```python
from instagram_processor import InstagramProcessor

processor = InstagramProcessor()
result = processor.process("photo.NEF")  # NEF, TIFF, JPG
print(result["output_path"])
```

### Возможности

- Поддержка NEF (Nikon RAW), TIFF, JPG
- Кадрирование 4:5 (2160x2700)
- Тёплая цветокоррекция (vintage style)
- Автоудаление артефактов
- Выход: 2-3 MB, 80% качество
- Время обработки: ~4 сек

### Параметры

| Параметр | Значение |
|----------|----------|
| Brightness | +10 |
| Contrast | +15% |
| Temperature | 5800K |
| Качество | 80% |
| Разрешение | 2160x2700 |

---

## Направление 2: AI Pipeline (НЕЗАВЕРШЕНО)

```
Ollama (анализ) → InstagramProcessor → ComfyUI → (описание)
```

### М Claude/Geminiодули

- `ai_preset_selector.py` — AI-автоподбор пресета
- `comfyui_enhancer.py` — AI-улучшение
- `product_desc_generator.py` — генерация описаний
- `ai_pipeline.py` — объединяющий пайплайн

### Требования

- ComfyUI на http://localhost:8188
- Ollama с llama3.2/llama3.3

⏸️ Требует доработки и интеграции.

---

## Структура проекта

```
├── instagram_processor.py    # ✅ Classic processor
├── vintage_processor.py     # ✅ Legacy processor
├── ai_pipeline.py           # ⏸️ Full AI pipeline
├── ai_preset_selector.py    # ⏸️ AI preset selection
├── comfyui_enhancer.py      # ⏸️ ComfyUI enhancement
├── product_desc_generator.py# ⏸️ Description generation
├── app.py                    # Streamlit web-интерфейс
└── output/instagram/        # Результаты
```

---

## Запуск

### Classic (рекомендуется)

```python
from instagram_processor import InstagramProcessor

processor = InstagramProcessor()
result = processor.process("input/photo.NEF")
```

### Web-интерфейс

```bash
streamlit run app.py
```

---

## Период работы

**Завершение активной разработки:** 2026-03-04

Classic-режим полностью рабочий. AI Pipeline требует дополнительной интеграции.

---

## Python

Python 3.13: `C:\Users\369\AppData\Local\Programs\Python\Python313\python.exe`

## Environment

Переменные в `.env`:
- OUTPUT_DIR
- COMFYUI_HOST, COMFYUI_PORT
- OLLAMA_HOST
