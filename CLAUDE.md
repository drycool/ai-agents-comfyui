# AI Agents System for ComfyUI

Система локальных AI-агентов для автоматизированной генерации изображений через ComfyUI.

## Project Overview

- **Основной скрипт**: `launcher.py` — точка входа в систему
- **Оркестратор**: `orchestrator.py` — координирует Ollama, Claude, Gemini
- **API клиент**: `comfy_client.py` — взаимодействие с ComfyUI
- **CLI обёртки**: `cli_wrappers.py` — Ollama, Claude, Gemini wrappers

## Commands

### Запуск генерации изображений
```bash
# Простой режим
python launcher.py "a cat on sunset beach"

# Улучшенный режим
python launcher.py "cyberpunk city" --mode enhanced

# Полный режим
python launcher.py "abstract art" --mode full

# Проверка статуса
python launcher.py --status

# Список моделей Ollama
python launcher.py --list-models
```

### Запуск web-интерфейса
```bash
# Установить зависимости
pip install streamlit opencv-python rawpy

# Запустить web-интерфейс
streamlit run app.py
```

### Требования окружения
- ComfyUI запущен на http://localhost:8188
- Ollama с моделями llama3.2/llama3.3
- Claude CLI и Gemini CLI установлены

## Code Style

- Python 3.10+
- Аннотации типов для функций и переменных
- docstrings для всех функций и классов
- Форматирование: black

## Architecture

```
Пользователь → Ollama (анализ) → Gemini (стиль) → Claude (промпт) → ComfyUI → Изображение
```

Три режима генерации: simple, enhanced, full

### Web Interface
- `app.py` — Streamlit web-интерфейс для обработки фото
- Запуск: `streamlit run app.py`
- Поддержка одиночной и пакетной обработки

## Git

- Ветки: feature/номер-описание
- Коммиты: глагол(область): описание

## Environment

Переменные в .env:
- COMFYUI_HOST, COMFYUI_PORT
- OLLAMA_HOST, OLLAMA_DEFAULT_MODEL
- OUTPUT_DIR, WORKFLOWS_DIR

## Web Interface

### Streamlit
```bash
# Установка зависимостей
pip install streamlit

# Запуск web-интерфейса
streamlit run app.py
```

Web-интерфейс доступен по http://localhost:8501
