# AI Agents System for ComfyUI

Система локальных AI-агентов для автоматизированной генерации изображений через ComfyUI.

## Project Overview

- **Основной скрипт**: `launcher.py` — точка входа в систему
- **Оркестратор**: `orchestrator.py` — координирует Ollama, Claude, Gemini
- **API клиент**: `comfy_client.py` — взаимодействие с ComfyUI
- **CLI обёртки**: `cli_wrappers.py` — Ollama, Claude, Gemini wrappers

## Commands

### Запуск генерации
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

## Git

- Ветки: feature/номер-описание
- Коммиты: глагол(область): описание

## Environment

Переменные в .env:
- COMFYUI_HOST, COMFYUI_PORT
- OLLAMA_HOST, OLLAMA_DEFAULT_MODEL
- OUTPUT_DIR, WORKFLOWS_DIR
