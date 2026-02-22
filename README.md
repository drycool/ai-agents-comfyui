# AI Agents System for ComfyUI

**Локальная система AI-агентов для автоматизированной генерации изображений**

---

## Оглавление

1. [Обзор](#обзор)
2. [Архитектура](#архитектура)
3. [Компоненты](#компоненты)
4. [Установка и запуск](#установка-и-запуск)
5. [Использование](#использование)
6. [Пайплайны генерации](#пайплайны-генерации)
7. [Конфигурация](#конфигурация)
8. [Расширение системы](#расширение-системы)

---

## Обзор

Система представляет собой локальную автоматизированную систему управления ComfyUI с использованием AI-моделей. Все компоненты работают локально, не требуя подключения к внешним API.

### Ключевые возможности

- **Три режима генерации**: простой, улучшенный, полный
- **Оркестрация AI-инструментов**: Ollama, Claude CLI, Gemini CLI
- **Гибкая настройка**: конфигурация через .env файл
- **Модульная архитектура**: легко расширяется новыми агентами
- **Командный интерфейс**: удобный launcher для запуска

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                       │
│              (Ollama: LLaMA 3.2 / Qwen)                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌─────────┐  ┌──────────┐  ┌───────────┐
   │ Claude  │  │  Gemini   │  │  ComfyUI   │
   │   CLI   │  │   CLI     │  │    API     │
   └─────────┘  └──────────┘  └───────────┘
        │             │             │
        └─────────────┴─────────────┘
                      │
              ┌───────▼───────┐
              │  Generated    │
              │   Images      │
              └───────────────┘
```

### Поток данных

1. **Вход**: Запрос пользователя на естественном языке
2. **Анализ**: Ollama анализирует запрос, извлекая параметры
3. **Стилизация**: Gemini предлагает художественный стиль
4. **Улучшение**: Claude улучшает промпт
5. **Генерация**: ComfyUI создаёт изображение
6. **Выход**: Готовое изображение

---

## Компоненты

### 1. ComfyUI API Client (`comfy_client.py`)

Клиент для взаимодействия с ComfyUI через REST API.

**Функции:**
- `queue_prompt()` — постановка workflow в очередь
- `get_history()` — получение истории выполнения
- `execute_workflow()` — выполнение workflow с ожиданием
- `get_images()` — получение сгенерированных изображений
- `download_image()` — скачивание изображения

**Вспомогательные функции:**
- `create_simple_sdxl_workflow()` — создание базового SDXL workflow

### 2. CLI Wrappers (`cli_wrappers.py`)

Обёртки для CLI инструментов.

#### OllamaWrapper
- `run()` — выполнение промпта через модель
- `list_models()` — список доступных моделей
- `pull()` — загрузка модели
- `check_connection()` — проверка соединения

#### ClaudeWrapper
- `run()` — выполнение промпта через Claude
- `enhance_prompt()` — улучшение промпта для генерации
- `generate_workflow()` — генерация workflow из описания
- `check_connection()` — проверка соединения

#### GeminiWrapper
- `run()` — выполнение промпта через Gemini
- `suggest_style()` — предложение художественного стиля
- `analyze_image()` — анализ изображения
- `check_connection()` — проверка соединения

### 3. Orchestrator Agent (`orchestrator.py`)

Главный агент-оркестратор, координирующий все компоненты.

**Методы:**
- `analyze_request()` — анализ запроса через Ollama
- `get_style_suggestions()` — получение стиля от Gemini
- `enhance_prompt()` — улучшение промпта через Claude
- `generate_workflow()` — создание workflow
- `process_simple()` — простой пайплайн
- `process_enhanced()` — улучшенный пайплайн
- `process_full()` — полный пайплайн с верификацией

### 4. Launcher (`launcher.py`)

Удобный интерфейс запуска системы.

**Команды:**
- `--status` / `-s` — проверка статуса всех подключений
- `--list-models` / `-l` — список моделей Ollama
- `--mode` — выбор режима (simple/enhanced/full)

---

## Установка и запуск

### Требования

- Python 3.10+
- Ollama 0.16.3+
- Claude CLI 2.1.50+
- Gemini CLI 0.29.5+
- ComfyUI с запущенным API

### Шаг 1: Установка зависимостей

```bash
cd D:/newproject
pip install -r requirements.txt
```

### Шаг 2: Настройка ComfyUI

1. Запустить ComfyUI с API:
   ```bash
   cd ComfyUI
   python main.py --listen 0.0.0.0 --port 8188
   ```

2. Включить API: **Settings → API → Enable API**

### Шаг 3: Проверка статуса

```bash
python launcher.py --status
```

### Шаг 4: Запуск генерации

```bash
# Простой режим
python launcher.py "a cat sitting on a sunset beach"

# Улучшенный режим
python launcher.py "cyberpunk city" --mode enhanced

# Полный режим
python launcher.py "abstract art" --mode full
```

---

## Использование

### Программный API

```python
from orchestrator import OrchestratorAgent

# Создание оркестратора
orchestrator = OrchestratorAgent()

# Простой режим
result = orchestrator.process_simple("a cat")

# Улучшенный режим
result = orchestrator.process_enhanced("a cat")

# Полный режим
result = orchestrator.process_full("a cat")

# Доступ к результатам
print(result["status"])      # "completed", "timeout"
print(result["prompt"])      # Использованный промпт
print(result["images"])       # Список изображений
```

### Прямое использование клиентов

```python
from comfy_client import ComfyAPIClient

comfy = ComfyAPIClient()

# Создание workflow
workflow = {
    "3": {"inputs": {"text": "a cat", "clip": ["9", 0]}, "class_type": "CLIPTextEncode"},
    # ... остальные узлы
}

# Выполнение
result = comfy.execute_workflow(workflow, wait=True)
```

```python
from cli_wrappers import OllamaWrapper, ClaudeWrapper, GeminiWrapper

# Ollama
response = OllamaWrapper.run("Привет", model="llama3.2")

# Claude
response = ClaudeWrapper.enhance_prompt("a cat", style="realistic")

# Gemini
response = GeminiWrapper.suggest_style("a futuristic city")
```

---

## Пайплайны генерации

### Simple (Простой)

```
Пользователь → Ollama (анализ) → ComfyUI → Изображение
```

- Быстрая генерация
- Минимальная обработка
- Подходит для простых запросов

**Пример:**
```bash
python launcher.py "red rose flower" --mode simple
```

### Enhanced (Улучшенный)

```
Пользователь → Ollama (анализ) → Gemini (стиль) → Claude (промпт) → ComfyUI → Изображение
```

- Улучшенное качество промпта
- Рекомендации по стилю
- Лучшие результаты

**Пример:**
```bash
python launcher.py "red rose flower" --mode enhanced
```

### Full (Полный)

```
Пользователь → Ollama → Gemini → Claude → ComfyUI → Ollama (верификация) → Результат
```

- Максимальное качество
- Верификация результата
- Полный контроль

**Пример:**
```bash
python launcher.py "red rose flower" --mode full
```

---

## Конфигурация

### Файл `.env`

```env
# ComfyUI Settings
COMFYUI_HOST=http://localhost:8188
COMFYUI_LISTEN=0.0.0.0
COMFYUI_PORT=8188

# Ollama Settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_DEFAULT_MODEL=llama3.3
OLLAMA_ORCHESTRATOR_MODEL=llama3.3
OLLAMA_LIGHT_MODEL=phi4

# Claude CLI Settings
CLAUDE_CLI_PATH=claude

# Gemini CLI Settings
GEMINI_CLI_PATH=gemini

# Paths
OUTPUT_DIR=./output
WORKFLOWS_DIR=./workflows

# Generation Settings
DEFAULT_WIDTH=1024
DEFAULT_HEIGHT=1024
DEFAULT_STEPS=20
DEFAULT_CFG=7.0
```

### Параметры генерации

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `width` | Ширина изображения | 1024 |
| `height` | Высота изображения | 1024 |
| `steps` | Количество шагов | 20 |
| `cfg` | CFG scale | 7.0 |
| `seed` | Seed для генерации | случайный |

---

## Расширение системы

### Добавление нового агента

1. Создать класс-обёртку в `cli_wrappers.py`:
```python
class NewAgentWrapper:
    @staticmethod
    def run(prompt: str) -> str:
        # Реализация
        pass
```

2. Добавить в `OrchestratorAgent.__init__()`:
```python
self.new_agent = NewAgentWrapper()
```

3. Использовать в пайплайне:
```python
result = self.new_agent.run(prompt)
```

### Добавление нового workflow

1. Создать JSON файл в `workflows/`
2. Использовать в коде:
```python
workflow = client.load_workflow_file("workflows/my_workflow.json")
```

### Создание веб-интерфейса

Для создания веб-интерфейса рекомендуется:

- **Streamlit**: `pip install streamlit`
- **Flask/FastAPI**: для REST API
- **Gradio**: для демонстрации

---

## Устранение проблем

### ComfyUI не подключается

```bash
# Проверить, что ComfyUI запущен
curl http://localhost:8188/system_stats

# Запустить ComfyUI
cd ComfyUI
python main.py --listen 0.0.0.0 --port 8188
```

### Ollama модели недоступны

```bash
# Список моделей
ollama list

# Загрузить модель
ollama pull llama3.2
```

### Ошибки Claude/Gemini CLI

```bash
# Проверить установку
claude --version
gemini --version

# Проверить доступность
python -c "from cli_wrappers import check_all_connections; print(check_all_connections())"
```

---

## Версии окружения

| Компонент | Версия |
|-----------|--------|
| Ollama | 0.16.3 |
| Claude CLI | 2.1.50 |
| Gemini CLI | 0.29.5 |
| CUDA | 13.0 |

### Доступные модели Ollama

- `llama3.2:latest` (2.0 GB)
- `llama3.1:8b` (4.9 GB)
- `llama3.1:latest` (4.9 GB)
- `qwen2.5-coder:7b` (4.7 GB)
- `qwen2.5-coder:1.5b-base` (986 MB)
- `erwan2/DeepSeek-R1-Distill-Qwen-7B` (4.7 GB)
- `gpt-oss:20b` (13 GB)

---

## Структура проекта

```
D:/newproject/
├── .env                    # Конфигурация
├── requirements.txt        # Python зависимости
├── comfy_client.py         # ComfyUI API клиент
├── cli_wrappers.py         # CLI обёртки
├── orchestrator.py        # Главный оркестратор
├── launcher.py             # Командный интерфейс
└── workflows/
    └── sdxl_basic.json    # Базовый SDXL workflow
```

---

## Лицензия

MIT License

---

## Авторы

Система создана для локальной автоматизации генерации изображений с использованием ComfyUI и AI-моделей.
