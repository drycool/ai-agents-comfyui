# Практическое руководство

## Варианты запуска системы

### Вариант 1: Командная строка (CLI)

Запуск через терминал с прямым указанием промпта.

```bash
# Активировать виртуальное окружение (если есть)
# source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate     # Windows

# Простой режим - быстрая генерация
python launcher.py "кот на подоконнике"

# Улучшенный режим - с стилизацией
python launcher.py "футуристичный город" --mode enhanced

# Полный режим - максимальное качество
python launcher.py "абстрактное искусство" --mode full
```

### Вариант 2: Программный API

Использование в Python-скриптах.

```python
from orchestrator import OrchestratorAgent

# Инициализация
orch = OrchestratorAgent()

# Простой вызов
result = orch.process_simple("пейзаж гор")

# С параметрами
result = orch.process_enhanced(
    "котик",
    width=512,
    height=512,
    steps=20,
    cfg=7.0
)

# Результат
print(f"Статус: {result['status']}")
print(f"Промпт: {result['prompt']}")
print(f"Изображения: {result['images']}")
```

### Вариант 3: Прямое использование клиентов

#### ComfyUI клиент
```python
from comfy_client import ComfyAPIClient

comfy = ComfyAPIClient()

# Создать свой workflow
workflow = comfy.create_simple_sdxl_workflow(
    prompt="красивый закат",
    width=1024,
    height=1024,
    steps=20
)

# Выполнить
result = comfy.execute_workflow(workflow, wait=True)
```

#### Ollama
```python
from cli_wrappers import OllamaWrapper

# Простой запрос
response = OllamaWrapper.run("Что такое Stable Diffusion?")

# С конкретной моделью
response = OllamaWrapper.run("Привет", model="qwen2.5-coder:7b")

# Проверить модели
models = OllamaWrapper.list_models()
```

#### Claude
```python
from cli_wrappers import ClaudeWrapper

# Улучшить промпт
enhanced = ClaudeWrapper.enhance_prompt(
    "кот",
    style="фотореалистичный"
)

# Сгенерировать workflow
workflow = ClaudeWrapper.generate_workflow(
    "создай workflow для SDXL"
)
```

#### Gemini
```python
from cli_wrappers import GeminiWrapper

# Предложить стиль
style = GeminiWrapper.suggest_style("футуристичный город")

# Проанализировать изображение
analysis = GeminiWrapper.analyze_image("path/to/image.png")
```

---

## Проверка статуса системы

```bash
# Проверить все подключения
python launcher.py --status
python launcher.py -s

# Список моделей Ollama
python launcher.py --list-models
python launcher.py -l
```

---

## Настройка параметров генерации

### Через .env файл
```env
# Размер изображения
DEFAULT_WIDTH=1024
DEFAULT_HEIGHT=1024

# Качество (больше = лучше, но медленнее)
DEFAULT_STEPS=20

# CFG Scale (1-20)
DEFAULT_CFG=7.0

# Модель по умолчанию
OLLAMA_DEFAULT_MODEL=llama3.2
```

### Через командную строку
```bash
# С указанием параметров
python launcher.py "запрос" --width 512 --height 512 --steps 10
```

---

## Типичные сценарии использования

### Сценарий 1: Быстрая генерация
Используйте простой режим для тестирования идей.
```bash
python launcher.py "кот" --mode simple
```

### Сценарий 2: Художественная генерация
Используйте улучшенный режим для качественных артов.
```bash
python launcher.py "дракон в лесу" --mode enhanced
```

### Сценарий 3: Пакетная генерация
Создайте скрипт для множества изображений.
```python
from orchestrator import OrchestratorAgent

orch = OrchestratorAgent()
prompts = ["кот", "собака", "птица"]

for prompt in prompts:
    result = orch.process_enhanced(prompt)
    print(f"Сохранено: {result['images']}")
```

---

## Структура выходных файлов

```
output/
├── 2026-02-22_15-30-00_cat.png    # timestamp_prompt.png
├── 2026-02-22_15-35-00_dog.png
└── ...
```

---

## Устранение проблем

### Ошибка: "ComfyUI not connected"
```bash
# Запустить ComfyUI
cd путь/к/ComfyUI
python main.py --listen 0.0.0.0 --port 8188

# Проверить доступность
curl http://localhost:8188/system_stats
```

### Ошибка: "Ollama not available"
```bash
# Запустить Ollama
ollama serve

# Проверить модели
ollama list
```

### Ошибка: "Claude/Gemini CLI not found"
```bash
# Проверить пути в .env
CLAUDE_CLI_PATH=claude
GEMINI_CLI_PATH=gemini
```

---

## Расширенные возможности

### Создание собственного workflow
```python
from comfy_client import ComfyAPIClient

comfy = ComfyAPIClient()

# Создать кастомный workflow
workflow = {
    "1": {
        "inputs": {"text": "ваш промпт", "clip": ["4", 0]},
        "class_type": "CLIPTextEncode"
    },
    # ... добавьте узлы
    "output": {
        "inputs": {"images": ["3", 0]},
        "class_type": "SaveImage"
    }
}

comfy.execute_workflow(workflow)
```

### Добавление нового агента
```python
# В cli_wrappers.py
class MyAgentWrapper:
    @staticmethod
    def run(prompt: str) -> str:
        # Ваша логика
        return result

# В orchestrator.py
self.my_agent = MyAgentWrapper()
```
