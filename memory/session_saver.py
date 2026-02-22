# -*- coding: utf-8 -*-
"""Утилита для сохранения прогресса сессии"""

import os
from datetime import datetime

SESSION_LOG = "memory/session_history.md"


def save_session(summary: str, next_steps: str = ""):
    """Сохранить итоги текущей сессии"""
    os.makedirs("memory", exist_ok=True)

    entry = f"""
---

### {datetime.now().strftime("%Y-%m-%d %H:%M")}

**Сделано:**
{summary}

**Следующие шаги:**
{next_steps}
"""

    if os.path.exists(SESSION_LOG):
        with open(SESSION_LOG, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = "# История сессий\n"

    with open(SESSION_LOG, "w", encoding="utf-8") as f:
        f.write(entry + content)

    print(f"Сессия сохранена в {SESSION_LOG}")


if __name__ == "__main__":
    import sys
    summary = sys.argv[1] if len(sys.argv) > 1 else "Без описания"
    next_steps = sys.argv[2] if len(sys.argv) > 2 else ""
    save_session(summary, next_steps)
