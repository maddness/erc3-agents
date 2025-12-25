import os
import json
from dotenv import load_dotenv
from erc3 import ERC3

# Загрузить переменные окружения из .env файла
load_dotenv(dotenv_path="../.env")

# Проверить наличие API ключа
if not os.getenv("ERC3_API_KEY"):
    print("ERROR: ERC3_API_KEY не найден в .env файле")
    exit(1)

# Инициализация клиента ERC3
core = ERC3()

# Запуск сессии
print("Запуск сессии на платформе ERC3...")
res = core.start_session(
    benchmark="store",
    workspace="my",
    name="Claude Interactive Agent",
    architecture="Claude Code as reasoning agent"
)

print(f"✓ Сессия создана: {res.session_id}")

# Получить список задач
status = core.session_status(res.session_id)
print(f"✓ В сессии {len(status.tasks)} задач(и)")

# Сохранить информацию о сессии для интерактивной работы
session_info = {
    "session_id": res.session_id,
    "tasks": []
}

for idx, task in enumerate(status.tasks, 1):
    print(f"\n{'='*60}")
    print(f"Задача {idx}: {task.task_id}")
    print(f"Спецификация: {task.spec_id}")
    print(f"Текст задачи: {task.task_text}")
    print(f"{'='*60}")

    session_info["tasks"].append({
        "task_id": task.task_id,
        "spec_id": task.spec_id,
        "task_text": task.task_text,
        "completed": False
    })

# Сохранить информацию о сессии
with open("session_state.json", "w", encoding="utf-8") as f:
    json.dump(session_info, f, indent=2, ensure_ascii=False)

print(f"\n✓ Информация о сессии сохранена в session_state.json")
print(f"\nГотов к выполнению задач!")
