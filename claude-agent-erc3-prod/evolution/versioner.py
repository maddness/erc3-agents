"""
Versioner - управление версиями конфигурации агента.

Структура:
evolution/
  state.json           # текущая версия, история scores
  v001/
    config.json        # system_prompt, tools, rules, examples
    results/           # результаты прогонов
    analysis.json      # анализ ошибок
    changes.md         # что изменено
"""

import json
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class VersionHistory:
    version: int
    score: float
    tasks_passed: int
    total: int
    timestamp: str


@dataclass
class State:
    current_version: int
    history: list[dict]


class Versioner:
    def __init__(self, base_dir: str = "evolution"):
        self.base_dir = Path(base_dir)
        self.state_file = self.base_dir / "state.json"

    def init_state(self, initial_config: dict) -> None:
        """Инициализировать state.json и v001/ с начальной конфигурацией."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Создать state.json
        state = {
            "current_version": 1,
            "history": []
        }
        self._save_state(state)

        # Создать v001/
        v001_dir = self.base_dir / "v001"
        v001_dir.mkdir(exist_ok=True)
        (v001_dir / "results").mkdir(exist_ok=True)

        # Сохранить начальный config
        config_file = v001_dir / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(initial_config, f, indent=2, ensure_ascii=False)

        # Создать changes.md
        changes_file = v001_dir / "changes.md"
        with open(changes_file, "w", encoding="utf-8") as f:
            f.write("# v001 - Initial Version\n\n")
            f.write(f"Created: {datetime.now().isoformat()}\n\n")
            f.write("## Description\n\n")
            f.write("Initial configuration extracted from agent.py\n")

    def get_state(self) -> dict:
        """Получить текущее состояние."""
        if not self.state_file.exists():
            raise FileNotFoundError(f"State file not found: {self.state_file}")
        with open(self.state_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_state(self, state: dict) -> None:
        """Сохранить состояние."""
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def get_current_version(self) -> int:
        """Получить номер текущей версии."""
        return self.get_state()["current_version"]

    def get_version_dir(self, version: Optional[int] = None) -> Path:
        """Получить директорию версии."""
        if version is None:
            version = self.get_current_version()
        return self.base_dir / f"v{version:03d}"

    def get_current_config(self) -> dict:
        """Загрузить текущую конфигурацию."""
        version_dir = self.get_version_dir()
        config_file = version_dir / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_new_version(
        self,
        new_config: dict,
        analysis: dict,
        changes_description: str,
        score: float = 0.0,
        tasks_passed: int = 0,
        total: int = 16
    ) -> int:
        """
        Сохранить новую версию конфигурации.

        Returns:
            Номер новой версии
        """
        state = self.get_state()
        current_version = state["current_version"]
        new_version = current_version + 1

        # Создать директорию новой версии
        new_dir = self.base_dir / f"v{new_version:03d}"
        new_dir.mkdir(exist_ok=True)
        (new_dir / "results").mkdir(exist_ok=True)

        # Сохранить config
        with open(new_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(new_config, f, indent=2, ensure_ascii=False)

        # Сохранить analysis
        with open(new_dir / "analysis.json", "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        # Создать changes.md
        with open(new_dir / "changes.md", "w", encoding="utf-8") as f:
            f.write(f"# v{new_version:03d} - Changes from v{current_version:03d}\n\n")
            f.write(f"Created: {datetime.now().isoformat()}\n\n")
            f.write("## Changes\n\n")
            f.write(changes_description)

        # Обновить историю предыдущей версии (если есть score)
        if score > 0 or tasks_passed > 0:
            state["history"].append({
                "version": current_version,
                "score": score,
                "tasks_passed": tasks_passed,
                "total": total,
                "timestamp": datetime.now().isoformat()
            })

        # Обновить текущую версию
        state["current_version"] = new_version
        self._save_state(state)

        return new_version

    def record_run_result(self, score: float, tasks_passed: int, total: int = 16) -> None:
        """Записать результат прогона для текущей версии."""
        state = self.get_state()
        current_version = state["current_version"]

        # Проверить, есть ли уже запись для этой версии
        for entry in state["history"]:
            if entry["version"] == current_version:
                # Обновить существующую запись
                entry["score"] = score
                entry["tasks_passed"] = tasks_passed
                entry["total"] = total
                entry["timestamp"] = datetime.now().isoformat()
                self._save_state(state)
                return

        # Добавить новую запись
        state["history"].append({
            "version": current_version,
            "score": score,
            "tasks_passed": tasks_passed,
            "total": total,
            "timestamp": datetime.now().isoformat()
        })
        self._save_state(state)

    def rollback(self) -> int:
        """
        Откатиться к предыдущей версии.

        Returns:
            Номер версии после отката
        """
        state = self.get_state()
        current_version = state["current_version"]

        if current_version <= 1:
            raise ValueError("Cannot rollback from v001")

        # Просто меняем current_version на предыдущую
        state["current_version"] = current_version - 1
        self._save_state(state)

        return state["current_version"]

    def get_history(self) -> list[dict]:
        """Получить историю всех версий."""
        return self.get_state()["history"]

    def get_last_score(self) -> Optional[float]:
        """Получить score последнего прогона."""
        history = self.get_history()
        if not history:
            return None
        return history[-1]["score"]

    def save_session_results(self, session_id: str, results: list[dict]) -> Path:
        """
        Сохранить результаты сессии в директорию текущей версии.

        Returns:
            Путь к сохранённому файлу
        """
        version_dir = self.get_version_dir()
        results_dir = version_dir / "results"
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"{timestamp}_{session_id}.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results_file

    # === Failed Fixes Memory ===

    def _hash_fix(self, fix: dict) -> str:
        """Создать хеш фикса для сравнения."""
        import hashlib
        content = fix.get("content", "")
        fix_type = fix.get("type", "")
        key = f"{fix_type}:{content[:100]}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def record_failed_fix(self, task_id: str, fix: dict, version: int) -> None:
        """Записать фикс который не помог."""
        state = self.get_state()
        if "failed_fixes" not in state:
            state["failed_fixes"] = []

        fix_hash = self._hash_fix(fix)

        # Найти существующую запись или создать новую
        existing = next(
            (f for f in state["failed_fixes"]
             if f["task_id"] == task_id and f["fix_hash"] == fix_hash),
            None
        )

        if existing:
            existing["versions_tried"].append(version)
            existing["last_tried"] = datetime.now().isoformat()
        else:
            state["failed_fixes"].append({
                "task_id": task_id,
                "fix_hash": fix_hash,
                "fix_content": fix.get("content", "")[:200],
                "fix_type": fix.get("type", ""),
                "versions_tried": [version],
                "last_tried": datetime.now().isoformat()
            })

        self._save_state(state)

    def get_failed_fixes_for_task(self, task_id: str) -> list[dict]:
        """Получить все ранее неудачные фиксы для задачи."""
        state = self.get_state()
        return [f for f in state.get("failed_fixes", []) if f["task_id"] == task_id]

    def get_all_failed_fixes(self) -> list[dict]:
        """Получить все неудачные фиксы."""
        state = self.get_state()
        return state.get("failed_fixes", [])

    def clear_failed_fix(self, task_id: str, fix_hash: str) -> bool:
        """Удалить запись о неудачном фиксе (если задача решена другим способом)."""
        state = self.get_state()
        if "failed_fixes" not in state:
            return False

        initial_len = len(state["failed_fixes"])
        state["failed_fixes"] = [
            f for f in state["failed_fixes"]
            if not (f["task_id"] == task_id and f["fix_hash"] == fix_hash)
        ]

        if len(state["failed_fixes"]) < initial_len:
            self._save_state(state)
            return True
        return False
