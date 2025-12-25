"""
Analyzer - анализ провалов агента с помощью Claude Haiku.

Загружает логи задачи, отправляет в Haiku для анализа,
возвращает структурированные выводы и предложения по улучшению.
"""

import os
import json
import httpx
import anthropic
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")


@dataclass
class AnalysisResult:
    """Результат анализа одной задачи."""
    task_id: str
    task_text: str
    passed: bool
    root_cause: Optional[str] = None
    agent_mistake: Optional[str] = None
    missed_context: list[str] = None
    suggested_fix: Optional[dict] = None
    raw_response: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_text": self.task_text,
            "passed": self.passed,
            "root_cause": self.root_cause,
            "agent_mistake": self.agent_mistake,
            "missed_context": self.missed_context or [],
            "suggested_fix": self.suggested_fix,
            "raw_response": self.raw_response
        }


ANALYSIS_PROMPT = """Ты анализируешь провал AI-агента для корпоративной системы.

## Контекст системы
Агент работает в Aetherion Analytics GmbH - AI консалтинговой компании.
У агента есть доступ к API для работы с сотрудниками, проектами, клиентами, wiki.
Агент должен соблюдать уровни доступа (Level 1-3) и не раскрывать чувствительные данные.

## Задача которую получил агент
{task_text}

## Информация о пользователе (whoami)
{whoami_info}

## Действия агента (tool_calls)
{tool_calls_summary}

## Рассуждения агента (text_blocks)
{text_blocks_summary}

## Финальный ответ агента
{final_response}

## Результат оценки
{eval_result}

## Твоя задача
Проанализируй почему агент провалил задачу. Ответь СТРОГО в JSON формате:

```json
{{
  "root_cause": "Краткое описание корневой причины провала (1-2 предложения)",
  "agent_mistake": "Что конкретно агент сделал неправильно или не сделал",
  "missed_context": ["Список вещей которые агент не учёл или пропустил"],
  "suggested_fix": {{
    "type": "add_rule|patch_tool|add_example|update_prompt",
    "priority": "high|medium|low",
    "content": "Конкретное предложение по исправлению",
    "rationale": "Почему это поможет"
  }}
}}
```

ВАЖНО:
- Будь конкретным, не давай общих советов
- suggested_fix.type:
  - "add_rule" - добавить правило в список rules (для поведенческих ошибок)
  - "patch_tool" - изменить описание инструмента (если агент не понял как использовать)
  - "add_example" - добавить пример в examples (для повторяющихся паттернов)
  - "update_prompt" - изменить base_prompt (для фундаментальных проблем)
- content должен быть готов к прямому использованию
"""


class Analyzer:
    """Анализатор провалов агента."""

    def __init__(self, model: str = "claude-opus-4-5-20251101"):
        self.model = model
        base_url = os.getenv("ANTHROPIC_BASE_URL")
        http_client = httpx.Client(verify=False)
        self.client = anthropic.Anthropic(
            base_url=base_url,
            http_client=http_client
        ) if base_url else anthropic.Anthropic()

    def analyze_task(self, task_log: dict, eval_result: dict) -> AnalysisResult:
        """
        Анализировать один провал задачи.

        Args:
            task_log: Лог задачи из TaskLogger
            eval_result: Результат оценки от ERC3

        Returns:
            AnalysisResult с выводами
        """
        task_id = task_log.get("task_id", "unknown")
        task_text = task_log.get("task_text", "")

        # Если задача прошла - не анализируем
        passed = eval_result.get("passed", False)
        if passed:
            return AnalysisResult(
                task_id=task_id,
                task_text=task_text,
                passed=True
            )

        # Извлечь whoami
        whoami_info = self._extract_whoami(task_log)

        # Извлечь tool_calls
        tool_calls_summary = self._extract_tool_calls(task_log)

        # Извлечь text_blocks (рассуждения)
        text_blocks_summary = self._extract_text_blocks(task_log)

        # Извлечь финальный ответ
        final_response = self._extract_final_response(task_log)

        # Форматировать eval_result
        eval_str = json.dumps(eval_result, ensure_ascii=False, indent=2)

        # Построить промпт
        prompt = ANALYSIS_PROMPT.format(
            task_text=task_text,
            whoami_info=whoami_info,
            tool_calls_summary=tool_calls_summary,
            text_blocks_summary=text_blocks_summary,
            final_response=final_response,
            eval_result=eval_str
        )

        # Вызвать Haiku
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )

            raw_text = response.content[0].text
            analysis = self._parse_response(raw_text)

            return AnalysisResult(
                task_id=task_id,
                task_text=task_text,
                passed=False,
                root_cause=analysis.get("root_cause"),
                agent_mistake=analysis.get("agent_mistake"),
                missed_context=analysis.get("missed_context", []),
                suggested_fix=analysis.get("suggested_fix"),
                raw_response=raw_text
            )

        except Exception as e:
            return AnalysisResult(
                task_id=task_id,
                task_text=task_text,
                passed=False,
                root_cause=f"Ошибка анализа: {str(e)}",
                raw_response=str(e)
            )

    def analyze_session(self, logs_file: Path, eval_results: list[dict]) -> list[AnalysisResult]:
        """
        Анализировать все провалы сессии.

        Args:
            logs_file: Путь к файлу логов сессии
            eval_results: Список результатов оценки для каждой задачи

        Returns:
            Список AnalysisResult для провальных задач
        """
        with open(logs_file, "r", encoding="utf-8") as f:
            logs = json.load(f)

        results = []

        # Сопоставить логи с eval_results по task_id
        eval_by_task = {r.get("task_id", r.get("id")): r for r in eval_results}

        for task_log in logs:
            task_id = task_log.get("task_id")
            eval_result = eval_by_task.get(task_id, {"passed": False})

            if not eval_result.get("passed", False):
                result = self.analyze_task(task_log, eval_result)
                results.append(result)

        return results

    def _extract_whoami(self, task_log: dict) -> str:
        """Извлечь информацию whoami из логов."""
        logs = task_log.get("logs", [])

        for entry in logs:
            # Новый формат: llm_turn с tool_calls
            if entry.get("type") == "llm_turn":
                for tc in entry.get("tool_calls", []):
                    if tc.get("tool") == "whoami":
                        output = tc.get("output", {})
                        if isinstance(output, dict):
                            return json.dumps(output, ensure_ascii=False, indent=2)
                        return str(output)

            # Старый формат: tool_result
            if entry.get("type") == "tool_result":
                name = entry.get("name", "")
                if name == "whoami":
                    result = entry.get("result", {})
                    if isinstance(result, dict):
                        return json.dumps(result, ensure_ascii=False, indent=2)
                    return str(result)

        return "Не найдено"

    def _extract_tool_calls(self, task_log: dict) -> str:
        """Извлечь summary всех tool_calls."""
        logs = task_log.get("logs", [])
        calls = []

        for entry in logs:
            if entry.get("type") == "llm_turn":
                for tc in entry.get("tool_calls", []):
                    # Новый формат: tool, input
                    name = tc.get("tool") or tc.get("name", "unknown")
                    args = tc.get("input") or tc.get("arguments", {})
                    args_str = json.dumps(args, ensure_ascii=False)
                    if len(args_str) > 200:
                        args_str = args_str[:200] + "..."
                    calls.append(f"- {name}({args_str})")

        if not calls:
            return "Нет вызовов инструментов"

        return "\n".join(calls)

    def _extract_text_blocks(self, task_log: dict) -> str:
        """Извлечь рассуждения агента из text_blocks."""
        logs = task_log.get("logs", [])
        blocks = []

        for entry in logs:
            if entry.get("type") == "llm_turn":
                turn = entry.get("turn", "?")
                for text in entry.get("text_blocks", []):
                    if text.strip():
                        # Укоротить если слишком длинный
                        if len(text) > 500:
                            text = text[:500] + "..."
                        blocks.append(f"[Turn {turn}]: {text}")

        if not blocks:
            return "Нет рассуждений"

        return "\n\n".join(blocks)

    def _extract_final_response(self, task_log: dict) -> str:
        """Извлечь финальный ответ агента."""
        logs = task_log.get("logs", [])

        # Ищем последний respond() вызов в новом формате
        for entry in reversed(logs):
            if entry.get("type") == "llm_turn":
                for tc in entry.get("tool_calls", []):
                    if tc.get("tool") == "respond":
                        # input содержит аргументы respond()
                        args = tc.get("input", {})
                        return json.dumps(args, ensure_ascii=False, indent=2)

        # Старый формат: tool_result
        for entry in reversed(logs):
            if entry.get("type") == "tool_result":
                name = entry.get("name", "")
                if name == "respond":
                    args = entry.get("arguments", {})
                    return json.dumps(args, ensure_ascii=False, indent=2)

        # Или последний text_block
        for entry in reversed(logs):
            if entry.get("type") == "llm_turn":
                texts = entry.get("text_blocks", [])
                if texts:
                    return texts[-1]

        return "Не найдено"

    def _parse_response(self, text: str) -> dict:
        """Парсить JSON из ответа Haiku."""
        # Попробовать найти JSON в тексте
        try:
            # Прямой парсинг
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Найти JSON между ```json и ```
        import re
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Найти любой JSON-подобный блок
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {"root_cause": "Не удалось распарсить ответ", "raw": text}

    def summarize_failures(self, results: list[AnalysisResult]) -> dict:
        """
        Создать сводку по всем провалам.

        Returns:
            dict с паттернами ошибок и приоритизированными fixes
        """
        failed = [r for r in results if not r.passed]

        if not failed:
            return {"total_failed": 0, "patterns": [], "top_fixes": []}

        # Группировать по root_cause
        causes = {}
        for r in failed:
            cause = r.root_cause or "unknown"
            if cause not in causes:
                causes[cause] = []
            causes[cause].append(r.task_id)

        # Собрать все fixes
        fixes = []
        for r in failed:
            if r.suggested_fix:
                fixes.append({
                    "task_id": r.task_id,
                    **r.suggested_fix
                })

        # Сортировать fixes по priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        fixes.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))

        return {
            "total_failed": len(failed),
            "patterns": [
                {"cause": cause, "count": len(tasks), "tasks": tasks}
                for cause, tasks in sorted(causes.items(), key=lambda x: -len(x[1]))
            ],
            "top_fixes": fixes[:5]  # Top 5 fixes
        }
