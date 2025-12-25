"""
Evolver - генерация улучшений конфигурации на основе анализа ошибок.

Принимает текущий config и результаты анализа,
генерирует новый config с исправлениями.
"""

import os
import json
import httpx
import anthropic
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from datetime import datetime
from .analyzer import AnalysisResult

load_dotenv(dotenv_path="../.env")


def get_rule_text(rule) -> str:
    """Получить текст правила (поддержка старого и нового формата)."""
    if isinstance(rule, dict):
        return rule.get("text", "")
    return str(rule)


def get_rule_timestamp(rule) -> str:
    """Получить timestamp правила."""
    if isinstance(rule, dict):
        return rule.get("added_at", "unknown")
    return "legacy"


def make_rule(text: str) -> dict:
    """Создать правило с timestamp."""
    return {
        "text": text,
        "added_at": datetime.now().isoformat(timespec='seconds')
    }


# Список валидных инструментов агента для валидации
VALID_TOOLS = {
    "whoami",
    "respond",
    "employees_list",
    "employees_search",
    "employees_get",
    "employees_update",
    "wiki_list",
    "wiki_load",
    "wiki_search",
    "customers_list",
    "customers_get",
    "customers_search",
    "projects_list",
    "projects_get",
    "projects_search",
    "projects_status_update",
    "get_rulebook",
}


DEDUP_PROMPT = """Сравни два правила для AI-агента и определи, являются ли они дубликатами или очень похожими.

Существующее правило:
{existing_rule}

Новое правило:
{new_rule}

Ответь ТОЛЬКО одним словом:
- DUPLICATE - если правила говорят об одном и том же
- SIMILAR - если правила пересекаются на >70%
- UNIQUE - если правила достаточно разные

Ответ:"""


CONSOLIDATE_PROMPT = """Проанализируй список правил для AI-агента и объедини похожие/дублирующиеся правила.

ВАЖНО: Правила добавлялись ПОСЛЕДОВАТЕЛЬНО во времени. Более новые правила (с более поздней датой) имеют БОЛЬШИЙ ВЕС - они уточняют, исправляют или дополняют предыдущие. При объединении:
- Сохраняй детали из НОВЫХ правил (они появились для исправления конкретных ошибок)
- Старые правила могут быть неполными или неточными
- Если новое правило противоречит старому - доверяй новому

Текущие правила (от старых к новым):
{rules_list}

Задача:
1. Найди правила которые говорят об одном и том же
2. Объедини их, ПРИОРИТИЗИРУЯ формулировки из НОВЫХ правил
3. Сохрани уникальные детали которые появились в поздних правилах

Ответь в JSON формате:
```json
{{
  "consolidated_rules": [
    "Объединённое правило 1",
    "Объединённое правило 2"
  ],
  "removed_count": 3,
  "reasoning": "Краткое объяснение что было объединено"
}}
```
"""


@dataclass
class EvolutionResult:
    """Результат эволюции."""
    success: bool
    new_config: Optional[dict] = None
    changes_description: str = ""
    applied_fixes: list[dict] = None
    raw_response: Optional[str] = None

    def __post_init__(self):
        if self.applied_fixes is None:
            self.applied_fixes = []


EVOLUTION_PROMPT = """Ты эволюционируешь конфигурацию AI-агента на основе анализа его ошибок.

## Текущая конфигурация

### base_prompt
{base_prompt}

### rules (текущие правила)
{current_rules}

### tool_patches (патчи описаний инструментов)
{current_patches}

### examples (примеры для промпта)
{current_examples}

## Анализ провалов

{analysis_summary}

## Детали провалов

{failures_details}

## Твоя задача

На основе анализа провалов, сгенерируй ИЗМЕНЕНИЯ для конфигурации.
Ответь СТРОГО в JSON формате:

```json
{{
  "reasoning": "Краткое объяснение логики изменений",
  "changes": {{
    "add_rules": [
      "Новое правило 1 (краткое, actionable)",
      "Новое правило 2"
    ],
    "remove_rules": [
      "Текст правила для удаления (точное совпадение)"
    ],
    "add_tool_patches": {{
      "tool_name": "Новое/дополненное описание инструмента"
    }},
    "add_examples": [
      {{
        "input": "Пример запроса пользователя",
        "output": "Ожидаемое поведение/ответ агента",
        "explanation": "Почему это правильно"
      }}
    ],
    "update_base_prompt": null
  }},
  "expected_impact": "Какие задачи это должно исправить"
}}
```

ВАЖНЫЕ ПРАВИЛА:
1. Правила должны быть короткими и конкретными (1-2 предложения)
2. НЕ дублируй существующие правила
3. НЕ добавляй правила которые уже есть в base_prompt
4. Фокусируйся на КОНКРЕТНЫХ ошибках из анализа
5. Максимум 3 новых правила за раз (лучше меньше, но точнее)
6. tool_patches дополняют, а не заменяют базовое описание
7. examples должны быть реалистичными для контекста Aetherion Analytics
8. update_base_prompt используй ТОЛЬКО для критических изменений

Если нет уверенных исправлений - лучше вернуть пустые массивы чем плохие правила.
"""


class Evolver:
    """Генератор эволюций конфигурации."""

    def __init__(self, model: str = "claude-opus-4-5-20251101"):
        self.model = model
        base_url = os.getenv("ANTHROPIC_BASE_URL")
        http_client = httpx.Client(verify=False)
        self.client = anthropic.Anthropic(
            base_url=base_url,
            http_client=http_client
        ) if base_url else anthropic.Anthropic()

    def _validate_tool_names(self, text: str) -> list[str]:
        """
        Проверить текст на упоминание несуществующих инструментов.
        Возвращает список невалидных имён.
        """
        import re
        # Ищем паттерны типа "tool_name()", "через tool_name", "используй tool_name"
        potential_tools = re.findall(r'\b([a-z_]+(?:_[a-z]+)*)\s*\(', text.lower())
        potential_tools += re.findall(r'(?:через|используй|вызови|метод|функци[яю])\s+([a-z_]+(?:_[a-z]+)*)', text.lower())

        invalid = []
        for tool in potential_tools:
            # Проверяем только если похоже на имя инструмента
            if '_' in tool or tool in ['whoami', 'respond']:
                if tool not in VALID_TOOLS:
                    invalid.append(tool)

        return list(set(invalid))

    def _check_duplicate(self, new_rule: str, existing_rules: list[str]) -> tuple[bool, Optional[str]]:
        """
        Проверить, является ли новое правило дубликатом существующих.
        Использует LLM для семантического сравнения.

        Returns:
            (is_duplicate, similar_rule) - флаг и текст похожего правила если найден
        """
        if not existing_rules:
            return False, None

        # Быстрая проверка на точное совпадение
        if new_rule in existing_rules:
            return True, new_rule

        # Быстрая эвристика - проверка ключевых слов
        new_keywords = set(new_rule.lower().split())
        for existing in existing_rules:
            existing_keywords = set(existing.lower().split())
            overlap = len(new_keywords & existing_keywords) / max(len(new_keywords), 1)
            if overlap > 0.7:
                # Высокое пересечение - проверяем через LLM
                try:
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=50,
                        messages=[{
                            "role": "user",
                            "content": DEDUP_PROMPT.format(
                                existing_rule=existing,
                                new_rule=new_rule
                            )
                        }]
                    )
                    result = response.content[0].text.strip().upper()
                    if result in ["DUPLICATE", "SIMILAR"]:
                        return True, existing
                except Exception:
                    pass

        return False, None

    def consolidate_rules(self, rules: list) -> tuple[list, str]:
        """
        Консолидировать список правил, объединяя похожие.

        Returns:
            (consolidated_rules, reasoning)
        """
        if len(rules) <= 3:
            return rules, "Слишком мало правил для консолидации"

        # Форматировать правила с датами для LLM
        formatted_rules = []
        for i, rule in enumerate(rules):
            text = get_rule_text(rule)
            timestamp = get_rule_timestamp(rule)
            formatted_rules.append(f"{i+1}. [{timestamp}] {text}")

        rules_list = "\n".join(formatted_rules)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": CONSOLIDATE_PROMPT.format(rules_list=rules_list)
                }]
            )

            raw_text = response.content[0].text
            result = self._parse_response(raw_text)

            if result and "consolidated_rules" in result:
                # Валидация tool names в консолидированных правилах
                valid_rules = []
                rejected = []
                for text in result["consolidated_rules"]:
                    invalid_tools = self._validate_tool_names(text)
                    if invalid_tools:
                        rejected.append(f"{text[:50]}... (invalid: {invalid_tools})")
                    else:
                        valid_rules.append(make_rule(text))

                reasoning = result.get("reasoning", "")
                removed = result.get("removed_count", len(rules) - len(valid_rules))

                if rejected:
                    reasoning += f" REJECTED {len(rejected)} rules with invalid tools."

                return valid_rules, f"Удалено {removed} дублей. {reasoning}"

        except Exception as e:
            return rules, f"Ошибка консолидации: {e}"

        return rules, "Не удалось консолидировать"

    def evolve(
        self,
        current_config: dict,
        analysis_results: list[AnalysisResult],
        analysis_summary: dict
    ) -> EvolutionResult:
        """
        Сгенерировать новую версию конфигурации.

        Args:
            current_config: Текущая конфигурация
            analysis_results: Результаты анализа провалов
            analysis_summary: Сводка от Analyzer.summarize_failures()

        Returns:
            EvolutionResult с новым config
        """
        # Подготовить контекст
        base_prompt = current_config.get("base_prompt", "")
        current_rules = current_config.get("rules", [])
        current_patches = current_config.get("tool_patches", {})
        current_examples = current_config.get("examples", [])

        # Форматировать для промпта
        rules_str = "\n".join(f"- {r}" for r in current_rules) if current_rules else "(пусто)"
        patches_str = json.dumps(current_patches, ensure_ascii=False, indent=2) if current_patches else "(пусто)"
        examples_str = json.dumps(current_examples, ensure_ascii=False, indent=2) if current_examples else "(пусто)"

        # Сводка анализа
        summary_str = json.dumps(analysis_summary, ensure_ascii=False, indent=2)

        # Детали провалов
        failures = []
        for r in analysis_results:
            if not r.passed:
                failures.append({
                    "task_id": r.task_id,
                    "task_text": r.task_text[:200] + "..." if len(r.task_text) > 200 else r.task_text,
                    "root_cause": r.root_cause,
                    "agent_mistake": r.agent_mistake,
                    "suggested_fix": r.suggested_fix
                })
        failures_str = json.dumps(failures, ensure_ascii=False, indent=2)

        # Построить промпт
        prompt = EVOLUTION_PROMPT.format(
            base_prompt=base_prompt[:2000] + "..." if len(base_prompt) > 2000 else base_prompt,
            current_rules=rules_str,
            current_patches=patches_str,
            current_examples=examples_str,
            analysis_summary=summary_str,
            failures_details=failures_str
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )

            raw_text = response.content[0].text
            changes = self._parse_response(raw_text)

            if not changes or "changes" not in changes:
                return EvolutionResult(
                    success=False,
                    changes_description="Не удалось распарсить ответ",
                    raw_response=raw_text
                )

            # Применить изменения
            new_config = self._apply_changes(current_config, changes["changes"])

            # Извлечь лог валидации
            validation_log = new_config.pop("_validation_log", [])

            # Сформировать описание
            description = self._format_changes_description(changes)
            if validation_log:
                description += "\n\n### Validation Log\n" + "\n".join(f"- {log}" for log in validation_log)

            return EvolutionResult(
                success=True,
                new_config=new_config,
                changes_description=description,
                applied_fixes=changes.get("changes", {}),
                raw_response=raw_text
            )

        except Exception as e:
            return EvolutionResult(
                success=False,
                changes_description=f"Ошибка: {str(e)}",
                raw_response=str(e)
            )

    def _apply_changes(self, config: dict, changes: dict) -> dict:
        """Применить изменения к конфигурации с валидацией и дедупликацией."""
        new_config = {
            "version": config.get("version", 1) + 1,
            "description": config.get("description", ""),
            "base_prompt": config.get("base_prompt", ""),
            "rules": list(config.get("rules", [])),
            "tool_patches": dict(config.get("tool_patches", {})),
            "examples": list(config.get("examples", [])),
            "rulebook_sections": config.get("rulebook_sections", {}),
            "_validation_log": []  # Лог валидации для отладки
        }

        validation_log = new_config["_validation_log"]

        # Получить тексты существующих правил для сравнения
        existing_texts = [get_rule_text(r) for r in new_config["rules"]]

        # Добавить новые правила с валидацией и дедупликацией
        add_rules = changes.get("add_rules", [])
        for rule_text in add_rules:
            if not rule_text:
                continue

            # 1. Валидация tool names
            invalid_tools = self._validate_tool_names(rule_text)
            if invalid_tools:
                validation_log.append(f"REJECTED rule (invalid tools {invalid_tools}): {rule_text[:80]}...")
                continue

            # 2. Проверка на точный дубликат (по тексту)
            if rule_text in existing_texts:
                validation_log.append(f"SKIPPED exact duplicate: {rule_text[:80]}...")
                continue

            # 3. Семантическая проверка на дубликат
            is_dup, similar = self._check_duplicate(rule_text, existing_texts)
            if is_dup:
                validation_log.append(f"SKIPPED semantic duplicate of '{similar[:50]}...': {rule_text[:80]}...")
                continue

            # Правило прошло валидацию - добавляем с timestamp
            new_rule = make_rule(rule_text)
            new_config["rules"].append(new_rule)
            existing_texts.append(rule_text)
            validation_log.append(f"ADDED rule [{new_rule['added_at']}]: {rule_text[:80]}...")

        # Удалить правила (по тексту)
        remove_rules = changes.get("remove_rules", [])
        for rule_text in remove_rules:
            for i, existing in enumerate(new_config["rules"]):
                if get_rule_text(existing) == rule_text:
                    new_config["rules"].pop(i)
                    validation_log.append(f"REMOVED rule: {rule_text[:80]}...")
                    break

        # Добавить tool patches с валидацией имени инструмента
        add_patches = changes.get("add_tool_patches", {})
        for tool_name, description in add_patches.items():
            if tool_name not in VALID_TOOLS:
                validation_log.append(f"REJECTED patch (invalid tool '{tool_name}'): {description[:50]}...")
                continue
            new_config["tool_patches"][tool_name] = description
            validation_log.append(f"ADDED patch for {tool_name}")

        # Добавить examples
        add_examples = changes.get("add_examples", [])
        for example in add_examples:
            if example and example not in new_config["examples"]:
                new_config["examples"].append(example)
                validation_log.append(f"ADDED example: {str(example)[:50]}...")

        # Обновить base_prompt (если указано)
        update_prompt = changes.get("update_base_prompt")
        if update_prompt:
            new_config["base_prompt"] = update_prompt
            validation_log.append("UPDATED base_prompt")

        # Консолидация если правил слишком много
        if len(new_config["rules"]) > 5:
            consolidated, reason = self.consolidate_rules(new_config["rules"])
            if len(consolidated) < len(new_config["rules"]):
                new_config["rules"] = consolidated
                validation_log.append(f"CONSOLIDATED: {reason}")

        return new_config

    def _format_changes_description(self, changes: dict) -> str:
        """Форматировать описание изменений для changes.md."""
        lines = []

        reasoning = changes.get("reasoning", "")
        if reasoning:
            lines.append(f"**Reasoning:** {reasoning}\n")

        c = changes.get("changes", {})

        add_rules = c.get("add_rules", [])
        if add_rules:
            lines.append("### Added Rules")
            for rule in add_rules:
                lines.append(f"- {rule}")
            lines.append("")

        remove_rules = c.get("remove_rules", [])
        if remove_rules:
            lines.append("### Removed Rules")
            for rule in remove_rules:
                lines.append(f"- ~~{rule}~~")
            lines.append("")

        add_patches = c.get("add_tool_patches", {})
        if add_patches:
            lines.append("### Tool Patches")
            for tool, desc in add_patches.items():
                lines.append(f"- **{tool}**: {desc[:100]}...")
            lines.append("")

        add_examples = c.get("add_examples", [])
        if add_examples:
            lines.append("### Added Examples")
            for ex in add_examples:
                lines.append(f"- Input: {ex.get('input', '')[:50]}...")
            lines.append("")

        expected = changes.get("expected_impact", "")
        if expected:
            lines.append(f"### Expected Impact\n{expected}")

        return "\n".join(lines)

    def _parse_response(self, text: str) -> dict:
        """Парсить JSON из ответа."""
        import re

        # Попробовать прямой парсинг
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Найти JSON между ```json и ```
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Найти JSON между { и }
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {}

    def quick_fix(self, current_config: dict, single_failure: AnalysisResult) -> EvolutionResult:
        """
        Быстрый fix для одной конкретной ошибки.
        Используется в focused mode для итерации по одной задаче.
        """
        if single_failure.passed:
            return EvolutionResult(
                success=False,
                changes_description="Задача прошла, fix не нужен"
            )

        fix = single_failure.suggested_fix
        if not fix:
            return EvolutionResult(
                success=False,
                changes_description="Нет suggested_fix в анализе"
            )

        # Применить suggested_fix напрямую
        changes = {
            "add_rules": [],
            "remove_rules": [],
            "add_tool_patches": {},
            "add_examples": [],
            "update_base_prompt": None
        }

        fix_type = fix.get("type", "")
        content = fix.get("content", "")

        if fix_type == "add_rule" and content:
            changes["add_rules"] = [content]
        elif fix_type == "patch_tool" and content:
            # Предполагаем формат "tool_name: description"
            if ":" in content:
                tool_name, desc = content.split(":", 1)
                changes["add_tool_patches"] = {tool_name.strip(): desc.strip()}
        elif fix_type == "add_example" and content:
            changes["add_examples"] = [{"input": content, "output": "", "explanation": fix.get("rationale", "")}]
        elif fix_type == "update_prompt" and content:
            changes["update_base_prompt"] = content

        new_config = self._apply_changes(current_config, changes)

        # Проверить лог валидации
        validation_log = new_config.pop("_validation_log", [])
        was_rejected = any("REJECTED" in log or "SKIPPED" in log for log in validation_log)

        description = f"Quick fix для {single_failure.task_id}:\n"
        description += f"- Type: {fix_type}\n"
        description += f"- Content: {content}\n"
        description += f"- Rationale: {fix.get('rationale', 'N/A')}\n"

        if validation_log:
            description += f"\nValidation:\n" + "\n".join(f"  {log}" for log in validation_log)

        # Если fix был отклонён - сообщить об этом
        if was_rejected and not any("ADDED" in log for log in validation_log):
            return EvolutionResult(
                success=False,
                changes_description=description + "\n\nFix был отклонён валидацией.",
                applied_fixes=[]
            )

        return EvolutionResult(
            success=True,
            new_config=new_config,
            changes_description=description,
            applied_fixes=[fix]
        )
