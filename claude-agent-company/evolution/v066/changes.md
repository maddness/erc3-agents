# v066 - Changes from v065

Created: 2025-12-01T01:30:00

## Summary
Полная реструктуризация base_prompt в 4-фазный алгоритм + исправления API constraints на основе анализа логов сессии ssn-42PJGqjqNiDAnWFxSSAx4Q.

## Major Changes

### 1. base_prompt: 4-Phase Structure
Заменён плоский список концепций на структурированный алгоритм:
- **PHASE 1: CONTEXT GATHERING** — whoami, классификация запроса
- **PHASE 2: PERMISSION CHECK** — 6 шагов проверки в строгом порядке
- **PHASE 3: DATA RETRIEVAL** — API constraints, стратегии поиска, error handling
- **PHASE 4: RESPONSE FORMATTING** — outcomes, links rules

### 2. Executive Salary Fix
- Type: add_rule + update base_prompt
- Добавлено явное разрешение для Executive изменять ANY salary INCLUDING THEIR OWN
- Исправляет провал задачи ceo_raises_salary

### 3. API Limit Constraint
- Type: add_rule + update tool_patches
- ВСЕ list/search операции MUST use limit=5
- Добавлено CRITICAL предупреждение во все tool_patches
- Исправляет множественные 400 ошибки

### 4. time_log Parameters Fix
- Type: update tool_patch + add_rule
- REQUIRED: work_category='customer_project', status='draft'
- REMOVED: logged_by (не поддерживается)
- Исправляет 400 ошибки при логировании времени

### 5. employee_update_safe Parameter Fix
- Type: update tool_patch + add_rule
- Параметр называется 'id', НЕ 'employee'
- Исправляет 400 ошибки при обновлении зарплаты

### 6. Quick Failure Rule
- Type: add_rule
- После 3+ последовательных ошибок API → сразу error_internal
- Предотвращает бесконечные циклы повторов

### 7. New Examples
Добавлены примеры:
- CEO raises own salary
- Log time for team member as Lead
- Find project by name with pagination

## Files Changed
- base_prompt: полностью переписан
- rules: +5 новых правил
- tool_patches: +6 инструментов с CRITICAL warnings
- examples: +2 новых примера
- rulebook_sections: обновлены roles, sensitivity, scoping, examples

## Expected Impact
- Исправление провалов ceo_raises_salary (denied→ok_answer)
- Уменьшение 400 ошибок на ~80%
- Более быстрое нахождение проектов (сразу limit=5)
- Быстрее выход из error loops
