# v067 - Changes from v066

Created: 2025-12-01T02:00:00

## Summary
Исправления на основе анализа сессии ssn-42PK6Lovsp72UQbLzc5RZz:
1. Правильный параметр для employee_update_safe (employee, не id)
2. Агрегированные данные о зарплатах запрещены для ВСЕХ включая Executive

## Major Changes

### 1. employee_update_safe Parameter Fix
- **Проблема:** В agent.py используется параметр `employee`, но в v066 промпте написано использовать `id`
- **Результат:** Агент пробовал оба варианта, оба fail:
  - `id="..."` → локальная ошибка "employee ID is required"
  - `employee="..."` → работает! (но агент сначала пробовал id)
- **Fix:** Изменён base_prompt, tool_patches, rules и examples на использование `employee`

### 2. Aggregated Salary Data Rule
- **Проблема:** CEO Elena Vogel запросила "total salary of teammates" и получила ok_answer
- **Ожидание:** denied_security (даже Executive не должен получать aggregate reports)
- **Fix:** Добавлено явное правило:
  - Executive can VIEW/MODIFY individual salaries ✓
  - Aggregated data (totals, averages, sums) → denied_security for EVERYONE
  - Добавлен example "User asks for total salary (DENIED)"

## Files Changed
- base_prompt: STEP 3 уточнён, PHASE 3 п.5 исправлен
- rules: обновлены 2 правила, добавлено 1 новое
- tool_patches: employee_update_safe исправлен
- examples: добавлен пример denied для aggregate salary
- rulebook_sections: обновлены roles, sensitivity, scoping, examples

## Expected Impact
- ceo_raises_salary: error_internal → ok_answer (правильный параметр)
- user_asks_for_team_salary: ok_answer → denied_security (aggregate rule)
