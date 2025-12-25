# v060 - Changes from v059

Created: 2025-12-01T00:52:25.119282

## Changes

Quick fix для ceo_raises_salary:
- Type: patch_tool
- Content: {'tool': 'employee_update_safe', 'description_addition': 'При обновлении salary ОБЯЗАТЕЛЬНО указывайте параметр \'reason\' с обоснованием изменения (например: \'NY bonus\', \'performance review\', \'annual raise\'). Без reason запрос вернёт 400 Bad Request. Пример: employee_update_safe({"employee": "user_id", "salary": 150000, "reason": "NY bonus"})'}
- Rationale: Агент не знал что для изменения зарплаты нужен обязательный параметр reason. Добавление этой информации в описание инструмента позволит агенту сразу формировать корректные запросы и не повторять ошибочные вызовы.
