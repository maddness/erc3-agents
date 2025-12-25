# v010 - Changes from v009

Created: 2025-11-30T00:45:10.443643

## Changes

Quick fix для broken_system:
- Type: add_rule
- Content: When responding with outcome='error_internal' or 'error_user', the 'links' array MUST be empty. Links should only be provided when the task is successfully completed and they are directly relevant to the requested information. Do not include contextual links (like employee profile or location) that were discovered during troubleshooting but don't answer the user's question.
- Rationale: Явное правило предотвратит добавление нерелевантных ссылок в ошибочные ответы. Агент понял что система сломана, но не знал что links должны быть пустыми при ошибках.

Validation:
  ADDED rule [2025-11-30T00:45:10]: When responding with outcome='error_internal' or 'error_user', the 'links' array...