#!/bin/bash
# Запуск агента claude-agent-erc3-prod
#
# Варианты использования:
#   ./run.sh task t017           - одна задача
#   ./run.sh tasks t013,t014     - несколько задач
#   ./run.sh all                 - все задачи последовательно
#   ./run.sh parallel 5          - все задачи параллельно (5 воркеров)
#   ./run.sh version             - текущая версия конфига
#   ./run.sh failed              - упавшие задачи из последней сессии
#   ./run.sh log t017            - лог конкретной задачи

cd "$(dirname "$0")"

case "$1" in
  # Одна задача
  task)
    ./venv/bin/python3 -c "
from agent import ERC3Agent
agent = ERC3Agent()
agent.run_session(benchmark='erc3-prod', workspace='erc3-prod', task_filter=['$2'])
"
    ;;

  # Несколько задач (через запятую: t013,t014,t017)
  tasks)
    TASKS=$(echo "$2" | sed "s/,/', '/g")
    ./venv/bin/python3 -c "
from agent import ERC3Agent
agent = ERC3Agent()
agent.run_session(benchmark='erc3-prod', workspace='erc3-prod', task_filter=['$TASKS'])
"
    ;;

  # Все задачи (последовательно)
  all)
    ./venv/bin/python3 -c "
from agent import ERC3Agent
agent = ERC3Agent()
agent.run_session(benchmark='erc3-prod', workspace='erc3-prod')
"
    ;;

  # Все задачи (параллельно)
  parallel)
    WORKERS=${2:-5}
    ./venv/bin/python3 -c "
from agent import ERC3Agent
agent = ERC3Agent()
agent.run_session_parallel(benchmark='erc3-prod', workspace='erc3-prod', max_workers=$WORKERS)
"
    ;;

  # Показать текущую версию конфига
  version)
    cat evolution/state.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Current: v{d[\"current_version\"]:03d}')"
    ;;

  # Показать упавшие задачи из последней сессии
  failed)
    LAST_SESSION=$(ls -d output/*/ 2>/dev/null | tail -1)
    if [ -z "$LAST_SESSION" ]; then
      echo "No sessions found"
      exit 1
    fi
    echo "Session: $LAST_SESSION"
    cd "$LAST_SESSION" && python3 -c "
import json, glob
failed = []
passed = 0
for f in sorted(glob.glob('t*.json')):
    try:
        data = json.load(open(f))
        ends = [x for x in data if x.get('type')=='task_end']
        if ends:
            if ends[-1]['score'] < 1.0:
                failed.append((f.replace('.json',''), ends[-1]['score']))
            else:
                passed += 1
    except: pass
print(f'Passed: {passed}, Failed: {len(failed)}')
for t, s in failed:
    print(f'  {t}: {s}')
"
    ;;

  # Показать лог задачи
  log)
    LAST_SESSION=$(ls -d output/*/ 2>/dev/null | tail -1)
    cat "${LAST_SESSION}$2.json" | python3 -m json.tool | less
    ;;

  # Помощь
  *)
    echo "Usage: ./run.sh <command> [args]"
    echo ""
    echo "Commands:"
    echo "  task <id>        Run single task (e.g., ./run.sh task t017)"
    echo "  tasks <ids>      Run multiple tasks (e.g., ./run.sh tasks t013,t014,t017)"
    echo "  all              Run all tasks sequentially"
    echo "  parallel [n]     Run all tasks in parallel with n workers (default: 5)"
    echo "  version          Show current evolution config version"
    echo "  failed           Show failed tasks from last session"
    echo "  log <id>         Show task log (e.g., ./run.sh log t017)"
    ;;
esac
