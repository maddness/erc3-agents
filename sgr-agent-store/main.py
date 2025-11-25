import textwrap
from openai import OpenAI
from store_agent import run_agent
from erc3 import ERC3

client = OpenAI()
core = ERC3()
MODEL_ID = "gpt-4o"

# Start session with metadata
res = core.start_session(
    benchmark="store",
    workspace="my",
    name=f"Simple SGR Agent ({MODEL_ID})",
    architecture="NextStep SGR Agent with OpenAI")

status = core.session_status(res.session_id)
print(f"Session has {len(status.tasks)} tasks")

for task in status.tasks:
    print("="*40)
    print(f"Starting Task: {task.task_id} ({task.spec_id}): {task.task_text}")
    # start the task
    core.start_task(task)
    try:
        run_agent(MODEL_ID, core, task)
    except Exception as e:
        print(e)
    result = core.complete_task(task)
    if result.eval:
        explain = textwrap.indent(result.eval.logs, "  ")
        print(f"\nSCORE: {result.eval.score}\n{explain}\n")

core.submit_session(res.session_id)











