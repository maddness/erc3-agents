import time
from typing import Annotated, List, Union, Literal
from annotated_types import MaxLen, MinLen
from pydantic import BaseModel, Field
from erc3 import store, ApiException, TaskInfo, ERC3
from openai import OpenAI

client = OpenAI()

class ReportTaskCompletion(BaseModel):
    tool: Literal["report_completion"]
    completed_steps_laconic: List[str]
    code: Literal["completed", "failed"]

class NextStep(BaseModel):
    current_state: str
    # we'll use only the first step, discarding all the rest.
    plan_remaining_steps_brief: Annotated[List[str], MinLen(1), MaxLen(5)] =  Field(..., description="explain your thoughts on how to accomplish - what steps to execute")
    # now let's continue the cascade and check with LLM if the task is done
    task_completed: bool
    # Routing to one of the tools to execute the first remaining step
    # if task is completed, model will pick ReportTaskCompletion
    function: Union[
        ReportTaskCompletion,
        store.Req_ListProducts,
        store.Req_ViewBasket,
        store.Req_ApplyCoupon,
        store.Req_RemoveCoupon,
        store.Req_AddProductToBasket,
        store.Req_RemoveItemFromBasket,
        store.Req_CheckoutBasket,
    ] = Field(..., description="execute first remaining step")

system_prompt = """
You are a business assistant helping customers of OnlineStore.

- Clearly report when tasks are done.
- If ListProducts returns non-zero "NextOffset", it means there are more products available.
- You can apply coupon codes to get discounts. Use ViewBasket to see current discount and total.
- Only one coupon can be applied at a time. Apply a new coupon to replace the current one, or remove it explicitly.
"""

CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_CLR = "\x1B[0m"

def run_agent(model: str, api: ERC3, task: TaskInfo):

    store_api = api.get_store_client(task)

    # log will contain conversation context for the agent within task
    log = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task.task_text},
    ]

    # let's limit number of reasoning steps by 20, just to be safe
    for i in range(30):
        step = f"step_{i + 1}"
        print(f"Next {step}... ", end="")

        started = time.time()

        completion = client.beta.chat.completions.parse(
            model=model,
            response_format=NextStep,
            messages=log,
            max_completion_tokens=16384,
        )

        api.log_llm(
            task_id=task.task_id,
            model=model, # must match slug from OpenRouter
            duration_sec=time.time() - started,
            usage=completion.usage,
        )

        job = completion.choices[0].message.parsed

        # if SGR wants to finish, then quit loop
        if isinstance(job.function, ReportTaskCompletion):
            print(f"[blue]agent {job.function.code}[/blue]. Summary:")
            for s in job.function.completed_steps_laconic:
                print(f"- {s}")
            break

        # print next sep for debugging
        print(job.plan_remaining_steps_brief[0], f"\n  {job.function}")

        # Let's add tool request to conversation history as if OpenAI asked for it.
        # a shorter way would be to just append `job.model_dump_json()` entirely
        log.append({
            "role": "assistant",
            "content": job.plan_remaining_steps_brief[0],
            "tool_calls": [{
                "type": "function",
                "id": step,
                "function": {
                    "name": job.function.__class__.__name__,
                    "arguments": job.function.model_dump_json(),
                }}]
        })

        # now execute the tool by dispatching command to our handler
        try:
            result = store_api.dispatch(job.function)
            txt = result.model_dump_json(exclude_none=True, exclude_unset=True)
            print(f"{CLI_GREEN}OUT{CLI_CLR}: {txt}")
        except ApiException as e:
            txt = e.detail
            # print to console as ascii red
            print(f"{CLI_RED}ERR: {e.api_error.error}{CLI_CLR}")

        # and now we add results back to the convesation history, so that agent
        # we'll be able to act on the results in the next reasoning step.
        log.append({"role": "tool", "content": txt, "tool_call_id": step})
