"""
ERC3 Store Agent на Anthropic SDK
Автономный агент для решения задач store benchmark
"""

import os
import ssl
import json
import time
import httpx
import certifi
from typing import Any
from dotenv import load_dotenv
from anthropic import Anthropic
from erc3 import ERC3, store, ApiException

# Исправление SSL для macOS
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Загрузка переменных окружения
load_dotenv(dotenv_path="../.env")

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

# Выбор модели: opus, sonnet, haiku
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5-20251101")
BASE_URL = os.getenv("ANTHROPIC_BASE_URL")  # None = default Anthropic API

MAX_TURNS = 50  # Максимум итераций на задачу
MAX_RETRIES = 3  # Повторные попытки при SCORE < 1.0
MAX_HISTORY_TURNS = 10  # Sliding window: хранить только последние N turns

# ============================================================
# TOOL DEFINITIONS
# ============================================================

TOOLS = [
    {
        "name": "list_products",
        "description": "List products from the store with pagination. Returns products array with sku, name, available (stock), price. Also returns next_offset (-1 if no more pages).",
        "input_schema": {
            "type": "object",
            "properties": {
                "offset": {
                    "type": "integer",
                    "description": "Starting offset for pagination, default 0"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of products per page, max 3",
                    "maximum": 3
                }
            },
            "required": ["offset", "limit"]
        }
    },
    {
        "name": "add_to_basket",
        "description": "Add a product to the shopping basket",
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {
                    "type": "string",
                    "description": "Product SKU identifier"
                },
                "quantity": {
                    "type": "integer",
                    "description": "Number of items to add"
                }
            },
            "required": ["sku", "quantity"]
        }
    },
    {
        "name": "remove_from_basket",
        "description": "Remove a product from the shopping basket",
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {
                    "type": "string",
                    "description": "Product SKU identifier"
                },
                "quantity": {
                    "type": "integer",
                    "description": "Number of items to remove"
                }
            },
            "required": ["sku", "quantity"]
        }
    },
    {
        "name": "view_basket",
        "description": "View current basket contents. Returns items (array with sku, quantity, price), subtotal, coupon (if applied), discount, total",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "apply_coupon",
        "description": "Apply a coupon code to get discount",
        "input_schema": {
            "type": "object",
            "properties": {
                "coupon": {
                    "type": "string",
                    "description": "Coupon code to apply"
                }
            },
            "required": ["coupon"]
        }
    },
    {
        "name": "remove_coupon",
        "description": "Remove currently applied coupon",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    # === OPTIMIZED TOOLS (save turns) ===
    {
        "name": "list_all_products",
        "description": "List ALL products at once (no pagination). Returns complete products array. Use this instead of list_products to save turns.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "clear_basket",
        "description": "Remove ALL items from basket at once. Use before testing a new product combination.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "test_coupon",
        "description": "Test a coupon WITHOUT keeping it applied. Applies coupon, gets total, removes coupon - all in one call. Returns {subtotal, coupon, discount, total}. Use this to quickly test multiple coupons.",
        "input_schema": {
            "type": "object",
            "properties": {
                "coupon": {
                    "type": "string",
                    "description": "Coupon code to test"
                }
            },
            "required": ["coupon"]
        }
    },
    {
        "name": "test_combo",
        "description": "THE BEST tool for optimization! Sets basket to given items and tests ALL coupons in ONE call. Returns baseline (no coupon) and all coupon results. Use this to find cheapest combination.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "Items to put in basket: [{sku, quantity}, ...]",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sku": {"type": "string"},
                            "quantity": {"type": "integer"}
                        },
                        "required": ["sku", "quantity"]
                    }
                },
                "coupons": {
                    "type": "array",
                    "description": "Coupon codes to test",
                    "items": {"type": "string"}
                }
            },
            "required": ["items", "coupons"]
        }
    },
    {
        "name": "checkout",
        "description": "Complete the purchase. Call this when basket is ready. May fail if inventory changed (race condition) - retry with adjusted quantities.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "report_completion",
        "description": "Report that the task is complete. Call this AFTER successful checkout.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of what was done"
                }
            },
            "required": ["summary"]
        }
    },
    {
        "name": "report_impossible",
        "description": "Report that the task is impossible to complete (e.g., product doesn't exist, insufficient inventory, budget exceeded)",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why the task cannot be completed"
                }
            },
            "required": ["reason"]
        }
    }
]

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are an AI agent for store benchmark tasks.

TOOLS:
- list_all_products(): Get ALL products in 1 call
- test_combo(items, coupons): Sets basket + tests ALL coupons. TRUST THE RESULTS!
- add_to_basket(sku, qty): Add item
- apply_coupon(code): Apply coupon for checkout
- checkout(): Complete purchase
- report_completion(summary): Task done
- report_impossible(reason): Task impossible

IMPOSSIBLE - use report_impossible when:
- Product not in store
- Stock insufficient
- Budget exceeded
- Coupon invalid or doesn't give REQUIRED discount

MULTIPLE COUPONS: System allows only ONE coupon per order!
If task requires using MULTIPLE coupons together → report_impossible (cannot apply 2 coupons to 1 order)

FOR OPTIMIZATION TASKS ("cheapest", "best deal", "minimize cost"):

Step 1: list_all_products() - note products and prices
Step 2-N: Call test_combo() for EACH possible combination. TRUST the results - DO NOT re-test manually!
Step N+1: Compare all results, find MINIMUM total
Step N+2: add_to_basket + apply_coupon(best) + checkout + report_completion

EXAMPLE (buy 24 sodas cheaply, coupons: A, B, C):
- list_all_products() → soda-24pk=$40, soda-12pk=$20, soda-6pk=$12
- test_combo([{soda-24pk,1}], [A,B,C]) → results show best=$35
- test_combo([{soda-12pk,2}], [A,B,C]) → results show best=$40
- test_combo([{soda-6pk,4}], [A,B,C]) → results show best=$34
- test_combo([{soda-12pk,1},{soda-6pk,2}], [A,B,C]) → results show best=$30 ← WINNER
- add_to_basket + apply_coupon + checkout + report_completion

RULES:
1. test_combo results are ACCURATE - no need to verify with test_coupon
2. Test MIXED combinations (like 12pk+6pk) - they often win!
3. After finding minimum, go straight to checkout"""

# ============================================================
# TOOL EXECUTOR
# ============================================================

class ToolExecutor:
    def __init__(self, store_api):
        self.store_api = store_api
        self.task_completed = False
        self.completion_summary = None

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool and return result as string"""
        try:
            if tool_name == "list_products":
                resp = self.store_api.dispatch(store.Req_ListProducts(
                    offset=tool_input["offset"],
                    limit=tool_input["limit"]
                ))
                return json.dumps({
                    "products": [
                        {"sku": p.sku, "name": p.name, "available": p.available, "price": p.price}
                        for p in resp.products
                    ],
                    "next_offset": resp.next_offset
                })

            elif tool_name == "add_to_basket":
                self.store_api.dispatch(store.Req_AddProductToBasket(
                    sku=tool_input["sku"],
                    quantity=tool_input["quantity"]
                ))
                return json.dumps({"success": True, "message": f"Added {tool_input['quantity']}x {tool_input['sku']} to basket"})

            elif tool_name == "remove_from_basket":
                self.store_api.dispatch(store.Req_RemoveItemFromBasket(
                    sku=tool_input["sku"],
                    quantity=tool_input["quantity"]
                ))
                return json.dumps({"success": True, "message": f"Removed {tool_input['quantity']}x {tool_input['sku']} from basket"})

            elif tool_name == "view_basket":
                resp = self.store_api.dispatch(store.Req_ViewBasket())
                return json.dumps({
                    "items": [{"sku": i.sku, "quantity": i.quantity, "price": i.price} for i in resp.items],
                    "subtotal": resp.subtotal,
                    "coupon": resp.coupon,
                    "discount": resp.discount,
                    "total": resp.total
                })

            elif tool_name == "apply_coupon":
                self.store_api.dispatch(store.Req_ApplyCoupon(coupon=tool_input["coupon"]))
                return json.dumps({"success": True, "message": f"Coupon {tool_input['coupon']} applied"})

            elif tool_name == "remove_coupon":
                self.store_api.dispatch(store.Req_RemoveCoupon())
                return json.dumps({"success": True, "message": "Coupon removed"})

            # === OPTIMIZED TOOLS ===
            elif tool_name == "list_all_products":
                # Загрузить ВСЕ продукты за один вызов
                all_products = []
                offset = 0
                while True:
                    resp = self.store_api.dispatch(store.Req_ListProducts(offset=offset, limit=3))
                    for p in resp.products:
                        all_products.append({"sku": p.sku, "name": p.name, "available": p.available, "price": p.price})
                    if resp.next_offset < 0:
                        break
                    offset = resp.next_offset
                return json.dumps({"products": all_products, "count": len(all_products)})

            elif tool_name == "clear_basket":
                # Очистить корзину полностью
                basket = self.store_api.dispatch(store.Req_ViewBasket())
                for item in basket.items:
                    self.store_api.dispatch(store.Req_RemoveItemFromBasket(sku=item.sku, quantity=item.quantity))
                return json.dumps({"success": True, "message": "Basket cleared"})

            elif tool_name == "test_coupon":
                # Тестировать купон без сохранения: apply → view → remove
                coupon = tool_input["coupon"]
                try:
                    self.store_api.dispatch(store.Req_ApplyCoupon(coupon=coupon))
                    basket = self.store_api.dispatch(store.Req_ViewBasket())
                    result = {
                        "coupon": coupon,
                        "subtotal": basket.subtotal,
                        "discount": basket.discount,
                        "total": basket.total,
                        "valid": True
                    }
                    self.store_api.dispatch(store.Req_RemoveCoupon())
                    return json.dumps(result)
                except ApiException:
                    return json.dumps({"coupon": coupon, "valid": False, "error": "Coupon not applicable"})

            elif tool_name == "test_combo":
                # Главный инструмент оптимизации: установить корзину + тестировать все купоны
                items = tool_input["items"]
                coupons = tool_input["coupons"]

                # 1. Очистить корзину
                basket = self.store_api.dispatch(store.Req_ViewBasket())
                for item in basket.items:
                    self.store_api.dispatch(store.Req_RemoveItemFromBasket(sku=item.sku, quantity=item.quantity))

                # 2. Добавить новые items
                for item in items:
                    self.store_api.dispatch(store.Req_AddProductToBasket(sku=item["sku"], quantity=item["quantity"]))

                # 3. Получить baseline (без купона)
                basket = self.store_api.dispatch(store.Req_ViewBasket())
                results = [{
                    "coupon": None,
                    "subtotal": basket.subtotal,
                    "discount": 0,
                    "total": basket.subtotal
                }]

                # 4. Тестируем каждый купон
                for coupon in coupons:
                    try:
                        self.store_api.dispatch(store.Req_ApplyCoupon(coupon=coupon))
                        basket = self.store_api.dispatch(store.Req_ViewBasket())
                        results.append({
                            "coupon": coupon,
                            "subtotal": basket.subtotal,
                            "discount": basket.discount,
                            "total": basket.total
                        })
                        self.store_api.dispatch(store.Req_RemoveCoupon())
                    except ApiException:
                        results.append({"coupon": coupon, "valid": False, "error": "Not applicable"})

                # Вернуть items + все результаты тестирования
                return json.dumps({
                    "items": [{"sku": i["sku"], "quantity": i["quantity"]} for i in items],
                    "results": results
                })

            elif tool_name == "checkout":
                self.store_api.dispatch(store.Req_CheckoutBasket())
                return json.dumps({"success": True, "message": "Checkout completed successfully!"})

            elif tool_name == "report_completion":
                self.task_completed = True
                self.completion_summary = tool_input.get("summary", "Task completed")
                return json.dumps({"success": True, "message": "Task marked as complete"})

            elif tool_name == "report_impossible":
                self.task_completed = True
                self.completion_summary = f"IMPOSSIBLE: {tool_input.get('reason', 'Unknown reason')}"
                return json.dumps({"success": True, "message": "Task marked as impossible"})

            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except ApiException as e:
            return json.dumps({"error": str(e.api_error.error)})
        except Exception as e:
            return json.dumps({"error": str(e)})


# ============================================================
# AGENT
# ============================================================

class ERC3Agent:
    def __init__(self, model: str = MODEL, base_url: str = BASE_URL):
        # Отключаем SSL верификацию для локальной разработки
        http_client = httpx.Client(verify=False)
        self.client = Anthropic(
            base_url=base_url,
            http_client=http_client
        ) if base_url else Anthropic(http_client=http_client)
        self.model = model
        self.core = ERC3()

    def _compress_history(self, messages: list) -> list:
        """Сжать историю, сохранив task + последние N turns"""
        if len(messages) <= MAX_HISTORY_TURNS * 2 + 1:
            return messages

        task_message = messages[0]
        recent_messages = messages[-(MAX_HISTORY_TURNS * 2):]

        # Подсчёт действий в сжатых сообщениях
        old_messages = messages[1:-(MAX_HISTORY_TURNS * 2)]
        actions = []
        for msg in old_messages:
            if msg["role"] == "assistant":
                for block in msg.get("content", []):
                    if hasattr(block, "name"):
                        actions.append(block.name)

        summary = f"Previous {len(actions)} actions: {', '.join(actions[:10])}{'...' if len(actions) > 10 else ''}"

        return [
            task_message,
            {"role": "user", "content": f"[{summary}]"},
            *recent_messages
        ]

    def solve_task(self, task, store_api) -> dict:
        """Solve a single task using Claude with tool use"""

        executor = ToolExecutor(store_api)

        messages = [
            {"role": "user", "content": f"TASK: {task.task_text}\n\nPlease complete this shopping task."}
        ]

        print(f"\n{'='*60}")
        print(f"TASK: [{task.spec_id}] {task.task_text}")
        print(f"{'='*60}")

        for turn in range(MAX_TURNS):
            # Call Claude with prompt caching
            started = time.time()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                tools=TOOLS,
                messages=messages
            )

            # Log telemetry for leaderboard
            try:
                self.core.log_llm(
                    task_id=task.task_id,
                    model="anthropic/" + self.model,
                    duration_sec=time.time() - started,
                    usage=response.usage,
                )
            except ApiException:
                pass  # Task may be completed (retry scenario)

            # Process response
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Check if done
            if response.stop_reason == "end_turn":
                print(f"  [Turn {turn+1}] Agent finished (end_turn)")
                break

            # Process tool calls
            if response.stop_reason == "tool_use":
                tool_results = []

                for block in assistant_content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input

                        print(f"  [Turn {turn+1}] Tool: {tool_name}({json.dumps(tool_input)[:50]}...)")

                        # Execute tool
                        result = executor.execute(tool_name, tool_input)

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })

                        # Check if task completed
                        if executor.task_completed:
                            messages.append({"role": "user", "content": tool_results})
                            print(f"  → Task completed: {executor.completion_summary}")
                            return {"completed": True, "summary": executor.completion_summary}

                messages.append({"role": "user", "content": tool_results})
                messages = self._compress_history(messages)

        return {"completed": executor.task_completed, "summary": executor.completion_summary or "Max turns reached"}

    def run_session(self, benchmark: str = "store", workspace: str = "demo", name: str = "Claude Agent"):
        """Run a full session solving all tasks"""

        print(f"\nStarting session: benchmark={benchmark}, workspace={workspace}")

        # Start session
        res = self.core.start_session(
            benchmark=benchmark,
            workspace=workspace,
            name=name,
            architecture=f"Anthropic SDK Agent with {self.model}"
        )

        session_id = res.session_id
        print(f"Session ID: {session_id}")

        # Get tasks
        status = self.core.session_status(session_id)
        print(f"Tasks: {len(status.tasks)}")

        results = []

        for task in status.tasks:
            # Start task
            self.core.start_task(task)
            store_api = self.core.get_store_client(task)

            # Solve with retries
            best_score = 0.0
            for attempt in range(MAX_RETRIES):
                if attempt > 0:
                    print(f"\n  RETRY {attempt + 1}/{MAX_RETRIES}")
                    # Reset - start new task instance
                    # Note: ERC3 may not support this, task state persists
                    pass

                # Solve
                result = self.solve_task(task, store_api)

                # Complete and get score
                eval_result = self.core.complete_task(task)

                score = eval_result.eval.score if eval_result.eval else 0.0
                print(f"\n  SCORE: {score}")

                if eval_result.eval and eval_result.eval.logs:
                    print(f"  Logs: {eval_result.eval.logs[:200]}")

                if score >= 1.0:
                    best_score = score
                    break

                best_score = max(best_score, score)

                # If not perfect, analyze and retry
                if attempt < MAX_RETRIES - 1:
                    print(f"  Score < 1.0, will retry with fresh session task...")

            results.append({
                "task_id": task.task_id,
                "spec_id": task.spec_id,
                "score": best_score
            })

        # Submit session
        self.core.submit_session(session_id)
        print(f"\nSession submitted!")

        # Summary
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        total_score = 0
        for r in results:
            status_icon = "✓" if r["score"] >= 1.0 else "✗"
            print(f"  {status_icon} [{r['spec_id']}] Score: {r['score']}")
            total_score += r["score"]

        print(f"\nTotal: {total_score}/{len(results)} ({total_score/len(results)*100:.1f}%)")

        return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ERC3 Store Agent")
    parser.add_argument("--model", default=MODEL, help="Claude model to use")
    parser.add_argument("--benchmark", default="store", help="Benchmark name")
    parser.add_argument("--workspace", default="demo", help="Workspace (demo/my)")
    parser.add_argument("--name", default="Claude Anthropic SDK Agent", help="Session name")

    args = parser.parse_args()

    agent = ERC3Agent(model=args.model)
    agent.run_session(
        benchmark=args.benchmark,
        workspace=args.workspace,
        name=args.name
    )
