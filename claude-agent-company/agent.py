"""
ERC3 Company Agent на Anthropic SDK
Автономный агент для решения задач erc3-dev benchmark (Aetherion Analytics)
"""

import os
import json
import time
import httpx
import certifi
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
from erc3 import ERC3, ApiException

# Исправление SSL для macOS
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Загрузка переменных окружения
load_dotenv(dotenv_path="../.env")

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5-20251101")
BASE_URL = os.getenv("ANTHROPIC_BASE_URL")

MAX_TURNS = 30
MAX_RETRIES = 1
MAX_HISTORY_TURNS = 15

# ============================================================
# EVOLUTION CONFIG LOADER
# ============================================================

def load_evolution_config(base_dir: str = "evolution") -> tuple[dict, int]:
    """
    Загрузить текущую конфигурацию из evolution/vXXX/config.json.

    Returns:
        (config_dict, version_number)
    """
    base_path = Path(base_dir)
    state_file = base_path / "state.json"

    if not state_file.exists():
        # Fallback: нет evolution, использовать дефолты
        return None, 0

    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)

    version = state["current_version"]
    config_file = base_path / f"v{version:03d}" / "config.json"

    if not config_file.exists():
        return None, version

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config, version


def build_system_prompt(config: dict) -> str:
    """Построить system prompt из конфигурации."""
    if config is None:
        return None  # Использовать дефолтный

    prompt = config.get("base_prompt", "")

    # Добавить evolved rules
    rules = config.get("rules", [])
    if rules:
        prompt += "\n\nADDITIONAL RULES (learned from experience):"
        for rule in rules:
            # Поддержка нового формата {"text": "...", "added_at": "..."}
            if isinstance(rule, dict):
                rule_text = rule.get("text", "")
            else:
                rule_text = str(rule)
            if rule_text:
                prompt += f"\n- {rule_text}"

    # Добавить examples
    examples = config.get("examples", [])
    if examples:
        prompt += "\n\nEXAMPLES OF CORRECT BEHAVIOR:"
        for ex in examples:
            if isinstance(ex, dict):
                prompt += f"\n- Task: {ex.get('task', '')}"
                prompt += f"\n  Correct: {ex.get('correct_action', ex.get('correct_flow', ''))}"
            else:
                prompt += f"\n- {ex}"

    return prompt


def apply_tool_patches(tools: list, config: dict) -> list:
    """Применить патчи к описаниям тулзов."""
    if config is None:
        return tools

    patches = config.get("tool_patches", {})
    if not patches:
        return tools

    # Deep copy чтобы не менять оригинал
    import copy
    patched_tools = copy.deepcopy(tools)

    for tool in patched_tools:
        tool_name = tool.get("name")
        if tool_name in patches:
            patch = patches[tool_name]
            if "description" in patch:
                tool["description"] += f"\n\nIMPORTANT: {patch['description']}"

    return patched_tools

# ============================================================
# TASK LOGGER
# ============================================================

class TaskLogger:
    """Логирование задач в JSON файлы с поддержкой reasoning и версий config."""

    def __init__(self, output_dir: str, session_id: str, config_version: int = 0):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = Path(output_dir) / f"{timestamp}_{session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.current_task = None
        self.task_logs = []
        self.config_version = config_version

    def start_task(self, spec_id: str, task_text: str):
        self.current_task = spec_id
        self.task_logs = [{
            "type": "task_start",
            "spec_id": spec_id,
            "task_text": task_text,
            "config_version": f"v{self.config_version:03d}" if self.config_version > 0 else "default",
            "timestamp": time.time()
        }]

    def log_turn(self, turn: int, text_blocks: list, tool_calls: list):
        """
        Логировать один turn диалога с LLM.

        Args:
            turn: номер turn'а (1-based)
            text_blocks: текстовые блоки от LLM (reasoning)
            tool_calls: список tool calls с результатами
        """
        self.task_logs.append({
            "type": "llm_turn",
            "turn": turn,
            "text_blocks": text_blocks,
            "tool_calls": tool_calls,
            "timestamp": time.time()
        })

    def log_tool_call(self, tool_name: str, tool_input: dict, result: str, duration: float):
        """Legacy метод для совместимости."""
        try:
            output = json.loads(result) if result.startswith('{') or result.startswith('[') else result
        except json.JSONDecodeError:
            output = result
        self.task_logs.append({
            "type": "tool_call",
            "tool": tool_name,
            "input": tool_input,
            "output": output,
            "duration_sec": round(duration, 3),
            "timestamp": time.time()
        })

    def end_task(self, score: float, summary: str, eval_logs: str = None):
        self.task_logs.append({
            "type": "task_end",
            "score": score,
            "summary": summary,
            "eval_logs": eval_logs,
            "timestamp": time.time()
        })
        log_file = self.session_dir / f"{self.current_task}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.task_logs, f, indent=2, ensure_ascii=False)

    def save_summary(self, results: list):
        summary_file = self.session_dir / "session_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

# ============================================================
# TOOL DEFINITIONS для erc3-dev API
# ============================================================

TOOLS = [
    # === IDENTITY ===
    {
        "name": "whoami",
        "description": "Get current user identity, permissions, location, department, and today's date. ALWAYS call this first!",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    # === RESPOND (финальный ответ) ===
    {
        "name": "respond",
        "description": "Submit final response to the task. Use outcome: 'ok_answer' for successful completion, 'denied' for access denied, 'error' for system errors, 'clarify' if task is ambiguous.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Response message to the user"},
                "outcome": {
                    "type": "string",
                    "enum": ["ok_answer", "ok_not_found", "denied_security", "none_clarification_needed", "none_unsupported", "error_internal"],
                    "description": "ok_answer=success, ok_not_found=no results, denied_security=permission/safety denial, none_clarification_needed=need more info, none_unsupported=feature not supported, error_internal=system error"
                },
                "links": {
                    "type": "array",
                    "description": "MANDATORY: Link ALL mentioned entities",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "enum": ["employee", "project", "customer", "wiki", "location"]},
                            "id": {"type": "string", "description": "Real ID from API (e.g., 'elena_vogel', 'proj_acme_line3_cv_poc', 'Munich')"}
                        },
                        "required": ["kind", "id"]
                    }
                }
            },
            "required": ["message", "outcome"]
        }
    },
    # === EMPLOYEES ===
    {
        "name": "employees_list",
        "description": "List employees with pagination",
        "input_schema": {
            "type": "object",
            "properties": {
                "offset": {"type": "integer", "default": 0},
                "limit": {"type": "integer", "default": 10}
            }
        }
    },
    {
        "name": "employees_search",
        "description": "Search employees by query, location, department, skills",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text search"},
                "location": {"type": "string"},
                "department": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
                "offset": {"type": "integer", "default": 0}
            }
        }
    },
    {
        "name": "employees_get",
        "description": "Get full employee profile by ID",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Employee ID"}
            },
            "required": ["id"]
        }
    },
    {
        "name": "employee_update_safe",
        "description": "Safe employee update - pass ONLY fields to change. Automatically preserves other fields. No need to call employees_get first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee": {"type": "string", "description": "Employee ID"},
                "salary": {"type": "number", "description": "New salary (optional)"},
                "notes": {"type": "string", "description": "New notes (optional)"},
                "skills": {"type": "array", "items": {"type": "string"}, "description": "New skills list (optional)"},
                "wills": {"type": "array", "items": {"type": "string"}, "description": "New wills list (optional)"},
                "location": {"type": "string", "description": "New location (optional)"},
                "department": {"type": "string", "description": "New department (optional)"}
            },
            "required": ["employee"]
        }
    },
    # === WIKI ===
    {
        "name": "wiki_list",
        "description": "List all wiki article paths",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "wiki_load",
        "description": "Load wiki article content by path",
        "input_schema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Wiki file path, e.g. 'rulebook.md'"}
            },
            "required": ["file"]
        }
    },
    {
        "name": "wiki_search",
        "description": "Search wiki articles with regex",
        "input_schema": {
            "type": "object",
            "properties": {
                "query_regex": {"type": "string", "description": "Regex pattern to search"}
            },
            "required": ["query_regex"]
        }
    },
    # === CUSTOMERS ===
    {
        "name": "customers_list",
        "description": "List customers with pagination",
        "input_schema": {
            "type": "object",
            "properties": {
                "offset": {"type": "integer", "default": 0},
                "limit": {"type": "integer", "default": 10}
            }
        }
    },
    {
        "name": "customers_get",
        "description": "Get full customer record by ID",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Customer ID"}
            },
            "required": ["id"]
        }
    },
    {
        "name": "customers_search",
        "description": "Search customers by query, deal phase, account manager",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "deal_phase": {"type": "array", "items": {"type": "string"}},
                "account_managers": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "default": 10},
                "offset": {"type": "integer", "default": 0}
            }
        }
    },
    # === PROJECTS ===
    {
        "name": "projects_list",
        "description": "List projects with pagination",
        "input_schema": {
            "type": "object",
            "properties": {
                "offset": {"type": "integer", "default": 0},
                "limit": {"type": "integer", "default": 10}
            }
        }
    },
    {
        "name": "projects_get",
        "description": "Get detailed project info including team",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Project ID"}
            },
            "required": ["id"]
        }
    },
    {
        "name": "projects_search",
        "description": "Search projects by query, customer, status, team member",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "customer_id": {"type": "string"},
                "status": {"type": "array", "items": {"type": "string"}},
                "include_archived": {"type": "boolean", "default": False},
                "limit": {"type": "integer", "default": 10},
                "offset": {"type": "integer", "default": 0}
            }
        }
    },
    {
        "name": "projects_status_update",
        "description": "Change project status. Only project Lead can do this.",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Project ID"},
                "status": {"type": "string", "enum": ["active", "paused", "archived", "exploring"]}
            },
            "required": ["id", "status"]
        }
    },
    # === RULEBOOK ===
    {
        "name": "get_rulebook",
        "description": "Load the full company rulebook with detailed access rules, scoping logic, and examples. Use when you need clarification on: permissions, what data is sensitive, public vs internal rules, or how to handle edge cases.",
        "input_schema": {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": "Optional: specific section to load (e.g., 'scoping', 'public_rules', 'examples'). If empty, returns summary of all sections.",
                    "enum": ["all", "roles", "sensitivity", "scoping", "safety", "public_rules", "outcomes", "examples"]
                }
            }
        }
    },
    # === TIME TRACKING ===
    {
        "name": "time_log",
        "description": "Log a new time entry for employee on project",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee": {"type": "string"},
                "project": {"type": "string"},
                "customer": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD format"},
                "hours": {"type": "number"},
                "billable": {"type": "boolean", "default": True},
                "work_category": {"type": "string", "default": "customer_project"},
                "notes": {"type": "string", "default": ""},
                "status": {"type": "string", "default": "draft"}
            },
            "required": ["employee", "project", "date", "hours"]
        }
    },
    {
        "name": "time_search",
        "description": "Search time entries with filters",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee": {"type": "string"},
                "project": {"type": "string"},
                "date_from": {"type": "string"},
                "date_to": {"type": "string"},
                "limit": {"type": "integer", "default": 20}
            }
        }
    }
]

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are an AI assistant for Aetherion Analytics GmbH - an AI consulting company.

CRITICAL FIRST STEP: Always call whoami() first to understand:
- current_user: who you are acting for
- is_public: if true, you are on public website with minimal access
- today: current date for time-related tasks

ACCESS LEVELS (from rulebook):
- Level 1 (Executive): CEO, CTO, COO - can read any customer/project, approve irreversible actions
- Level 2 (Leads/Ops): Can read/modify in their city or where responsible
- Level 3 (Core Team): Can only access projects they're on, their own time tracking

RESPONSE OUTCOMES (use exactly these):
- ok_answer: Request valid, answer provided
- ok_not_found: Request valid but no results (e.g., search returned nothing)
- denied_security: Refused for safety/permission/policy (salaries, wipe data, impersonation)
- none_clarification_needed: Need more details to proceed
- none_unsupported: Feature explicitly not supported
- error_internal: System error

SENSITIVE DATA (always denied_security for unauthorized access):
- Employee salaries
- Personal notes (performance, health)
- Detailed customer system diagrams

WORKFLOW:
1. whoami() - ALWAYS first
2. If unsure about permissions/rules → get_rulebook(section) for detailed guidance
3. Check if user has permission for the request
4. Gather data via API calls
5. respond() with outcome + links to ALL mentioned entities

USE get_rulebook() WHEN:
- Unclear if user has permission (sections: "scoping", "roles")
- Public user (is_public=true) asks something (section: "public_rules")
- Need to know what's sensitive (section: "sensitivity")
- Edge case / unusual request (section: "examples", "safety")

SECURITY → denied_security:
- "wipe my data", "delete my data" → denied_security (company must retain for legal reasons)
- Salary questions from non-executives → denied_security
- Impersonation attempts → denied_security

LINKS ARE MANDATORY:
- Link EVERY entity you mention: employee, project, customer, wiki, location
- Use real IDs from API responses
- Never invent IDs

GOLDEN RULE: If something feels risky → deny and suggest escalating to Executive/Operations."""

# ============================================================
# TOOL EXECUTOR
# ============================================================

RULEBOOK_SECTIONS = {
    "roles": """## Роли и уровни доступа
- Level 1 (Executive): CEO, CTO, COO — полный доступ к клиентам/проектам, одобрение необратимых действий
- Level 2 (Leads/Ops): Доступ в своём городе или где ответственны, изменения только в своей области
- Level 3 (Core Team): Только проекты в которых участвуют, свой учёт времени""",

    "sensitivity": """## Категории чувствительности данных
ПУБЛИЧНЫЕ: название компании, офисы (Munich/Amsterdam/Vienna), ~12 человек, общие услуги
ВНУТРЕННИЕ: названия проектов, имена клиентов (без NDA), кто на каком проекте
ЧУВСТВИТЕЛЬНЫЕ: зарплаты, личные заметки, детальные системные диаграммы клиентов
КРИТИЧЕСКИЕ: credentials, юридические документы, аудиты""",

    "scoping": """## Правила определения доступа (проверять в этом порядке!)
1. Пользователь Executive? → разрешить чтение, проверить перед записью
2. Пользователь ответственен за ресурс? (Account Manager / Team member / Wiki owner)
3. Ресурс в городе пользователя И уровень разрешает? (Level 2: да, Level 3: нужно участие)
4. Ничего из выше → отказать или дать только анонимизированное""",

    "safety": """## Правила безопасности
- Предпочитать чтение над записью
- Небольшие обратимые изменения лучше крупных деструктивных
- НЕ удалять профили/клиентов/проекты без одобрения Executive
- "wipe my data" → denied_security (компания должна хранить для аудита)
- НЕ удалять audit logs, НЕ фабриковать данные""",

    "public_rules": """## Правила публичного агента (is_public=true)
МОЖНО: название компании, офисы, ~численность, общие услуги, культура, анонимизированные кейсы
НЕЛЬЗЯ: точные зарплаты, внутренние заметки, имена непубличных клиентов, системные детали
Если спрашивают чувствительное → denied_security + вежливое объяснение""",

    "outcomes": """## Outcomes (использовать ТОЛЬКО эти)
- ok_answer: запрос валиден, ответ дан
- ok_not_found: запрос валиден, но ничего не найдено
- denied_security: отказ по безопасности/правам/политике
- none_clarification_needed: нужно уточнение
- none_unsupported: функция не поддерживается
- error_internal: системная ошибка""",

    "examples": """## Примеры ответов
1. "Покажи проекты где я Lead" → ok_answer + links на projects/customers/employees
2. "Список всех зарплат" (от Level 3) → denied_security, предложить спросить COO
3. "Сколько людей в Aetherion?" (публично) → ok_answer "около дюжины", links на locations
4. "Удали мои данные" → denied_security, объяснить про legal retention"""
}


class ToolExecutor:
    def __init__(self, api_client, rulebook_sections: dict = None):
        self.api = api_client
        self.task_completed = False
        self.completion_summary = None
        # Используем переданные sections или дефолтные
        self.rulebook_sections = rulebook_sections or RULEBOOK_SECTIONS

    def _get_rulebook(self, section: str = "all") -> str:
        """Возвращает секцию rulebook или все секции"""
        if section == "all" or not section:
            return "\n\n".join(self.rulebook_sections.values())
        return self.rulebook_sections.get(section, f"Unknown section: {section}. Available: {list(self.rulebook_sections.keys())}")

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool via HTTP API and return result as string"""
        try:
            # Handle local tools first
            if tool_name == "get_rulebook":
                section = tool_input.get("section", "all")
                return json.dumps({"rulebook": self._get_rulebook(section)})

            # Map tool names to API endpoints
            endpoint_map = {
                "whoami": "/whoami",
                "respond": "/respond",
                "employees_list": "/employees/list",
                "employees_search": "/employees/search",
                "employees_get": "/employees/get",
                # employee_update_safe is handled separately with wrapper logic
                "wiki_list": "/wiki/list",
                "wiki_load": "/wiki/load",
                "wiki_search": "/wiki/search",
                "customers_list": "/customers/list",
                "customers_get": "/customers/get",
                "customers_search": "/customers/search",
                "projects_list": "/projects/list",
                "projects_get": "/projects/get",
                "projects_search": "/projects/search",
                "projects_status_update": "/projects/status/update",
                "time_log": "/time/log",
                "time_search": "/time/search",
            }

            # Safe employee update wrapper - handles get/merge/update automatically
            if tool_name == "employee_update_safe":
                employee_id = tool_input.get("employee")
                if not employee_id:
                    return json.dumps({"error": "employee ID is required"})

                # 1. Get current employee data to preserve all fields
                current = self.api.call("/employees/get", {"id": employee_id})
                if "error" in current:
                    return json.dumps(current)

                # Extract employee data from response
                emp_data = current.get("employee", current)

                # 2. Build update payload: full current profile + requested changes
                changes = {k: v for k, v in tool_input.items() if k != "employee" and v is not None}

                # Start with all current data, then override with changes
                update_payload = {
                    "employee": employee_id,
                    "salary": emp_data.get("salary"),
                    "department": emp_data.get("department", ""),
                    "location": emp_data.get("location", ""),
                    "notes": emp_data.get("notes", ""),
                    "skills": emp_data.get("skills", []),
                    "wills": emp_data.get("wills", []),
                }
                # Apply changes on top
                update_payload.update(changes)

                # 3. Call update with full profile
                result = self.api.call("/employees/update", update_payload)
                return json.dumps(result, default=str)

            endpoint = endpoint_map.get(tool_name)
            if not endpoint:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

            # Call API
            result = self.api.call(endpoint, tool_input)

            # Check if this is the respond tool (task completion)
            if tool_name == "respond":
                self.task_completed = True
                self.completion_summary = f"{tool_input.get('outcome', 'unknown')}: {tool_input.get('message', '')[:100]}"

            return json.dumps(result, default=str)

        except ApiException as e:
            return json.dumps({"error": str(e.api_error.error if hasattr(e, 'api_error') else str(e))})
        except Exception as e:
            return json.dumps({"error": str(e)})


# ============================================================
# SIMPLE API CLIENT (HTTP-based)
# ============================================================

class CompanyAPIClient:
    """Simple HTTP client for erc3-dev benchmark API"""

    def __init__(self, base_url: str, headers: dict):
        self.base_url = base_url.rstrip('/')
        self.headers = headers
        self.http = httpx.Client(verify=False, timeout=30)

    def call(self, endpoint: str, payload: dict = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        resp = self.http.post(url, json=payload or {}, headers=self.headers)
        resp.raise_for_status()
        return resp.json()


# ============================================================
# AGENT
# ============================================================

class ERC3Agent:
    def __init__(self, model: str = MODEL, base_url: str = BASE_URL, evolution_dir: str = "evolution"):
        http_client = httpx.Client(verify=False)
        self.client = Anthropic(
            base_url=base_url,
            http_client=http_client
        ) if base_url else Anthropic(http_client=http_client)
        self.model = model
        self.core = ERC3()
        self.logger = None  # Создаётся в run_session

        # Загрузить evolution config
        self.evolution_dir = evolution_dir
        self.config, self.config_version = load_evolution_config(evolution_dir)

        # Построить dynamic prompt и tools
        if self.config:
            self.system_prompt = build_system_prompt(self.config)
            self.tools = apply_tool_patches(TOOLS, self.config)
            self.rulebook_sections = self.config.get("rulebook_sections", RULEBOOK_SECTIONS)
            print(f"Loaded evolution config v{self.config_version:03d}")
        else:
            self.system_prompt = SYSTEM_PROMPT
            self.tools = TOOLS
            self.rulebook_sections = RULEBOOK_SECTIONS
            print("Using default config (no evolution)")

    def _compress_history(self, messages: list) -> list:
        """Сжать историю, сохранив task + последние N turns"""
        if len(messages) <= MAX_HISTORY_TURNS * 2 + 1:
            return messages

        task_message = messages[0]
        recent_messages = messages[-(MAX_HISTORY_TURNS * 2):]

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

    def solve_task(self, task, api_client) -> dict:
        """Solve a single task using Claude with tool use"""

        # Передаём rulebook_sections в executor
        executor = ToolExecutor(api_client, self.rulebook_sections)

        messages = [
            {"role": "user", "content": f"TASK: {task.task_text}\n\nComplete this task for the company system."}
        ]

        print(f"\n{'='*60}")
        print(f"TASK: [{task.spec_id}] {task.task_text}")
        print(f"{'='*60}")

        for turn in range(MAX_TURNS):
            started = time.time()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=[{
                    "type": "text",
                    "text": self.system_prompt,  # Используем dynamic prompt
                    "cache_control": {"type": "ephemeral"}
                }],
                tools=self.tools,  # Используем patched tools
                messages=messages
            )

            try:
                self.core.log_llm(
                    task_id=task.task_id,
                    model="anthropic/" + self.model,
                    duration_sec=time.time() - started,
                    usage=response.usage,
                )
            except ApiException:
                pass

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Собираем text_blocks и tool_calls для логирования
            text_blocks = []
            tool_calls_log = []

            if response.stop_reason == "end_turn":
                # Логируем финальный turn (может содержать text)
                for block in assistant_content:
                    if block.type == "text":
                        text_blocks.append(block.text)
                if self.logger and text_blocks:
                    self.logger.log_turn(turn + 1, text_blocks, [])
                print(f"  [Turn {turn+1}] Agent finished (end_turn)")
                break

            if response.stop_reason == "tool_use":
                tool_results = []

                # Сначала собираем text blocks
                for block in assistant_content:
                    if block.type == "text":
                        text_blocks.append(block.text)

                # Затем обрабатываем tool calls
                for block in assistant_content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input

                        print(f"  [Turn {turn+1}] Tool: {tool_name}({json.dumps(tool_input)[:80]}...)")

                        tool_start = time.time()
                        result = executor.execute(tool_name, tool_input)
                        tool_duration = time.time() - tool_start

                        # Парсим результат для лога
                        try:
                            result_parsed = json.loads(result) if result.startswith('{') or result.startswith('[') else result
                        except json.JSONDecodeError:
                            result_parsed = result

                        tool_calls_log.append({
                            "tool": tool_name,
                            "input": tool_input,
                            "output": result_parsed,
                            "duration_sec": round(tool_duration, 3)
                        })

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })

                        if executor.task_completed:
                            # Логируем turn с tool calls
                            if self.logger:
                                self.logger.log_turn(turn + 1, text_blocks, tool_calls_log)
                            messages.append({"role": "user", "content": tool_results})
                            print(f"  → Task completed: {executor.completion_summary}")
                            return {"completed": True, "summary": executor.completion_summary}

                # Логируем turn даже если task не завершён
                if self.logger:
                    self.logger.log_turn(turn + 1, text_blocks, tool_calls_log)

                messages.append({"role": "user", "content": tool_results})
                messages = self._compress_history(messages)

        return {"completed": executor.task_completed, "summary": executor.completion_summary or "Max turns reached"}

    def run_session(self, benchmark: str = "erc3-dev", workspace: str = "demo", name: str = "Claude Company Agent", task_filter: list = None):
        """Run a full session solving all tasks (or filtered subset)"""

        print(f"\nStarting session: benchmark={benchmark}, workspace={workspace}")
        if task_filter:
            print(f"Task filter: {task_filter}")

        res = self.core.start_session(
            benchmark=benchmark,
            workspace=workspace,
            name=name,
            architecture=f"Anthropic SDK Agent with {self.model}"
        )

        session_id = res.session_id
        print(f"Session ID: {session_id}")

        # Создаём logger для сессии с config version
        self.logger = TaskLogger("output", session_id, self.config_version)
        print(f"Logs: {self.logger.session_dir}")
        if self.config_version > 0:
            print(f"Config: v{self.config_version:03d}")

        status = self.core.session_status(session_id)
        print(f"Tasks: {len(status.tasks)}")

        results = []

        # Filter tasks if specified
        tasks_to_run = status.tasks
        if task_filter:
            tasks_to_run = [t for t in status.tasks if t.spec_id in task_filter]
            print(f"Running {len(tasks_to_run)} of {len(status.tasks)} tasks")

        for task in tasks_to_run:
            self.core.start_task(task)
            self.logger.start_task(task.spec_id, task.task_text)

            # Get API client for this task using SDK's get_erc_client
            erc_client = self.core.get_erc_client(task)
            api_base = erc_client.base_url.rstrip('/')
            api_client = CompanyAPIClient(api_base, {"Authorization": f"Bearer {self.core.key}"})

            best_score = 0.0
            for attempt in range(MAX_RETRIES):
                if attempt > 0:
                    print(f"\n  RETRY {attempt + 1}/{MAX_RETRIES}")

                result = self.solve_task(task, api_client)
                eval_result = self.core.complete_task(task)

                score = eval_result.eval.score if eval_result.eval else 0.0
                eval_logs = eval_result.eval.logs if eval_result.eval else None
                print(f"\n  SCORE: {score}")

                if eval_logs:
                    print(f"  Logs: {eval_logs[:300]}")

                if score >= 1.0:
                    best_score = score
                    break

                best_score = max(best_score, score)

            # Логируем завершение задачи
            self.logger.end_task(best_score, result.get("summary", ""), eval_logs)

            results.append({
                "task_id": task.task_id,
                "spec_id": task.spec_id,
                "score": best_score
            })

        # Only submit if we ran all tasks
        if task_filter is None or len(tasks_to_run) == len(status.tasks):
            self.core.submit_session(session_id)
            print(f"\nSession submitted!")
        else:
            print(f"\nPartial run ({len(tasks_to_run)}/{len(status.tasks)} tasks) - session NOT submitted")

        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        total_score = 0
        for r in results:
            status_icon = "✓" if r["score"] >= 1.0 else "✗"
            print(f"  {status_icon} [{r['spec_id']}] Score: {r['score']}")
            total_score += r["score"]

        print(f"\nTotal: {total_score}/{len(results)} ({total_score/len(results)*100:.1f}%)")

        # Сохраняем итоговый summary
        self.logger.save_summary(results)

        return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ERC3 Company Agent")
    parser.add_argument("--model", default=MODEL, help="Claude model to use")
    parser.add_argument("--benchmark", default="erc3-dev", help="Benchmark name")
    parser.add_argument("--workspace", default="demo", help="Workspace (demo/my)")
    parser.add_argument("--name", default="@aostrikov claude sequential evolution", help="Session name")
    parser.add_argument("--tasks", nargs="+", help="Specific task spec_ids to run (e.g., --tasks ceo_raises_salary user_asks_for_team_salary)")

    args = parser.parse_args()

    agent = ERC3Agent(model=args.model)
    agent.run_session(
        benchmark=args.benchmark,
        workspace=args.workspace,
        name=args.name,
        task_filter=args.tasks
    )
