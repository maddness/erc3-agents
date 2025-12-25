"""
ERC3 Company Agent на Anthropic SDK
Автономный агент для решения задач erc3-dev benchmark (Aetherion Analytics)
"""

import os
import json
import time
import httpx
import certifi
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic, RateLimitError
from erc3 import ERC3, ApiException
from evolution.versioner import Versioner

# Исправление SSL для macOS
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Загрузка переменных окружения
load_dotenv(dotenv_path="../.env")

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

MODEL = os.getenv("ANTHROPIC_MODEL")
BASE_URL = os.getenv("ANTHROPIC_BASE_URL")

MAX_TURNS = 100
MAX_RETRIES = 1
MAX_HISTORY_TURNS = 15

# Thread synchronization
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print"""
    with print_lock:
        print(*args, **kwargs)

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
        self.config_version = config_version
        # Thread-safe: per-task logs storage
        self._task_logs = {}  # spec_id -> list of log entries
        self._lock = threading.Lock()

    def start_task(self, spec_id: str, task_text: str):
        with self._lock:
            self._task_logs[spec_id] = [{
                "type": "task_start",
                "spec_id": spec_id,
                "task_text": task_text,
                "config_version": f"v{self.config_version:03d}" if self.config_version > 0 else "default",
                "timestamp": time.time()
            }]

    def log_turn(self, spec_id: str, turn: int, text_blocks: list, tool_calls: list):
        """
        Логировать один turn диалога с LLM.

        Args:
            spec_id: идентификатор задачи
            turn: номер turn'а (1-based)
            text_blocks: текстовые блоки от LLM (reasoning)
            tool_calls: список tool calls с результатами
        """
        with self._lock:
            if spec_id in self._task_logs:
                self._task_logs[spec_id].append({
                    "type": "llm_turn",
                    "turn": turn,
                    "text_blocks": text_blocks,
                    "tool_calls": tool_calls,
                    "timestamp": time.time()
                })

    def log_tool_call(self, spec_id: str, tool_name: str, tool_input: dict, result: str, duration: float):
        """Legacy метод для совместимости."""
        try:
            output = json.loads(result) if result.startswith('{') or result.startswith('[') else result
        except json.JSONDecodeError:
            output = result
        with self._lock:
            if spec_id in self._task_logs:
                self._task_logs[spec_id].append({
                    "type": "tool_call",
                    "tool": tool_name,
                    "input": tool_input,
                    "output": output,
                    "duration_sec": round(duration, 3),
                    "timestamp": time.time()
                })

    def end_task(self, spec_id: str, score: float, summary: str, eval_logs: str = None):
        with self._lock:
            if spec_id not in self._task_logs:
                return
            self._task_logs[spec_id].append({
                "type": "task_end",
                "score": score,
                "summary": summary,
                "eval_logs": eval_logs,
                "timestamp": time.time()
            })
            log_file = self.session_dir / f"{spec_id}.json"
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(self._task_logs[spec_id], f, indent=2, ensure_ascii=False)
            # Clean up to free memory
            del self._task_logs[spec_id]

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
        "description": "List employees with pagination. Returns basic data: id, name, email, salary, location, department. Does NOT include skills/wills - use find_employees_by_skill for skill/will queries, or employees_get for full profile.",
        "input_schema": {
            "type": "object",
            "properties": {
                "offset": {"type": "integer", "default": 0},
                "limit": {"type": "integer", "default": 5, "description": "API max is 5"}
            }
        }
    },
    {
        "name": "employees_search",
        "description": "Search employees by name, location, department, or skills/wills. Returns basic data (id, name, email, location, department). Use skills/wills filters to find employees with specific capabilities.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text search by name"},
                "location": {"type": "string"},
                "department": {"type": "string"},
                "skills": {
                    "type": "array",
                    "description": "Filter by skills. Each item: {name: 'skill_name', min_level: 1, max_level: 10}",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Skill name, e.g. 'skill_rail'"},
                            "min_level": {"type": "integer", "default": 1, "description": "Minimum level (1-10)"},
                            "max_level": {"type": "integer", "default": 0, "description": "Maximum level (0=no limit)"}
                        },
                        "required": ["name"]
                    }
                },
                "wills": {
                    "type": "array",
                    "description": "Filter by wills. Each item: {name: 'will_name', min_level: 1, max_level: 10}",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Will name, e.g. 'will_travel'"},
                            "min_level": {"type": "integer", "default": 1, "description": "Minimum level (1-10)"},
                            "max_level": {"type": "integer", "default": 0, "description": "Maximum level (0=no limit)"}
                        },
                        "required": ["name"]
                    }
                },
                "limit": {"type": "integer", "default": 5, "description": "API max is 5"},
                "offset": {"type": "integer", "default": 0}
            }
        }
    },
    {
        "name": "employees_get",
        "description": "Get FULL employee profile by ID including skills and wills with levels. Use for single employee lookup. For finding employees by skill/will level, use find_employees_by_skill instead.",
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
    # === COMPOSITE TOOLS ===
    {
        "name": "find_employees_by_skill",
        "description": "Find employees by skill(s) and/or will(s). Supports: 1) single skill/will, 2) skill AND will, 3) ARRAY of skills (for coaching queries). Returns employees with full profiles.",
        "input_schema": {
            "type": "object",
            "properties": {
                "skill": {"type": "string", "description": "Single skill name (e.g., 'skill_technical_coatings')"},
                "will": {"type": "string", "description": "Single will name (e.g., 'will_travel')"},
                "skills": {
                    "type": "array",
                    "description": "Array of skills to search. Use for coaching: [{name: 'skill_x', min_level: 7}, ...]. Returns employees matching ANY skill.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "min_level": {"type": "integer"}
                        },
                        "required": ["name", "min_level"]
                    }
                },
                "min_skill_level": {"type": "integer", "default": 1, "description": "Min level for single skill (find >= this)"},
                "max_skill_level": {"type": "integer", "description": "Max level for single skill (find <= this). Use for 'least skilled' queries."},
                "min_will_level": {"type": "integer", "default": 1, "description": "Min level for single will (find >= this)"},
                "max_will_level": {"type": "integer", "description": "Max level for single will (find <= this). Use for 'least interested' queries."},
                "sort_order": {"type": "string", "enum": ["desc", "asc"], "default": "desc", "description": "Sort by level: 'desc' (highest first, default) or 'asc' (lowest first for 'least skilled')"},
                "top_n": {"type": "integer", "default": 10, "description": "Max employees to return"},
                "exclude_employee_id": {"type": "string", "description": "Exclude this employee (for coaching - exclude the person being coached)"},
                "location": {"type": "string", "description": "Optional: filter by location"},
                "department": {"type": "string", "description": "Optional: filter by department"}
            }
        }
    },
    {
        "name": "find_projects_by_employee",
        "description": "Find all projects where a specific employee is involved. Returns projects with full details including team composition. Use this instead of iterating through all projects manually! For 'In which projects is X involved' use include_archived=True. For 'current/active projects' use include_archived=False.",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string", "description": "Employee ID to search for"},
                "employee_name": {"type": "string", "description": "Employee name (will search for ID first)"},
                "role": {"type": "string", "description": "Optional: filter by role (e.g., 'Lead', 'Engineer', 'Designer')"},
                "status": {"type": "string", "description": "Optional: filter by project status (e.g., 'active', 'completed')"},
                "include_archived": {"type": "boolean", "default": False, "description": "Include archived projects. Use True for 'all projects involved' or 'which projects', False for 'current/active projects'"}
            }
        }
    },
    {
        "name": "find_projects_by_role",
        "description": "Find projects that have (or don't have) a specific role in the team. Use for questions like 'which projects have a designer' or 'projects missing QA role'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "role": {"type": "string", "description": "Role to search for (e.g., 'Designer', 'QA', 'Lead')"},
                "has_role": {"type": "boolean", "default": True, "description": "True = find projects WITH this role, False = find projects WITHOUT this role"},
                "status": {"type": "string", "description": "Optional: filter by project status"}
            },
            "required": ["role"]
        }
    },
    {
        "name": "find_project_leads",
        "description": "Find ALL project leads with their salaries. Use for salary comparisons, parity analysis. Returns unique leads with salary, projects they lead, and total FTE as lead.",
        "input_schema": {
            "type": "object",
            "properties": {
                "min_salary": {"type": "integer", "description": "Optional: filter leads with salary >= this value"},
                "max_salary": {"type": "integer", "description": "Optional: filter leads with salary <= this value"},
                "exclude_employee_id": {"type": "string", "description": "Optional: exclude this employee from results (for 'higher than X' queries)"}
            }
        }
    },
    {
        "name": "find_employees_by_location",
        "description": "Find ALL employees at a specific location. Use for workload analysis (busiest/least busy at location). Returns employees with full profiles including id, name, salary, department. Handles pagination automatically.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "Location name (e.g., 'HQ – Italy', 'Munich Office – Germany', 'Serbian Plant')"},
                "department": {"type": "string", "description": "Optional: filter by department"}
            },
            "required": ["location"]
        }
    },
    {
        "name": "calculate_workloads",
        "description": "Calculate FTE workload for a list of employees in ONE call. Use for 'busiest/least busy' queries. Fetches all projects internally and returns workload map. Much faster than iterating projects manually!",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of employee IDs to calculate workload for"
                }
            },
            "required": ["employee_ids"]
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
    {
        "name": "wiki_update",
        "description": "Update wiki article content. To delete an article, set content to empty string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Wiki file path, e.g. 'marketing.md'"},
                "content": {"type": "string", "description": "New content for the wiki article"}
            },
            "required": ["file", "content"]
        }
    },
    {
        "name": "wiki_rename",
        "description": "Rename/move a wiki article. Copies content to new path and deletes original. Use this instead of manual load+update for renaming to preserve special characters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source wiki file path"},
                "destination": {"type": "string", "description": "Destination wiki file path"},
                "delete_source": {"type": "boolean", "description": "Delete source after copy (default: true)", "default": True}
            },
            "required": ["source", "destination"]
        }
    },
    {
        "name": "wiki_batch_update",
        "description": "Update multiple wiki files in ONE call. Use for bulk wiki operations like creating pages for all leads, all employees, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "updates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string", "description": "Wiki file path"},
                            "content": {"type": "string", "description": "Content to write"}
                        },
                        "required": ["file", "content"]
                    },
                    "description": "Array of {file, content} pairs to update"
                }
            },
            "required": ["updates"]
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
        "description": "Search projects by query, customer, status, or team member. Use team filter to find projects where specific employee is involved or has specific role.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "customer_id": {"type": "string"},
                "status": {"type": "array", "items": {"type": "string"}},
                "team": {
                    "type": "object",
                    "description": "Filter by team composition",
                    "properties": {
                        "employee_id": {"type": "string", "description": "Find projects with this employee"},
                        "role": {"type": "string", "description": "Find projects with this role (Lead, Engineer, Designer, QA, Ops, Other)"},
                        "min_time_slice": {"type": "number", "default": 0.0, "description": "Minimum FTE allocation"}
                    }
                },
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
    {
        "name": "projects_team_update",
        "description": "REPLACE project team. This is a REPLACE operation (not ADD). First get current team via projects_get, modify array, then send complete team. Only project Lead can modify.",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Project ID"},
                "team": {
                    "type": "array",
                    "description": "Complete team array (REPLACES existing team)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "employee": {"type": "string"},
                            "time_slice": {"type": "number"},
                            "role": {"type": "string"}
                        },
                        "required": ["employee", "time_slice", "role"]
                    }
                }
            },
            "required": ["id", "team"]
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
    },
    {
        "name": "time_update",
        "description": "Update a time entry. Use to void entries by setting status='voided'. Requires all fields from original entry.",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Time entry ID (e.g., 'te_050')"},
                "date": {"type": "string", "description": "Date YYYY-MM-DD"},
                "hours": {"type": "number"},
                "work_category": {"type": "string", "default": "customer_project"},
                "notes": {"type": "string"},
                "billable": {"type": "boolean"},
                "status": {"type": "string", "description": "draft, submitted, approved, invoiced, or 'voided' to cancel entry"}
            },
            "required": ["id", "status"]
        }
    },
    {
        "name": "time_get",
        "description": "Get a single time entry by ID",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Time entry ID"}
            },
            "required": ["id"]
        }
    },
    {
        "name": "time_summary_by_project",
        "description": "Get time summary aggregated by project. Use for 'how many hours logged on project X between dates'",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": "Start date YYYY-MM-DD"},
                "date_to": {"type": "string", "description": "End date YYYY-MM-DD"},
                "customers": {"type": "array", "items": {"type": "string"}, "description": "Filter by customer IDs"},
                "projects": {"type": "array", "items": {"type": "string"}, "description": "Filter by project IDs"},
                "employees": {"type": "array", "items": {"type": "string"}, "description": "Filter by employee IDs"},
                "billable": {"type": "string", "description": "Filter: 'billable', 'non_billable', or '' for all"}
            },
            "required": ["date_from", "date_to"]
        }
    },
    {
        "name": "time_summary_by_employee",
        "description": "Get time summary aggregated by employee. Use for 'how many hours did employee X log between dates'",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": "Start date YYYY-MM-DD"},
                "date_to": {"type": "string", "description": "End date YYYY-MM-DD"},
                "customers": {"type": "array", "items": {"type": "string"}, "description": "Filter by customer IDs"},
                "projects": {"type": "array", "items": {"type": "string"}, "description": "Filter by project IDs"},
                "employees": {"type": "array", "items": {"type": "string"}, "description": "Filter by employee IDs"},
                "billable": {"type": "string", "description": "Filter: 'billable', 'non_billable', or '' for all"}
            },
            "required": ["date_from", "date_to"]
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
                "wiki_update": "/wiki/update",
                "customers_list": "/customers/list",
                "customers_get": "/customers/get",
                "customers_search": "/customers/search",
                "projects_list": "/projects/list",
                "projects_get": "/projects/get",
                "projects_search": "/projects/search",
                "projects_status_update": "/projects/status/update",
                "projects_team_update": "/projects/team/update",
                "time_log": "/time/log",
                "time_search": "/time/search",
                "time_update": "/time/update",
                "time_get": "/time/get",
                "time_summary_by_project": "/time/summary/by-project",
                "time_summary_by_employee": "/time/summary/by-employee",
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

            # Composite tool: find employees by skill/will (uses SDK native filters)
            if tool_name == "find_employees_by_skill":
                skill_name = tool_input.get("skill")
                will_name = tool_input.get("will")
                skills_array = tool_input.get("skills", [])  # Array mode for coaching
                min_skill_level = tool_input.get("min_skill_level", tool_input.get("min_level", 1))
                max_skill_level = tool_input.get("max_skill_level")  # For "least skilled" queries
                min_will_level = tool_input.get("min_will_level", tool_input.get("min_level", 1))
                max_will_level = tool_input.get("max_will_level")  # For "least interested" queries
                sort_order = tool_input.get("sort_order", "desc")  # "desc" (highest first) or "asc" (lowest first)
                top_n = tool_input.get("top_n", 10)
                exclude_id = tool_input.get("exclude_employee_id")
                filter_location = tool_input.get("location")
                filter_department = tool_input.get("department")

                def get_level(profile, attr_name, attr_list):
                    for item in profile.get(attr_list, []):
                        if item.get("name") == attr_name:
                            return item.get("level", 0)
                    return 0

                def fetch_profile(emp_id):
                    try:
                        result = self.api.call("/employees/get", {"id": emp_id})
                        return result.get("employee", {})
                    except:
                        return None

                # MODE 1: Array of skills (for coaching queries)
                if skills_array:
                    # Collect all candidate IDs across all skill searches
                    all_candidate_ids = set()
                    for skill_filter in skills_array:
                        s_name = skill_filter.get("name")
                        s_min = skill_filter.get("min_level", 1)
                        search_params = {"limit": 5, "offset": 0}
                        search_params["skills"] = [{"name": s_name, "min_level": s_min, "max_level": 0}]
                        if filter_location:
                            search_params["location"] = filter_location
                        if filter_department:
                            search_params["department"] = filter_department

                        offset = 0
                        while True:
                            search_params["offset"] = offset
                            page = self.api.call("/employees/search", search_params)
                            employees = page.get("employees", [])
                            if not employees:
                                break
                            for emp in employees:
                                all_candidate_ids.add(emp["id"])
                            if page.get("next_offset", -1) < 0:
                                break
                            offset = page["next_offset"]

                    # Exclude target employee
                    if exclude_id:
                        all_candidate_ids.discard(exclude_id)

                    # Fetch profiles in parallel
                    with ThreadPoolExecutor(max_workers=10) as executor:
                        profiles = list(executor.map(fetch_profile, list(all_candidate_ids)))
                    profiles = [p for p in profiles if p]

                    # Build results with coaching info
                    results = []
                    for profile in profiles:
                        can_coach = []
                        for skill_filter in skills_array:
                            s_name = skill_filter.get("name")
                            s_min = skill_filter.get("min_level", 1)
                            level = get_level(profile, s_name, "skills")
                            if level >= s_min:
                                can_coach.append({"skill": s_name, "level": level, "required": s_min})

                        if can_coach:
                            results.append({
                                "employee": profile,
                                "can_coach_on": can_coach,
                                "coaching_skills_count": len(can_coach)
                            })

                    # Sort by number of coachable skills
                    results.sort(key=lambda x: x["coaching_skills_count"], reverse=True)
                    top_results = results[:top_n]

                    return json.dumps({
                        "mode": "multi_skill_coaching",
                        "total_coaches": len(results),
                        "coaches": top_results
                    }, default=str)

                # MODE 2: Single skill and/or will
                if not skill_name and not will_name:
                    return json.dumps({"error": "Either 'skill', 'will', or 'skills' array is required"})

                search_params = {"limit": 5, "offset": 0}
                if filter_location:
                    search_params["location"] = filter_location
                if filter_department:
                    search_params["department"] = filter_department
                if skill_name:
                    search_params["skills"] = [{"name": skill_name, "min_level": min_skill_level, "max_level": 0}]
                if will_name:
                    search_params["wills"] = [{"name": will_name, "min_level": min_will_level, "max_level": 0}]

                all_employee_ids = []
                offset = 0
                while True:
                    search_params["offset"] = offset
                    page = self.api.call("/employees/search", search_params)
                    employees = page.get("employees", [])
                    if not employees:
                        break
                    for emp in employees:
                        if exclude_id and emp["id"] == exclude_id:
                            continue
                        all_employee_ids.append(emp["id"])
                    if page.get("next_offset", -1) < 0:
                        break
                    offset = page["next_offset"]

                with ThreadPoolExecutor(max_workers=10) as executor:
                    profiles = list(executor.map(fetch_profile, all_employee_ids))
                profiles = [p for p in profiles if p]

                results = []
                both_mode = skill_name and will_name

                for profile in profiles:
                    skill_level = get_level(profile, skill_name, "skills") if skill_name else 0
                    will_level = get_level(profile, will_name, "wills") if will_name else 0

                    if both_mode:
                        # Check min/max bounds for both
                        if skill_level < min_skill_level or will_level < min_will_level:
                            continue
                        if max_skill_level and skill_level > max_skill_level:
                            continue
                        if max_will_level and will_level > max_will_level:
                            continue
                        results.append({
                            "employee": profile,
                            "skill_level": skill_level,
                            "will_level": will_level,
                            "combined_score": skill_level + will_level,
                            "skill_name": skill_name,
                            "will_name": will_name
                        })
                    else:
                        level = skill_level if skill_name else will_level
                        min_level = min_skill_level if skill_name else min_will_level
                        max_level = max_skill_level if skill_name else max_will_level
                        # Filter by min/max
                        if level < min_level:
                            continue
                        if max_level and level > max_level:
                            continue
                        results.append({
                            "employee": profile,
                            "matched_level": level,
                            "match_type": "skill" if skill_name else "will",
                            "match_name": skill_name or will_name
                        })

                # Sort by level (desc = highest first, asc = lowest first)
                sort_reverse = (sort_order != "asc")
                if both_mode:
                    results.sort(key=lambda x: x["combined_score"], reverse=sort_reverse)
                else:
                    results.sort(key=lambda x: x["matched_level"], reverse=sort_reverse)

                top_results = results[:top_n]

                return json.dumps({
                    "mode": "skill_AND_will" if both_mode else ("skill" if skill_name else "will"),
                    "total_matched": len(results),
                    "top_employees": top_results
                }, default=str)

            # Composite tool: find projects by employee (uses SDK native team filter)
            if tool_name == "find_projects_by_employee":
                employee_id = tool_input.get("employee_id")
                employee_name = tool_input.get("employee_name")
                filter_role = tool_input.get("role")
                filter_status = tool_input.get("status")
                include_archived = tool_input.get("include_archived", False)

                # Resolve employee name to ID if needed
                if not employee_id and employee_name:
                    search_result = self.api.call("/employees/search", {"query": employee_name, "limit": 1})
                    employees = search_result.get("employees", [])
                    if employees:
                        employee_id = employees[0].get("id")
                    else:
                        return json.dumps({"error": f"Employee '{employee_name}' not found"})

                if not employee_id:
                    return json.dumps({"error": "Either employee_id or employee_name is required"})

                # Step 1: Get project IDs
                all_project_ids = []
                offset = 0

                if include_archived:
                    # Use /projects/list to include ALL projects (including archived)
                    while True:
                        page = self.api.call("/projects/list", {"limit": 5, "offset": offset})
                        projects = page.get("projects", [])
                        if not projects:
                            break
                        for proj in projects:
                            all_project_ids.append(proj["id"])
                        if page.get("next_offset", -1) < 0:
                            break
                        offset = page["next_offset"]
                else:
                    # Use SDK native team filter (faster but excludes archived)
                    search_params = {"limit": 5, "offset": 0}
                    search_params["team"] = {"employee_id": employee_id}
                    if filter_role:
                        search_params["team"]["role"] = filter_role
                    if filter_status:
                        search_params["status"] = [filter_status]

                    while True:
                        search_params["offset"] = offset
                        page = self.api.call("/projects/search", search_params)
                        projects = page.get("projects", [])
                        if not projects:
                            break
                        for proj in projects:
                            all_project_ids.append(proj["id"])
                        if page.get("next_offset", -1) < 0:
                            break
                        offset = page["next_offset"]

                # Step 2: Fetch full project details in parallel
                def fetch_project(proj_id):
                    try:
                        result = self.api.call("/projects/get", {"id": proj_id})
                        return result.get("project", {})
                    except:
                        return None

                with ThreadPoolExecutor(max_workers=10) as executor:
                    projects = list(executor.map(fetch_project, all_project_ids))
                projects = [p for p in projects if p]

                # Step 3: Extract role info for the employee (filter by employee_id for include_archived mode)
                matching = []
                for proj in projects:
                    # Apply status filter if specified
                    if filter_status and proj.get("status") != filter_status:
                        continue
                    team = proj.get("team", [])
                    for member in team:
                        if member.get("employee") == employee_id:
                            # Apply role filter if specified
                            if filter_role and member.get("role") != filter_role:
                                continue
                            matching.append({
                                "project": proj,
                                "employee_role": member.get("role"),
                                "time_slice": member.get("time_slice")
                            })
                            break

                return json.dumps({
                    "employee_id": employee_id,
                    "projects_found": len(matching),
                    "projects": matching
                }, default=str)

            # Composite tool: find projects by role presence (uses SDK filter for has_role=true)
            if tool_name == "find_projects_by_role":
                role = tool_input.get("role")
                has_role = tool_input.get("has_role", True)
                filter_status = tool_input.get("status")

                if not role:
                    return json.dumps({"error": "role parameter is required"})

                # Build search params
                search_params = {"limit": 5, "offset": 0}
                if filter_status:
                    search_params["status"] = [filter_status]

                # For has_role=true, use SDK native team.role filter
                if has_role:
                    search_params["team"] = {"role": role}

                # Paginate to collect project IDs
                all_project_ids = []
                offset = 0
                while True:
                    search_params["offset"] = offset
                    page = self.api.call("/projects/search", search_params)
                    projects = page.get("projects", [])
                    if not projects:
                        break
                    for proj in projects:
                        all_project_ids.append(proj["id"])
                    if page.get("next_offset", -1) < 0:
                        break
                    offset = page["next_offset"]

                # Fetch full project details in parallel
                def fetch_project(proj_id):
                    try:
                        result = self.api.call("/projects/get", {"id": proj_id})
                        return result.get("project", {})
                    except:
                        return None

                with ThreadPoolExecutor(max_workers=10) as executor:
                    projects = list(executor.map(fetch_project, all_project_ids))
                projects = [p for p in projects if p]

                # Build results (for has_role=false, filter locally since SDK doesn't support NOT)
                matching = []
                for proj in projects:
                    team = proj.get("team", [])
                    roles_in_team = [m.get("role") for m in team]
                    role_present = role in roles_in_team

                    if has_role and role_present:
                        # Find the team member with this role
                        for member in team:
                            if member.get("role") == role:
                                matching.append({
                                    "project": proj,
                                    "role_holder": member
                                })
                                break
                    elif not has_role and not role_present:
                        matching.append({
                            "project": proj,
                            "missing_role": role
                        })

                return json.dumps({
                    "role": role,
                    "has_role": has_role,
                    "projects_found": len(matching),
                    "projects": matching
                }, default=str)

            # Composite tool: find all project leads with salaries
            if tool_name == "find_project_leads":
                min_salary = tool_input.get("min_salary")
                max_salary = tool_input.get("max_salary")
                exclude_id = tool_input.get("exclude_employee_id")

                # Step 1: Get all projects (team filter is unreliable, iterate all)
                all_project_ids = []
                offset = 0
                while True:
                    page = self.api.call("/projects/list", {"limit": 5, "offset": offset})
                    projects = page.get("projects", [])
                    if not projects:
                        break
                    for proj in projects:
                        all_project_ids.append(proj["id"])
                    if page.get("next_offset", -1) < 0:
                        break
                    offset = page["next_offset"]

                # Step 2: Fetch project details in parallel to get lead IDs
                def fetch_project(proj_id):
                    try:
                        result = self.api.call("/projects/get", {"id": proj_id})
                        return result.get("project", {})
                    except:
                        return None

                with ThreadPoolExecutor(max_workers=10) as executor:
                    projects = list(executor.map(fetch_project, all_project_ids))
                projects = [p for p in projects if p]

                # Step 3: Collect unique lead employee IDs and their projects
                lead_projects = {}  # {emp_id: [(project_id, project_name, time_slice), ...]}
                for proj in projects:
                    team = proj.get("team", [])
                    for member in team:
                        if member.get("role") == "Lead":
                            emp_id = member.get("employee")
                            if emp_id not in lead_projects:
                                lead_projects[emp_id] = []
                            lead_projects[emp_id].append({
                                "project_id": proj.get("id"),
                                "project_name": proj.get("name"),
                                "time_slice": member.get("time_slice", 0)
                            })

                # Step 4: Get salaries from employees_list (salary is in EmployeeBrief)
                employee_salaries = {}
                employee_names = {}
                offset = 0
                while True:
                    page = self.api.call("/employees/list", {"limit": 5, "offset": offset})
                    employees = page.get("employees", [])
                    if not employees:
                        break
                    for emp in employees:
                        employee_salaries[emp["id"]] = emp.get("salary", 0)
                        employee_names[emp["id"]] = emp.get("name", "")
                    if page.get("next_offset", -1) < 0:
                        break
                    offset = page["next_offset"]

                # Step 5: Build results with filters
                results = []
                for emp_id, projs in lead_projects.items():
                    # Exclude if requested
                    if exclude_id and emp_id == exclude_id:
                        continue

                    salary = employee_salaries.get(emp_id, 0)

                    # Apply salary filters
                    if min_salary and salary < min_salary:
                        continue
                    if max_salary and salary > max_salary:
                        continue

                    total_fte = sum(p.get("time_slice", 0) for p in projs)
                    results.append({
                        "employee_id": emp_id,
                        "name": employee_names.get(emp_id, ""),
                        "salary": salary,
                        "projects_as_lead": projs,
                        "total_lead_fte": round(total_fte, 2)
                    })

                # Sort by salary descending
                results.sort(key=lambda x: x["salary"], reverse=True)

                return json.dumps({
                    "total_leads": len(results),
                    "leads": results
                }, default=str)

            # Composite tool: find employees by location (for workload analysis)
            if tool_name == "find_employees_by_location":
                location = tool_input.get("location")
                department = tool_input.get("department")

                if not location:
                    return json.dumps({"error": "location parameter is required"})

                # Paginate through all employees at this location
                search_params = {"limit": 5, "offset": 0, "location": location}
                if department:
                    search_params["department"] = department

                all_employees = []
                offset = 0
                while True:
                    search_params["offset"] = offset
                    page = self.api.call("/employees/search", search_params)
                    employees = page.get("employees", [])
                    if not employees:
                        break
                    all_employees.extend(employees)
                    if page.get("next_offset", -1) < 0:
                        break
                    offset = page["next_offset"]

                return json.dumps({
                    "location": location,
                    "department": department,
                    "total_employees": len(all_employees),
                    "employees": all_employees
                }, default=str)

            # Composite tool: calculate workloads for multiple employees in ONE call
            if tool_name == "calculate_workloads":
                employee_ids = tool_input.get("employee_ids", [])

                if not employee_ids:
                    return json.dumps({"error": "employee_ids array is required"})

                # Initialize workload map
                workload_map = {emp_id: 0.0 for emp_id in employee_ids}
                employee_id_set = set(employee_ids)

                # Fetch all projects (pagination)
                all_project_ids = []
                offset = 0
                while True:
                    page = self.api.call("/projects/list", {"limit": 5, "offset": offset})
                    projects = page.get("projects", [])
                    if not projects:
                        break
                    for proj in projects:
                        all_project_ids.append(proj["id"])
                    if page.get("next_offset", -1) < 0:
                        break
                    offset = page["next_offset"]

                # Fetch project details in parallel
                def fetch_project(proj_id):
                    try:
                        result = self.api.call("/projects/get", {"id": proj_id})
                        return result.get("project", {})
                    except:
                        return None

                with ThreadPoolExecutor(max_workers=10) as executor:
                    projects = list(executor.map(fetch_project, all_project_ids))
                projects = [p for p in projects if p]

                # Calculate workloads (count ALL projects including archived)
                for proj in projects:
                    team = proj.get("team", [])
                    for member in team:
                        emp_id = member.get("employee")
                        if emp_id in employee_id_set:
                            workload_map[emp_id] += member.get("time_slice", 0)

                # Round values
                workload_map = {k: round(v, 2) for k, v in workload_map.items()}

                # Find min and max
                min_workload = min(workload_map.values()) if workload_map else 0
                max_workload = max(workload_map.values()) if workload_map else 0

                return json.dumps({
                    "total_employees": len(employee_ids),
                    "total_projects_scanned": len(projects),
                    "workloads": workload_map,
                    "min_workload": min_workload,
                    "max_workload": max_workload,
                    "employees_at_min": [k for k, v in workload_map.items() if v == min_workload],
                    "employees_at_max": [k for k, v in workload_map.items() if v == max_workload]
                }, default=str)

            # Special handler for time_update - fill missing fields from original entry
            if tool_name == "time_update":
                entry_id = tool_input.get("id")
                if not entry_id:
                    return json.dumps({"error": "id is required for time_update"})

                # Get current user for changed_by
                whoami_result = self.api.call("/whoami", {})
                current_user = whoami_result.get("current_user")

                # Get current entry to fill missing fields
                current = self.api.call("/time/get", {"id": entry_id})
                entry = current.get("entry", {})
                if not entry:
                    return json.dumps({"error": f"Time entry {entry_id} not found"})

                # Merge: use provided values or fall back to current
                update_params = {
                    "id": entry_id,
                    "date": tool_input.get("date", entry.get("date")),
                    "hours": tool_input.get("hours", entry.get("hours")),
                    "work_category": tool_input.get("work_category", entry.get("work_category", "customer_project")),
                    "notes": tool_input.get("notes", entry.get("notes", "")),
                    "billable": tool_input.get("billable", entry.get("billable", True)),
                    "status": tool_input.get("status", entry.get("status")),
                    "changed_by": current_user
                }

                result = self.api.call("/time/update", update_params)
                return json.dumps(result, default=str)

            # Special handler for wiki_rename - copy content without LLM transformation
            if tool_name == "wiki_rename":
                source = tool_input.get("source")
                destination = tool_input.get("destination")
                delete_source = tool_input.get("delete_source", True)

                if not source or not destination:
                    return json.dumps({"error": "source and destination are required"})

                # Load content from source (exact bytes, no LLM transformation)
                load_result = self.api.call("/wiki/load", {"file": source})
                content = load_result.get("content")
                if content is None:
                    return json.dumps({"error": f"Source file '{source}' not found"})

                # Save to destination with exact content
                self.api.call("/wiki/update", {"file": destination, "content": content})

                # Delete source if requested
                if delete_source:
                    self.api.call("/wiki/update", {"file": source, "content": ""})

                return json.dumps({
                    "success": True,
                    "source": source,
                    "destination": destination,
                    "deleted_source": delete_source
                })

            # Composite tool: batch wiki update - multiple files in one call
            if tool_name == "wiki_batch_update":
                updates = tool_input.get("updates", [])
                if not updates:
                    return json.dumps({"error": "updates array is required"})

                results = []
                for item in updates:
                    file_path = item.get("file")
                    content = item.get("content", "")
                    if file_path:
                        try:
                            self.api.call("/wiki/update", {"file": file_path, "content": content})
                            results.append({"file": file_path, "success": True})
                        except Exception as e:
                            results.append({"file": file_path, "success": False, "error": str(e)})

                return json.dumps({
                    "total": len(updates),
                    "successful": sum(1 for r in results if r.get("success")),
                    "results": results
                })

            endpoint = endpoint_map.get(tool_name)
            if not endpoint:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

            # Apply default values for pagination (required by API)
            if tool_name in ["employees_list", "employees_search", "projects_list", "projects_search",
                             "customers_list", "customers_search", "time_search"]:
                if "limit" not in tool_input:
                    tool_input["limit"] = 5
                if "offset" not in tool_input:
                    tool_input["offset"] = 0

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

        # Versioner для записи истории
        try:
            self.versioner = Versioner(evolution_dir)
        except Exception:
            self.versioner = None

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

    def _call_llm_with_retry(self, messages, max_retries=10, base_delay=10):
        """Call Claude API with exponential backoff retry for rate limits"""
        for attempt in range(max_retries):
            try:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=10000,
                    system=[{
                        "type": "text",
                        "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }],
                    tools=self.tools,
                    messages=messages
                )
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise  # Last attempt, re-raise
                delay = base_delay * (2 ** attempt)  # 10, 20, 40, 80, 160 sec
                print(f"  [Rate limit] Retry {attempt + 1}/{max_retries} in {delay}s...")
                time.sleep(delay)
        raise RuntimeError("Max retries exceeded")

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
            response = self._call_llm_with_retry(messages)

            try:
                # Собираем completion из text blocks и tool_use
                completion_parts = []
                for block in response.content:
                    if hasattr(block, 'text'):
                        completion_parts.append(block.text)
                    elif hasattr(block, 'type') and block.type == 'tool_use':
                        completion_parts.append(f"[tool_use: {block.name}]")

                completion_text = "\n".join(completion_parts) if completion_parts else "[no text content]"

                self.core.log_llm(
                    task_id=task.task_id,
                    model="anthropic/" + self.model,
                    duration_sec=time.time() - started,
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    cached_prompt_tokens=getattr(response.usage, 'cache_read_input_tokens', 0) or 0,
                    completion=completion_text,
                )
            except ApiException as e:
                print(f"  [log_llm ApiException] {e}")
            except Exception as e:
                print(f"  [log_llm Exception] {type(e).__name__}: {e}")

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
                    self.logger.log_turn(task.spec_id, turn + 1, text_blocks, [])
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
                                self.logger.log_turn(task.spec_id, turn + 1, text_blocks, tool_calls_log)
                            messages.append({"role": "user", "content": tool_results})
                            print(f"  → Task completed: {executor.completion_summary}")
                            return {"completed": True, "summary": executor.completion_summary}

                # Логируем turn даже если task не завершён
                if self.logger:
                    self.logger.log_turn(task.spec_id, turn + 1, text_blocks, tool_calls_log)

                messages.append({"role": "user", "content": tool_results})
                messages = self._compress_history(messages)

        return {"completed": executor.task_completed, "summary": executor.completion_summary or "Max turns reached"}

    def run_session(self, benchmark: str = "erc3-dev", workspace: str = "demo", name: str = "Claude Company Agent", task_filter: list = None, flags: list = None):
        """Run a full session solving all tasks (or filtered subset)"""

        print(f"\nStarting session: benchmark={benchmark}, workspace={workspace}")
        if task_filter:
            print(f"Task filter: {task_filter}")
        if flags:
            print(f"Competition flags: {flags}")

        res = self.core.start_session(
            benchmark=benchmark,
            workspace=workspace,
            name=name,
            architecture=f"Anthropic SDK Agent with {self.model}",
            flags=flags or []
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
            self.logger.end_task(task.spec_id, best_score, result.get("summary", ""), eval_logs)

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

        # Записать результат в evolution history
        if self.versioner and self.config_version > 0:
            tasks_passed = sum(1 for r in results if r["score"] >= 1.0)
            self.versioner.record_run_result(
                score=total_score,
                tasks_passed=tasks_passed,
                total=len(results)
            )
            print(f"[Evolution] Recorded v{self.config_version:03d}: {tasks_passed}/{len(results)} passed")

        return results

    def _run_single_task(self, task, session_id: str) -> dict:
        """Run a single task - for parallel execution"""
        # Each task gets its own API client
        erc_client = self.core.get_erc_client(task)
        api_base = erc_client.base_url.rstrip('/')
        api_client = CompanyAPIClient(api_base, {"Authorization": f"Bearer {self.core.key}"})

        # Logger handles its own locking
        self.logger.start_task(task.spec_id, task.task_text)

        best_score = 0.0
        result = {}
        eval_logs = None

        for attempt in range(MAX_RETRIES):
            if attempt > 0:
                safe_print(f"  [{task.spec_id}] RETRY {attempt + 1}/{MAX_RETRIES}")

            result = self.solve_task_parallel(task, api_client)
            eval_result = self.core.complete_task(task)

            score = eval_result.eval.score if eval_result.eval else 0.0
            eval_logs = eval_result.eval.logs if eval_result.eval else None

            safe_print(f"  [{task.spec_id}] DONE")

            if score >= 1.0:
                best_score = score
                break

            best_score = max(best_score, score)

        # Log completion (logger handles its own locking)
        self.logger.end_task(task.spec_id, best_score, result.get("summary", ""), eval_logs)

        return {
            "task_id": task.task_id,
            "spec_id": task.spec_id,
            "score": best_score
        }

    def solve_task_parallel(self, task, api_client) -> dict:
        """Solve a single task - parallel version with safe_print"""
        executor = ToolExecutor(api_client, self.rulebook_sections)

        messages = [
            {"role": "user", "content": f"TASK: {task.task_text}\n\nComplete this task for the company system."}
        ]

        safe_print(f"\n[{task.spec_id}] START: {task.task_text[:60]}...")

        for turn in range(MAX_TURNS):
            started = time.time()
            response = self._call_llm_with_retry(messages)

            try:
                completion_parts = []
                for block in response.content:
                    if hasattr(block, 'text'):
                        completion_parts.append(block.text)
                    elif hasattr(block, 'type') and block.type == 'tool_use':
                        completion_parts.append(f"[tool_use: {block.name}]")

                completion_text = "\n".join(completion_parts) if completion_parts else "[no text content]"

                self.core.log_llm(
                    task_id=task.task_id,
                    model="anthropic/" + self.model,
                    duration_sec=time.time() - started,
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    cached_prompt_tokens=getattr(response.usage, 'cache_read_input_tokens', 0) or 0,
                    completion=completion_text,
                )
            except Exception:
                pass  # Silent in parallel mode

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            text_blocks = []
            tool_calls_log = []

            if response.stop_reason == "end_turn":
                for block in assistant_content:
                    if block.type == "text":
                        text_blocks.append(block.text)
                if self.logger and text_blocks:
                    self.logger.log_turn(task.spec_id, turn + 1, text_blocks, [])
                break

            if response.stop_reason == "tool_use":
                tool_results = []

                for block in assistant_content:
                    if block.type == "text":
                        text_blocks.append(block.text)

                for block in assistant_content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input

                        tool_start = time.time()
                        result = executor.execute(tool_name, tool_input)
                        tool_duration = time.time() - tool_start

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
                            if self.logger:
                                self.logger.log_turn(task.spec_id, turn + 1, text_blocks, tool_calls_log)
                            messages.append({"role": "user", "content": tool_results})
                            return {"completed": True, "summary": executor.completion_summary}

                if self.logger:
                    self.logger.log_turn(task.spec_id, turn + 1, text_blocks, tool_calls_log)

                messages.append({"role": "user", "content": tool_results})
                messages = self._compress_history(messages)

        return {"completed": executor.task_completed, "summary": executor.completion_summary or "Max turns reached"}

    def run_session_parallel(self, benchmark: str = "erc3-dev", workspace: str = "demo",
                            name: str = None, task_filter: list = None,
                            flags: list = None, max_workers: int = 5):
        """Run a full session with parallel task execution"""

        # Default name format
        if name is None:
            name = f"@aostrikov claude evolution v{self.config_version:03d}"

        print(f"\nStarting PARALLEL session: benchmark={benchmark}, workers={max_workers}")
        if task_filter:
            print(f"Task filter: {task_filter}")
        if flags:
            print(f"Competition flags: {flags}")

        res = self.core.start_session(
            benchmark=benchmark,
            workspace=workspace,
            name=name,
            architecture=f"Anthropic SDK Agent PARALLEL ({max_workers}w) with {self.model}",
            flags=flags or []
        )

        session_id = res.session_id
        print(f"Session ID: {session_id}")

        self.logger = TaskLogger("output", session_id, self.config_version)
        print(f"Logs: {self.logger.session_dir}")
        if self.config_version > 0:
            print(f"Config: v{self.config_version:03d}")

        status = self.core.session_status(session_id)
        print(f"Tasks: {len(status.tasks)}")

        tasks_to_run = status.tasks
        if task_filter:
            tasks_to_run = [t for t in status.tasks if t.spec_id in task_filter]
            print(f"Running {len(tasks_to_run)} of {len(status.tasks)} tasks")

        # Start all tasks first (required by ERC3 API)
        print(f"\nStarting {len(tasks_to_run)} tasks...")
        for task in tasks_to_run:
            self.core.start_task(task)

        results = []
        print(f"\nRunning tasks in parallel with {max_workers} workers...\n")

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="Task") as pool:
            futures = {
                pool.submit(self._run_single_task, task, session_id): task
                for task in tasks_to_run
            }

            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    safe_print(f"  ✓ {result['spec_id']} completed")
                except Exception as e:
                    safe_print(f"  ✗ {task.spec_id} FAILED: {e}")
                    results.append({
                        "task_id": task.task_id,
                        "spec_id": task.spec_id,
                        "score": 0.0
                    })

        # Sort results by spec_id for consistent output
        results.sort(key=lambda x: x['spec_id'])

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
            status_icon = "✓" if r["score"] >= 1.0 else "○"
            print(f"  {status_icon} [{r['spec_id']}]")
            total_score += r["score"]

        print(f"\nTotal: {total_score}/{len(results)} ({total_score/len(results)*100:.1f}% if scored)")

        self.logger.save_summary(results)

        # Записать результат в evolution history
        if self.versioner and self.config_version > 0:
            tasks_passed = sum(1 for r in results if r["score"] >= 1.0)
            self.versioner.record_run_result(
                score=total_score,
                tasks_passed=tasks_passed,
                total=len(results)
            )
            print(f"[Evolution] Recorded v{self.config_version:03d}: {tasks_passed}/{len(results)} passed")

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
    parser.add_argument("--flags", nargs="+", default=[], help="Competition flags: compete_accuracy, compete_budget, compete_speed, compete_local")
    parser.add_argument("--extract-wiki", action="store_true", help="Extract unique wiki content to wiki/ folder")
    parser.add_argument("--parallel", action="store_true", help="Run tasks in parallel")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers (default: 5)")

    args = parser.parse_args()

    if args.extract_wiki:
        # Extract wiki content using same approach as run_session
        agent = ERC3Agent(model=args.model)
        core = agent.core
        session = core.start_session(benchmark=args.benchmark, workspace='wiki_extract', name='extractor', architecture='extraction', flags=args.flags)
        print(f'Session: {session.session_id}')

        status = core.session_status(session.session_id)
        print(f'Total tasks: {len(status.tasks)}')

        hashes = {}
        for i, t in enumerate(status.tasks):
            task = core.start_task(t)
            # Use same client approach as run_session
            erc_client = core.get_erc_client(task)
            api_base = erc_client.base_url.rstrip('/')
            api_client = CompanyAPIClient(api_base, {"Authorization": f"Bearer {core.key}"})

            # Get wiki files
            wiki_result = api_client.call("/wiki/list", {})
            files = wiki_result.get('paths', [])
            sha1 = wiki_result.get('sha1', 'unknown')[:8]

            if sha1 not in hashes:
                hashes[sha1] = {'files': files, 'count': 0, 'api_client': api_client, 'task': task}
            else:
                core.complete_task(task)
            hashes[sha1]['count'] += 1

            if (i+1) % 25 == 0:
                print(f'Scanned {i+1}/{len(status.tasks)}')

        print(f'\n=== UNIQUE HASHES: {len(hashes)} ===')
        for h, info in hashes.items():
            print(f'{h}: {info["count"]} tasks, {len(info["files"])} files')

        # Extract wiki content
        print('\n=== EXTRACTING ===')
        for sha1, info in hashes.items():
            wiki_dir = f'wiki/hash_{sha1}_prod'
            os.makedirs(wiki_dir, exist_ok=True)
            api_client = info['api_client']
            for f in info['files']:
                # Create subdirectories if needed
                file_path = f'{wiki_dir}/{f}'
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                content = api_client.call("/wiki/load", {"file": f}).get('content', '')
                with open(f'{wiki_dir}/{f}', 'w') as out:
                    out.write(content)
                print(f'  {wiki_dir}/{f}')
            core.complete_task(info['task'])

        print('\nDone!')
        exit(0)

    agent = ERC3Agent(model=args.model)

    if args.parallel:
        agent.run_session_parallel(
            benchmark=args.benchmark,
            workspace=args.workspace,
            name=args.name,
            task_filter=args.tasks,
            flags=args.flags,
            max_workers=args.workers
        )
    else:
        agent.run_session(
            benchmark=args.benchmark,
            workspace=args.workspace,
            name=args.name,
            task_filter=args.tasks,
            flags=args.flags
        )
