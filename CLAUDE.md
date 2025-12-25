# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agents for the ERC3: AI Agents in Action competition. Connect to the ERC3 platform and solve benchmark tasks.

**Two architectural approaches:**
1. **SGR (Schema-Guided Reasoning)**: Pydantic models + OpenAI structured outputs (`sgr-agent-*`)
2. **Anthropic SDK**: Native tool use with Claude (`claude-agent-*`)

## Environment Variables

```bash
export ERC3_API_KEY=key-...      # https://erc.timetoact-group.at/
export OPENAI_API_KEY=sk-...     # For SGR agents
export ANTHROPIC_API_KEY=...     # For Claude agents
```

## Running Agents

### claude-agent-erc3-prod (Production Agent)

```bash
cd claude-agent-erc3-prod

# Full run (103 tasks, parallel)
./run.sh parallel 5

# Single task (no session submit)
./run.sh task t017

# Multiple tasks
./run.sh tasks t013,t014,t017

# Check current config version
./run.sh version

# Show failed tasks from last session
./run.sh failed

# View task log
./run.sh log t017
```

### SGR Agents

```bash
cd sgr-agent-store  # or sgr-agent-erc3
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

## Architecture

### claude-agent-erc3-prod

**Evolution System** (`evolution/`):
- `state.json` → `current_version` points to active config
- `vXXX/config.json` → versioned configuration with:
  - `base_prompt`: core system prompt with permission rules
  - `rules[]`: learned behavioral rules (added incrementally)
  - `examples[]`: few-shot examples for correct behavior
  - `tool_patches{}`: enhanced tool descriptions
  - `rulebook_sections{}`: detailed access rules

**Key Components** (`agent.py`):
- `TOOLS[]`: 32 tool definitions (25 simple + 7 composite)
- `ToolExecutor`: Maps tool calls to ERC3 API
- `ERC3Agent.run_session()`: Sequential execution
- `ERC3Agent.run_session_parallel()`: Parallel with ThreadPoolExecutor
- `SessionLogger`: Per-task JSON logs in `output/` folder

**Composite Tools** (optimize API calls):
- `find_employees_by_skill` - skill/will search with sorting
- `find_projects_by_employee` - all projects for employee
- `find_project_leads` - all project leads with salary filters
- `calculate_workloads` - batch workload calculation
- `wiki_batch_update` - bulk wiki updates

### SGR Agents

**Core Pattern** (`store_agent.py`):
1. `NextStep` Pydantic model with fields: `current_state`, `plan_remaining_steps_brief`, `task_completed`, `function`
2. Loop: LLM → parse NextStep → dispatch tool → log result → repeat
3. `function` is Union of all benchmark tools + `ReportTaskCompletion`

## ERC3 SDK

```python
from erc3 import ERC3, store, ApiException

core = ERC3()
# For SGR: store_api = core.get_store_client(task); store_api.dispatch(store.Req_ListProducts(...))
# For Claude: HTTP client to {base_url}/{benchmark}/{task_id}/{endpoint}
core.log_llm(task_id, model, duration_sec, usage)  # Telemetry for leaderboard
```

**Benchmarks**: `store` (shop with cart), `erc3-dev`/`erc3-prod` (corporate assistant)

## Adding Agent for New Benchmark

1. Import models: `from erc3 import {benchmark}`
2. Define tools/NextStep schema with benchmark operations
3. Get client: `core.get_{benchmark}_client(task)` or HTTP client
4. Copy session loop from existing agent

## Web UI

https://erc.timetoact-group.at/ — view sessions, tasks, logs
