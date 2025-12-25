# ERC3 Agents

Agents for the [ERC3: AI Agents in Action](https://www.timetoact-group.at/events/enterprise-rag-challenge-part-3) competition.

## Quick Start

```bash
# Set environment variables
export ERC3_API_KEY=key-...       # Get from https://erc.timetoact-group.at/
export ANTHROPIC_API_KEY=...      # For Claude agents
export OPENAI_API_KEY=sk-...      # For SGR agents

# Run production agent (103 tasks, 5 parallel workers)
cd claude-agent-erc3-prod
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./run.sh parallel 5
```

## Agents

| Agent | Benchmark | Architecture | Score |
|-------|-----------|--------------|-------|
| `claude-agent-erc3-prod` | erc3-prod | Anthropic Claude + Tool Use | 100% |
| `sgr-agent-store` | store | OpenAI + Schema-Guided Reasoning | — |
| `sgr-agent-erc3` | erc3-dev | OpenAI + Schema-Guided Reasoning | — |

## Architecture

### Claude Agent (claude-agent-erc3-prod)

Production agent with evolution system for iterative prompt improvement:

```
evolution/
  state.json          # Current version pointer
  v103/
    config.json       # Prompt, rules, examples, tool patches
```

**Run commands:**
```bash
./run.sh parallel 5      # Full run, 5 workers
./run.sh task t017       # Single task
./run.sh failed          # Show failures from last run
./run.sh version         # Current config version
```

### SGR Agents (sgr-agent-*)

[Schema-Guided Reasoning](https://abdullin.com/schema-guided-reasoning/) with Pydantic models and OpenAI structured outputs.

## Results

Latest run (v103): **103/103 tasks (100%)**
- Wall-clock time: 6.6 min (5 workers)
- Avg per task: 18.9 sec
- Tool calls: 596 total (5.8 per task)

## Resources

- [Competition Page](https://www.timetoact-group.at/events/enterprise-rag-challenge-part-3)
- [ERC3 Platform](https://erc.timetoact-group.at/)
- [Web UI](https://erc.timetoact-group.at/) — view sessions, tasks, logs
