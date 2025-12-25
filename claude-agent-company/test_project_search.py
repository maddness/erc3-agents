#!/usr/bin/env python3
"""
Test project search strategies to find optimal approach.
"""

import os
import json
import httpx
from dotenv import load_dotenv
from erc3 import ERC3

load_dotenv(dotenv_path="../.env")

class SimpleAPI:
    def __init__(self, task):
        self.base_url = f"https://erc.timetoact-group.at/erc3-dev/{task.task_id}"
        self.http = httpx.Client(verify=False, timeout=30)

    def call(self, endpoint: str, payload: dict = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        try:
            resp = self.http.post(url, json=payload or {})
            return resp.json()
        except Exception as e:
            return {"error": str(e), "status": resp.status_code if 'resp' in dir() else None}

def test_search(api, description, params):
    """Test a search strategy and report results"""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"Params: {json.dumps(params)}")
    print("="*60)

    result = api.call("/projects/search", params)

    if "error" in result:
        print(f"ERROR: {result}")
        return 0

    projects = result.get("projects", [])
    next_offset = result.get("next_offset", -1)

    if not projects:
        print("NO RESULTS")
        return 0

    print(f"Found {len(projects)} projects (next_offset={next_offset}):")
    for p in projects:
        print(f"  - {p.get('name')[:50]} [{p.get('status')}]")

    return len(projects)

def main():
    core = ERC3()

    # Create test session
    session = core.start_session(
        benchmark="erc3-dev",
        workspace="demo",
        name="Search Test",
        architecture="test"
    )

    status = core.session_status(session.session_id)

    # Find a task that requires project search
    target_task = None
    for t in status.tasks:
        if t.spec_id == "project_status_change_by_lead":
            target_task = t
            break

    if not target_task:
        print("Task not found!")
        return

    task = core.start_task(target_task)
    api = SimpleAPI(task)

    print(f"\nTask: {target_task.task_text}")
    print(f"Looking for: 'Platform Safety Monitoring PoC'")

    # Test 1: Exact name with limit=5
    test_search(api, "Exact name, limit=5", {
        "query": "Platform Safety Monitoring PoC",
        "limit": 5
    })

    # Test 2: Exact name without limit
    test_search(api, "Exact name, no limit", {
        "query": "Platform Safety Monitoring PoC"
    })

    # Test 3: First word only
    test_search(api, "First word 'Platform'", {
        "query": "Platform",
        "limit": 5
    })

    # Test 4: Two words
    test_search(api, "Two words 'Platform Safety'", {
        "query": "Platform Safety",
        "limit": 5
    })

    # Test 5: Partial word
    test_search(api, "Partial 'Monitoring'", {
        "query": "Monitoring",
        "limit": 5
    })

    # Test 6: Empty query (pagination baseline)
    test_search(api, "Empty query (baseline)", {
        "limit": 5,
        "offset": 0
    })

    # Test 7: With include_archived
    test_search(api, "Empty + include_archived=true", {
        "limit": 5,
        "offset": 0,
        "include_archived": True
    })

    # Test 8: Query + include_archived
    test_search(api, "Query 'Platform' + include_archived", {
        "query": "Platform",
        "limit": 5,
        "include_archived": True
    })

    # Test 9: Different project name for comparison
    test_search(api, "Query 'Infrastructure Monitoring'", {
        "query": "Infrastructure Monitoring",
        "limit": 5
    })

    # Test 10: Single unique word
    test_search(api, "Query 'PoC'", {
        "query": "PoC",
        "limit": 5
    })

    print("\n" + "="*60)
    print("SEARCH TESTS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
