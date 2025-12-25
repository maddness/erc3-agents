#!/usr/bin/env python3
"""
Debug script for project_status_change_by_lead task.
Direct HTTP API exploration.
"""

import os
import json
import httpx
from dotenv import load_dotenv
from erc3 import ERC3

load_dotenv(dotenv_path="../.env")

class SimpleAPI:
    """Simple HTTP client for erc3-dev API"""
    def __init__(self, task):
        self.base_url = f"https://erc.timetoact-group.at/erc3-dev/{task.task_id}"
        self.http = httpx.Client(verify=False, timeout=30)

    def call(self, endpoint: str, payload: dict = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        resp = self.http.post(url, json=payload or {})
        resp.raise_for_status()
        return resp.json()

def main():
    core = ERC3()

    print("=" * 60)
    print("CREATING SESSION")
    print("=" * 60)

    session = core.start_session(
        benchmark="erc3-dev",
        workspace="demo",
        name="Debug Session",
        architecture="manual-debug"
    )
    print(f"Session ID: {session.session_id}")

    status = core.session_status(session.session_id)
    print(f"\nTotal tasks: {len(status.tasks)}")

    target_task = None
    for t in status.tasks:
        if t.spec_id == "project_status_change_by_lead":
            target_task = t
            print(f"\nFOUND: {t.spec_id}")
            print(f"Task text: {t.task_text}")

    if not target_task:
        print("Task not found!")
        return

    print("\n" + "=" * 60)
    print("STARTING TASK")
    print("=" * 60)

    task = core.start_task(target_task)
    api = SimpleAPI(task)

    print(f"Task ID: {task.task_id}")
    print(f"API Base: {api.base_url}")

    # STEP 1: whoami
    print("\n--- STEP 1: whoami ---")
    result = api.call("/whoami", {})
    print(json.dumps(result, indent=2))
    user_id = result.get("current_user")

    # STEP 2: employees_get (self)
    print("\n--- STEP 2: employees_get (self) ---")
    result = api.call("/employees/get", {"id": user_id})
    print(json.dumps(result, indent=2))

    # Check if employee has projects field
    emp = result.get("employee", {})
    emp_projects = emp.get("projects", [])
    print(f"\n>>> Employee projects in profile: {emp_projects}")

    # STEP 3: projects_list
    print("\n--- STEP 3: projects_list ---")
    result = api.call("/projects/list", {})
    print(json.dumps(result, indent=2))

    projects = result.get("projects", [])
    print(f"\n>>> Found {len(projects) if projects else 0} projects in list")

    # STEP 4: projects_search with empty query
    print("\n--- STEP 4: projects_search (empty) ---")
    result = api.call("/projects/search", {"query": ""})
    print(json.dumps(result, indent=2))

    # STEP 5: Extract project name from task text and search
    task_text = target_task.task_text
    # Parse "Change status of project X to Y"
    import re
    match = re.search(r"project (.+?) to (\w+)", task_text)
    if match:
        project_name = match.group(1)
        target_status = match.group(2)
        print(f"\n>>> Parsed from task: project='{project_name}', target_status='{target_status}'")

        print(f"\n--- STEP 5: projects_search ('{project_name}') ---")
        result = api.call("/projects/search", {"query": project_name})
        print(json.dumps(result, indent=2))

        # Try variations
        words = project_name.split()
        for word in words[:2]:
            if len(word) > 3:
                print(f"\n--- Search by word: '{word}' ---")
                result = api.call("/projects/search", {"query": word})
                projects_found = result.get("projects", [])
                if projects_found:
                    for p in projects_found:
                        print(f"  Found: {p.get('id')}: {p.get('name')} (status: {p.get('status')})")

    # STEP 6: Get customers and their projects
    print("\n--- STEP 6: customers_list ---")
    result = api.call("/customers/list", {})
    customers = result.get("customers", [])
    print(f"Found {len(customers)} customers")
    for c in customers[:5]:
        cid = c.get("id")
        cname = c.get("name")
        # Get customer details
        cdetail = api.call("/customers/get", {"id": cid})
        customer = cdetail.get("customer", {})
        cprojects = customer.get("projects", [])
        print(f"  {cid}: {cname} -> {len(cprojects)} projects")
        for p in cprojects[:3]:
            if isinstance(p, dict):
                print(f"      - {p.get('id')}: {p.get('name')}")
            else:
                print(f"      - {p}")

    # STEP 7: Try to find project by guessing ID
    if match:
        print("\n--- STEP 7: Try projects_get with guessed IDs ---")
        # Generate possible IDs from name
        name_lower = project_name.lower().replace(" ", "_").replace("&", "and")
        name_clean = re.sub(r'[^a-z0-9_]', '', name_lower)

        test_ids = [
            name_clean,
            f"proj_{name_clean}",
            name_lower.replace(" ", "-"),
            project_name.replace(" ", "_"),
        ]

        for pid in test_ids:
            result = api.call("/projects/get", {"id": pid})
            found = result.get("found", False)
            print(f"  {pid}: found={found}")
            if found:
                proj = result.get("project", {})
                print(f"    Name: {proj.get('name')}")
                print(f"    Status: {proj.get('status')}")
                print(f"    Lead: {proj.get('lead')}")
                print(f"    Members: {proj.get('members', [])}")

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE - Check which project the user is Lead of")
    print("=" * 60)

if __name__ == "__main__":
    main()
