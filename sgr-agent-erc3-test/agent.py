import json
from pathlib import Path
from typing import Annotated, List, Union, Literal, Optional
from annotated_types import MaxLen, MinLen, Gt, Lt
from erc3.erc3 import ProjectDetail
from pydantic import BaseModel, Field
from erc3 import erc3 as dev, ApiException, TaskInfo, ERC3, Erc3Client

from lib import MyLLM

# this is how you can add custom tools that work slightly better
class Req_DeleteWikiPage(BaseModel):
    tool: Literal["/wiki/delete"] = "/wiki/delete"
    file: str
    changed_by: Optional[dev.EmployeeID] = None

class Req_ListAllProjectsForUser(BaseModel):
    tool: Literal["/all-projects-for-user"] = "/all-projects-for-user"
    user: dev.EmployeeID

class Resp_ListAllProjectsForUser(BaseModel):
    lead_in: List[ProjectDetail]
    member_of: List[ProjectDetail]

class Req_ListAllCustomersForUser(BaseModel):
    tool: Literal["/all-customers-for-user"] = "/all-customers-for-user"
    user: dev.EmployeeID

class Resp_ListAllCustomersForUser(BaseModel):
    customers: List[dev.CompanyDetail]

# wrap this with more descriptive names to avoid confusing LLM
class GetTimesheetReportByProject(dev.Req_TimeSummaryByProject):
    pass
class GetTimesheetReportByEmployee(dev.Req_TimeSummaryByEmployee):
    pass
class CreateTimesheetEntryForUser(dev.Req_LogTimeEntry):
    pass

# next-step planner
class NextStep(BaseModel):
    current_state: str
    # we'll use only the first step, discarding all the rest.
    plan_remaining_steps_brief: Annotated[List[str], MinLen(1), MaxLen(5)] =  Field(..., description="explain your thoughts on how to accomplish - what steps to execute")
    # now let's continue the cascade and check with LLM if the task is done
    task_completed: bool
    # Routing to one of the tools to execute the first remaining step
    # if task is completed, model will pick ReportTaskCompletion
    function: Union[
        dev.Req_ProvideAgentResponse,
        dev.Req_ListProjects,
        dev.Req_SearchProjects,
        Req_ListAllProjectsForUser,
        dev.Req_GetProject,
        dev.Req_UpdateProjectTeam,
        dev.Req_UpdateProjectStatus,
        dev.Req_ListEmployees,
        dev.Req_SearchEmployees,
        dev.Req_GetEmployee,
        dev.Req_UpdateEmployeeInfo,
        dev.Req_ListCustomers,
        Req_ListAllCustomersForUser,
        dev.Req_GetCustomer,
        dev.Req_SearchCustomers,
        dev.Req_SearchTimeEntries,
        GetTimesheetReportByProject,
        dev.Req_TimeSummaryByEmployee,
        dev.Req_GetTimeEntry,
        CreateTimesheetEntryForUser,
        dev.Req_UpdateTimeEntry,
        Req_DeleteWikiPage,
    ] = Field(..., description="execute first remaining step")

CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_BLUE = "\x1B[34m"
CLI_CLR = "\x1B[0m"

# custom tool to list my projects
def list_my_projects(api: Erc3Client, user: str) -> Resp_ListAllProjectsForUser:
    page_limit = 32
    next_offset = 0
    lead_in = []
    member_of = []
    while True:
        try:
            prjs = api.search_projects(offset=next_offset, limit=page_limit, include_archived=True, team=dict(employee_id=user))

            for p in prjs.projects or []:
                detail = api.get_project(p.id).project
                role = [t for t in detail.team if t.employee == user][0].role

                if role == "Lead":
                    lead_in.append(detail)
                else:
                    member_of.append(detail)

            next_offset = prjs.next_offset
            if next_offset == -1:
                return Resp_ListAllProjectsForUser(lead_in=lead_in, member_of=member_of)
        except ApiException as e:
            if "page limit exceeded" in str(e):
                page_limit /= 2
                if page_limit <= 2:
                    raise
def list_my_customers(api: Erc3Client, user: str) -> Resp_ListAllCustomersForUser:
    page_limit = 32
    next_offset = 0
    loaded = []
    while True:
        try:
            custs = api.search_customers(offset=next_offset, limit=page_limit, account_managers=[user])

            for p in custs.companies or []:
                loaded.append(api.get_customer(p.id).company)

            next_offset = custs.next_offset
            if next_offset == -1:
                return Resp_ListAllCustomersForUser(customers=loaded)
        except ApiException as e:
            if "page limit exceeded" in str(e):
                page_limit /= 2
                if page_limit <= 2:
                    raise

# Tool do automatically distill wiki rules
def distill_rules(api: Erc3Client, llm: MyLLM, about: dev.Resp_WhoAmI) -> str:

    context_id = about.wiki_sha1

    loc = Path(f"context_{context_id}_v2.json")

    Category = Literal["applies_to_guests", "applies_to_users", "other"]

    class Rule(BaseModel):
        why_relevant_summary: str = Field(...)
        category: Category = Field(...)
        compact_rule: str

    class DistillWikiRules(BaseModel):
        company_name: str
        company_locations: List[str] = Field(..., description="list of locations where company operates")
        company_execs: List[str]
        rules: List[Rule]

    if  not loc.exists():
        print("New context discovered. Distilling rules once")
        schema = json.dumps(NextStep.model_json_schema())
        prompt = f"""
Carefully review the wiki below and identify most important security/scoping/data rules that will be highly relevant for the agent or user that are automating APIs of this company.

Pay attention to the rules that mention AI Agent or Public ChatBot. When talking about Public Chatbot use - applies_to_guests

Rules must be compact RFC-style, ok to use pseudo code for compactness. They will be used by an agent that operates following APIs: {schema}
""".strip()

        for path in api.list_wiki().paths:
            content = api.load_wiki(path)
            prompt += f"\n---- start of {path} ----\n\n{content}\n\n ---- end of {path} ----\n"


        messages = [{ "role": "system", "content": prompt}]

        distilled = llm.query(messages, DistillWikiRules)
        loc.write_text(distilled.model_dump_json(indent=2))

    else:
        distilled = DistillWikiRules.model_validate_json(loc.read_text())

    prompt = f"""You are AI Chatbot automating {distilled.company_name}.
    
Company locations: {distilled.company_locations}
Company execs: {distilled.company_execs}

Use available tools to execute task from the current user.

- To confirm project access - get or find project (and get after finding)
- Archival of entries or wiki deletion are not irreversible operations.
- Respond with proper Req_ProvideAgentResponse when:
    - Task is done
    - Task can't be completed (e.g. internal error, user is not allowed or clarification is needed)
- Make sure to always include links to relevant entities in response.

# Rules
"""
    relevant_categories: List[Category] = ["other"]
    if about.is_public:
        relevant_categories.append("applies_to_guests")
    else:
        relevant_categories.append("applies_to_users")

    for r in distilled.rules:
        if r.category in relevant_categories:
            prompt += f"\n- {r.compact_rule}"

    # append at the end to keep rules in context cache
    prompt += f"# Current context (trust it)\nDate:{about.today}"

    if about.is_public:
        prompt += "\nCurrent actor is GUEST (Anonymous user)"
    else:
        employee = api.get_employee(about.current_user).employee
        employee.skills = []
        employee.wills = []
        dump = employee.model_dump_json()
        prompt += f"\n# Current actor is authenticated user: {employee.name}:\n{dump}"

    return prompt


def my_dispatch(client: Erc3Client, cmd: BaseModel, about: dev.Resp_WhoAmI):
    # example how to add custom tools or tool handling
    if isinstance(cmd, dev.Req_UpdateEmployeeInfo):
        # first pull
        cur = client.get_employee(cmd.employee).employee

        cmd.notes = cmd.notes or cur.notes
        cmd.salary = cmd.salary or cur.salary
        cmd.wills = cmd.wills or cur.wills
        cmd.skills = cmd.skills or cur.skills
        cmd.location = cmd.location or cur.location
        cmd.department = cmd.department or cur.department
        return client.dispatch(cmd)

    if isinstance(cmd, Req_DeleteWikiPage):
        return client.dispatch(dev.Req_UpdateWiki(content="", changed_by=cmd.changed_by, file=cmd.file))

    if isinstance(cmd, Req_ListAllProjectsForUser):
        return list_my_projects(client, cmd.user)

    if isinstance(cmd, Req_ListAllCustomersForUser):
        return list_my_customers(client, cmd.user)

    if isinstance(cmd, dev.Req_ProvideAgentResponse):
        # drop link to current user
        cmd.links = [l for l in cmd.links if l.id != about.current_user]
        return client.dispatch(cmd)

    return client.dispatch(cmd)

def run_agent(model: str, api: ERC3, task: TaskInfo):
    erc_client = api.get_erc_client(task)
    about = erc_client.who_am_i()
    llm = MyLLM(api=api, model=model, task=task, max_tokens=32768)

    system_prompt = distill_rules(erc_client, llm, about)

    reason = Literal["security_violation", "request_not_supported_by_api", "more_information_needed", "may_pass"]

    class RequestPreflightCheck(BaseModel):
        current_actor: str
        preflight_check_explanation_brief: Optional[str]
        denial_reason: reason
        outcome_confidence_1_to_5: Annotated[int, Gt(0), Lt(6)]
        answer_requires_listing_actors_projects: bool

    # log will contain conversation context for the agent within task
    log = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Request: '{task.task_text}'"},
    ]

    preflight_check = llm.query(log, RequestPreflightCheck)
    confidence = preflight_check.outcome_confidence_1_to_5

    if confidence >=4:
        print(f"PREFLIGHT {confidence}: {preflight_check.preflight_check_explanation_brief}")
        if preflight_check.denial_reason == "request_not_supported_by_api":
            erc_client.provide_agent_response("Not supported", outcome="none_unsupported")
            return
        if preflight_check.denial_reason == "security_violation":
            erc_client.provide_agent_response("Security check failed", outcome="denied_security")
            return

    log.append({"role": "system", "content": preflight_check.preflight_check_explanation_brief})

    # let's limit number of reasoning steps by 20, just to be safe
    for i in range(20):
        step = f"step_{i + 1}"
        print(f"Next {step}... ", end="")

        job = llm.query(log, NextStep)

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
            result = my_dispatch(erc_client, job.function,about)
            txt = result.model_dump_json(exclude_none=True, exclude_unset=True)
            print(f"{CLI_GREEN}OUT{CLI_CLR}: {txt}")
            txt = "DONE: " + txt
        except ApiException as e:
            txt = e.detail
            # print to console as ascii red
            print(f"{CLI_RED}ERR: {e.api_error.error}{CLI_CLR}")

            txt = "ERROR: " + txt

            # if SGR wants to finish, then quit loop
        if isinstance(job.function, dev.Req_ProvideAgentResponse):
            print(f"{CLI_BLUE}agent {job.function.outcome}{CLI_CLR}. Summary:\n{job.function.message}")

            for link in job.function.links:
                print(f"  - link {link.kind}: {link.id}")
            break

        # and now we add results back to the convesation history, so that agent
        # we'll be able to act on the results in the next reasoning step.
        log.append({"role": "tool", "content": txt, "tool_call_id": step})