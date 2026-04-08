import json
import logging
from urllib.parse import urlparse

from app.planner.prompts import CORRECTIVE_REPLANNER_SYSTEM_PROMPT, REPLANNER_SYSTEM_PROMPT
from app.schemas.execution import LLMArtifact
from app.schemas.page_snapshot import PageSnapshot
from app.schemas.task_spec import TaskSpec
from app.utils.llm_client import LLMClient


class Replanner:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.last_artifact: LLMArtifact | None = None

    def revise_plan(
        self,
        user_goal: str,
        page_snapshot: PageSnapshot,
        previous_plan: TaskSpec | None = None,
        validation_error: str | None = None,
        invalid_plan: dict | None = None,
    ) -> TaskSpec:
        payload = {
            "user_goal": user_goal,
            "page_snapshot": page_snapshot.model_dump(mode="json"),
            "previous_plan": previous_plan.model_dump(mode="json") if previous_plan else None,
        }
        if validation_error:
            payload["repair_request"] = (
                f"Your previous plan was invalid. Validation error: {validation_error}. "
                "Return corrected JSON only."
            )
            payload["previous_invalid_plan"] = invalid_plan
        artifact = self.llm_client.generate_planner_artifact(
            system_prompt=REPLANNER_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            stage="replanner",
        )
        self.last_artifact = artifact
        normalized = self.normalize_final_plan(
            raw_plan=artifact.parsed_response,
            user_goal=user_goal,
            previous_plan=previous_plan,
            page_snapshot=page_snapshot,
        )
        return TaskSpec.model_validate(normalized)

    def build_corrective_plan(
        self,
        *,
        user_goal: str,
        page_snapshot: PageSnapshot,
        previous_plan: TaskSpec,
        execution_result: dict,
        verifier_verdict: dict,
    ) -> TaskSpec:
        payload = {
            "user_goal": user_goal,
            "page_snapshot": page_snapshot.model_dump(mode="json"),
            "previous_plan": previous_plan.model_dump(mode="json"),
            "execution_result": execution_result,
            "verifier_verdict": verifier_verdict,
        }
        artifact = self.llm_client.generate_planner_artifact(
            system_prompt=CORRECTIVE_REPLANNER_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            stage="corrective_replanner",
        )
        self.last_artifact = artifact
        normalized = self.normalize_final_plan(
            raw_plan=artifact.parsed_response,
            user_goal=user_goal,
            previous_plan=previous_plan,
            page_snapshot=page_snapshot,
        )
        return TaskSpec.model_validate(normalized)

    @staticmethod
    def normalize_final_plan(
        raw_plan: dict,
        user_goal: str,
        previous_plan: TaskSpec | None,
        page_snapshot: PageSnapshot,
    ) -> dict:
        plan = dict(raw_plan) if isinstance(raw_plan, dict) else {}
        context_start_url = (
            str(plan.get("start_url") or "")
            or (str(previous_plan.start_url) if previous_plan else "")
            or page_snapshot.url
            or "https://www.wikipedia.org"
        )

        steps = plan.get("steps")
        if not isinstance(steps, list):
            steps = []

        normalized_steps: list[dict] = []
        logger = logging.getLogger(__name__)
        for idx, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            current = dict(step)
            args = current.get("args")
            current["args"] = dict(args) if isinstance(args, dict) else {}

            if current.get("action") == "open_url" and not str(current["args"].get("url", "")).strip():
                logger.warning("Malformed open_url from model: missing args.url. Applying start_url normalization.")
                current["args"]["url"] = context_start_url

            current["step_id"] = idx
            normalized_steps.append(current)

        if not normalized_steps:
            normalized_steps = [
                {"step_id": 1, "action": "open_url", "args": {"url": context_start_url}},
                {"step_id": 2, "action": "finish", "args": {}},
            ]

        expected_result = plan.get("expected_result")
        if not isinstance(expected_result, dict):
            expected_result = {}

        if not isinstance(expected_result.get("description"), str) or not expected_result["description"].strip():
            if previous_plan and previous_plan.expected_result.description:
                expected_result["description"] = previous_plan.expected_result.description
            else:
                expected_result["description"] = f"Complete goal: {user_goal}"

        if not isinstance(expected_result.get("required_fields"), list):
            expected_result["required_fields"] = (
                list(previous_plan.expected_result.required_fields) if previous_plan else []
            )

        constraints = plan.get("constraints")
        if not isinstance(constraints, dict):
            constraints = {}
        if previous_plan:
            constraints = {
                "max_steps": constraints.get("max_steps", previous_plan.constraints.max_steps),
                "max_replans": constraints.get("max_replans", previous_plan.constraints.max_replans),
                "max_verification_retries": constraints.get(
                    "max_verification_retries",
                    previous_plan.constraints.max_verification_retries,
                ),
                "timeout_sec": constraints.get("timeout_sec", previous_plan.constraints.timeout_sec),
            }
        else:
            constraints = {
                "max_steps": constraints.get("max_steps", 10),
                "max_replans": constraints.get("max_replans", 1),
                "max_verification_retries": constraints.get("max_verification_retries", 1),
                "timeout_sec": constraints.get("timeout_sec", 30),
            }

        allowed_domains = plan.get("allowed_domains")
        if not isinstance(allowed_domains, list) or not allowed_domains:
            domain = urlparse(context_start_url).netloc
            if previous_plan and previous_plan.allowed_domains:
                allowed_domains = list(previous_plan.allowed_domains)
            else:
                allowed_domains = [domain] if domain else []

        return {
            "goal": plan.get("goal") or (previous_plan.goal if previous_plan else user_goal),
            "start_url": context_start_url,
            "allowed_domains": allowed_domains,
            "constraints": constraints,
            "expected_result": expected_result,
            "steps": normalized_steps,
        }
