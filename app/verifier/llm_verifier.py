import json

from app.schemas.execution import ExecutionResult
from app.schemas.task_spec import TaskSpec
from app.schemas.verification import VerificationPackage, VerificationVerdict
from app.utils.llm_client import LLMClient


VERIFIER_SYSTEM_PROMPT = """
Ты независимый модуль верификации результата веб-автоматизации.

Верни строго JSON-объект (без markdown и без пояснений) вида:
{
  "task_completed": true/false,
  "confidence": 0.0,
  "verdict": "accept" | "reject" | "uncertain",
  "issues": ["..."],
  "summary": "..."
}

Правила:
1) confidence обязательно от 0 до 1.
2) verdict=accept только если required_fields заполнены и цель достигнута.
3) Если данных не хватает — verdict=uncertain или reject и пояснение в issues.
"""


class LLMVerifier:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def verify(self, plan: TaskSpec, result: ExecutionResult) -> VerificationVerdict:
        package = VerificationPackage(
            user_goal=plan.goal,
            expected_result_description=plan.expected_result.description,
            required_fields=plan.expected_result.required_fields,
            extracted_data=result.extracted_data,
            final_url=result.final_url,
            page_title=result.page_title,
            page_text_excerpt=result.page_text_excerpt,
            screenshot_path=result.screenshot_path,
            logs=[log.model_dump() for log in result.logs],
        )

        user_prompt = json.dumps(package.model_dump(), ensure_ascii=False, indent=2)
        raw_json = self.llm_client.generate_verifier_json(
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        return VerificationVerdict.model_validate(raw_json)
