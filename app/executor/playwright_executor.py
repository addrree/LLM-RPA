from playwright.async_api import async_playwright

from app.config import SCREENSHOTS_DIR
from app.executor.action_handlers import ActionHandlers
from app.schemas.execution import ExecutionResult, StepLog
from app.schemas.task_spec import TaskSpec


class PlaywrightExecutor:
    def __init__(self):
        self.handlers = ActionHandlers()

    async def execute(self, plan: TaskSpec) -> ExecutionResult:
        extracted_data = {}
        logs = []
        screenshot_path = None

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                for step in plan.steps:
                    try:
                        if step.action == "finish":
                            logs.append(
                                StepLog(
                                    step_id=step.step_id,
                                    action=step.action,
                                    status="success",
                                    message="Workflow finished."
                                )
                            )
                            break

                        if step.action == "screenshot":
                            if "path" not in step.args:
                                step.args["path"] = str(
                                    SCREENSHOTS_DIR / f"step_{step.step_id}.png"
                                )

                        handler = getattr(self.handlers, step.action)
                        result = await handler(page, step.args)

                        if step.save_as:
                            extracted_data[step.save_as] = result

                        if step.action == "screenshot":
                            screenshot_path = result

                        logs.append(
                            StepLog(
                                step_id=step.step_id,
                                action=step.action,
                                status="success"
                            )
                        )
                    except Exception as step_error:
                        logs.append(
                            StepLog(
                                step_id=step.step_id,
                                action=step.action,
                                status="failed",
                                message=str(step_error)
                            )
                        )
                        raise step_error

                page_title = await page.title()
                final_url = page.url
                text_excerpt = (await page.locator("body").inner_text())[:3000]

                await browser.close()

                return ExecutionResult(
                    status="success",
                    extracted_data=extracted_data,
                    final_url=final_url,
                    page_title=page_title,
                    page_text_excerpt=text_excerpt,
                    screenshot_path=screenshot_path,
                    logs=logs,
                )

            except Exception as e:
                try:
                    page_title = await page.title()
                    final_url = page.url
                    text_excerpt = (await page.locator("body").inner_text())[:3000]
                except Exception:
                    page_title = None
                    final_url = None
                    text_excerpt = None

                await browser.close()

                return ExecutionResult(
                    status="failed",
                    extracted_data=extracted_data,
                    final_url=final_url,
                    page_title=page_title,
                    page_text_excerpt=text_excerpt,
                    screenshot_path=screenshot_path,
                    logs=logs,
                    error_message=str(e),
                )