from datetime import datetime, timezone
from typing import Any

from playwright.async_api import async_playwright

from app.config import SCREENSHOTS_DIR, VIDEOS_DIR
from app.executor.action_handlers import ActionHandlers
from app.schemas.execution import ExecutionResult, StepLog
from app.schemas.task_spec import TaskSpec

UTC = timezone.utc


class PlaywrightExecutor:
    def __init__(self, *, headless: bool = True, slow_mo: int = 0, record_video: bool = False):
        self.handlers = ActionHandlers()
        self.headless = headless
        self.slow_mo = slow_mo
        self.record_video = record_video

    async def _start_session(self) -> dict[str, Any]:
        p = await async_playwright().start()
        browser = await p.chromium.launch(headless=self.headless, slow_mo=self.slow_mo)
        context_kwargs = {}
        if self.record_video:
            context_kwargs["record_video_dir"] = str(VIDEOS_DIR)
        context = await browser.new_context(**context_kwargs)
        page = await context.new_page()
        return {
            "playwright": p,
            "browser": browser,
            "context": context,
            "page": page,
        }

    @staticmethod
    async def _close_session(session: dict[str, Any]) -> None:
        await session["context"].close()
        await session["browser"].close()
        await session["playwright"].stop()

    async def execute(
        self,
        plan: TaskSpec,
        *,
        session: dict[str, Any] | None = None,
        runtime_state: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        extracted_data = {}
        logs = []
        screenshot_path = None
        runtime_state = runtime_state if runtime_state is not None else {}
        owns_session = session is None

        if owns_session:
            try:
                session = await self._start_session()
            except Exception as launch_error:
                logs.append(
                    StepLog(
                        step_id=0,
                        action="launch_browser",
                        status="failed",
                        message=str(launch_error),
                    )
                )
                return ExecutionResult(
                    status="failed",
                    extracted_data=extracted_data,
                    final_url=None,
                    page_title=None,
                    page_text_excerpt=None,
                    screenshot_path=None,
                    logs=logs,
                    error_message=str(launch_error),
                )

        page = session["page"]

        try:
            for step in plan.steps:
                try:
                    if step.action == "finish":
                        logs.append(
                            StepLog(
                                step_id=step.step_id,
                                action=step.action,
                                status="success",
                                message="Workflow finished.",
                            )
                        )
                        break

                    if step.action == "screenshot" and "path" not in step.args:
                        step.args["path"] = str(SCREENSHOTS_DIR / f"step_{step.step_id}.png")

                    handler = getattr(self.handlers, step.action)
                    result = await handler(page, step.args, runtime_state)

                    if step.save_as:
                        extracted_data[step.save_as] = result

                    if step.action == "screenshot":
                        screenshot_path = result
                    if step.action == "observe_page" and isinstance(result, dict):
                        screenshot_path = result.get("screenshot_path") or screenshot_path

                    logs.append(
                        StepLog(
                            step_id=step.step_id,
                            action=step.action,
                            status="success",
                        )
                    )
                    debug_note = step.args.pop("_executor_note", None)
                    if debug_note:
                        logs.append(
                            StepLog(
                                step_id=step.step_id,
                                action=step.action,
                                status="success",
                                message=debug_note,
                            )
                        )
                except Exception as step_error:
                    logs.append(
                        StepLog(
                            step_id=step.step_id,
                            action=step.action,
                            status="failed",
                            message=str(step_error),
                        )
                    )
                    raise step_error

            page_title = await page.title()
            final_url = page.url
            text_excerpt = (await page.locator("body").inner_text())[:3000]

            if owns_session:
                await self._close_session(session)

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
            if screenshot_path is None:
                timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                emergency_path = SCREENSHOTS_DIR / f"emergency_{timestamp}.png"
                try:
                    await page.screenshot(path=str(emergency_path))
                    screenshot_path = str(emergency_path)
                    logs.append(
                        StepLog(
                            step_id=0,
                            action="emergency_screenshot",
                            status="success",
                            message=f"Saved failure screenshot to {screenshot_path}",
                        )
                    )
                except Exception as screenshot_error:
                    logs.append(
                        StepLog(
                            step_id=0,
                            action="emergency_screenshot",
                            status="failed",
                            message=str(screenshot_error),
                        )
                    )

            try:
                page_title = await page.title()
                final_url = page.url
                text_excerpt = (await page.locator("body").inner_text())[:3000]
            except Exception:
                page_title = None
                final_url = None
                text_excerpt = None

            if owns_session:
                await self._close_session(session)

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
