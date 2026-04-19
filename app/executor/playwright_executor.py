from datetime import datetime, timezone
import asyncio
from typing import Any

from playwright.async_api import async_playwright

from app.config import SCREENSHOTS_DIR, VIDEOS_DIR
from app.executor.action_handlers import ActionHandlers, StructuredExtractionError
from app.schemas.execution import ExecutionResult, StepLog
from app.schemas.task_spec import TaskSpec

UTC = timezone.utc


class PlaywrightExecutor:
    BROWSER_RETRYABLE_ACTIONS = {"open_url", "click", "wait_for", "navigate_to_relevant_section"}

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
        tracing_started = False
        await context.tracing.start(screenshots=True, snapshots=True, sources=False)
        tracing_started = True
        page = await context.new_page()
        return {
            "playwright": p,
            "browser": browser,
            "context": context,
            "page": page,
            "tracing_started": tracing_started,
        }

    @staticmethod
    async def _stop_tracing_if_started(session: dict[str, Any], *, path: str | None = None) -> None:
        context = session.get("context")
        if context is None or not session.get("tracing_started", False):
            return
        if path:
            await context.tracing.stop(path=path)
        else:
            await context.tracing.stop()
        session["tracing_started"] = False

    @staticmethod
    async def _close_session(session: dict[str, Any]) -> None:
        try:
            await PlaywrightExecutor._stop_tracing_if_started(session)
        except Exception:
            pass
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
                    failure_type="browser_launch_error",
                    technical_failure=True,
                )

        page = session["page"]
        current_step = None

        try:
            for step in plan.steps:
                current_step = step
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

                    result = await self._run_step_with_browser_retries(
                        page=page,
                        session=session,
                        step=step,
                        runtime_state=runtime_state,
                        logs=logs,
                    )

                    if step.save_as:
                        extracted_data[step.save_as] = result
                        runtime_state["extracted_data"] = extracted_data

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
                    if isinstance(step_error, StructuredExtractionError):
                        diagnostic = {
                            "code": step_error.code,
                            "details": step_error.details,
                        }
                        if runtime_state is not None:
                            runtime_state.setdefault("structured_diagnostics", []).append(diagnostic)
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
                retry_artifacts=list(runtime_state.get("retry_artifacts", [])),
            )

        except Exception as e:
            if session and session.get("context") is not None:
                trace_path = VIDEOS_DIR / f"trace_failure_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.zip"
                try:
                    await self._stop_tracing_if_started(session, path=str(trace_path))
                    logs.append(
                        StepLog(
                            step_id=0,
                            action="trace_dump",
                            status="success",
                            message=f"Saved trace to {trace_path}",
                        )
                    )
                except Exception as trace_error:
                    logs.append(
                        StepLog(
                            step_id=0,
                            action="trace_dump",
                            status="failed",
                            message=str(trace_error),
                        )
                    )
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
                failure_type=self._classify_failure_type(str(e)),
                failed_action=current_step.action if current_step else (logs[-1].action if logs else None),
                failed_args=dict(current_step.args) if current_step else {},
                technical_failure=self._is_technical_failure(str(e)),
                retry_artifacts=list(runtime_state.get("retry_artifacts", [])),
            )

    async def _run_step_with_browser_retries(self, *, page, session, step, runtime_state, logs):
        handler = getattr(self.handlers, step.action)
        if step.action not in self.BROWSER_RETRYABLE_ACTIONS:
            return await handler(page, step.args, runtime_state)

        max_attempts = self._resolve_max_retry_attempts(step=step, runtime_state=runtime_state)
        base_delay = 0.35
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                if step.action in {"click", "wait_for"}:
                    await self._pre_retry_state_check(page=page, step=step)
                return await handler(page, step.args, runtime_state)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                await self._capture_retry_artifacts(
                    page=page,
                    step=step,
                    runtime_state=runtime_state,
                    attempt=attempt,
                    error=exc,
                )
                if not self._is_retryable_browser_error(str(exc)) or attempt >= max_attempts:
                    raise
                backoff_sec = base_delay * (2 ** (attempt - 1))
                logs.append(
                    StepLog(
                        step_id=step.step_id,
                        action=f"{step.action}_retry",
                        status="success",
                        message=f"Transient browser failure on attempt {attempt}, retrying in {backoff_sec:.2f}s: {exc}",
                    )
                )
                await asyncio.sleep(backoff_sec)
        raise last_error if last_error else RuntimeError("Unknown step execution error")

    @staticmethod
    def _resolve_max_retry_attempts(*, step, runtime_state) -> int:
        base = 2
        benchmark_context = runtime_state.get("benchmark_context", {}) if isinstance(runtime_state, dict) else {}
        task_family = str(benchmark_context.get("task_family", "")).strip()
        if step.action == "open_url":
            return 2
        if step.action == "wait_for":
            return 1 if task_family in {"single_value_extraction", "anchored_value_extraction"} else base
        if step.action == "click":
            return 2 if task_family in {"navigation_then_extraction", "multi_step_information_retrieval"} else 1
        if step.action == "navigate_to_relevant_section":
            return 2
        return base

    async def _pre_retry_state_check(self, *, page, step) -> None:
        if step.action == "wait_for" and step.args.get("selector"):
            await page.wait_for_selector(step.args["selector"], state="attached", timeout=3000)
            return
        if step.action == "click":
            await page.wait_for_timeout(50)

    async def _capture_retry_artifacts(self, *, page, step, runtime_state, attempt: int, error: Exception) -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        screenshot_path = SCREENSHOTS_DIR / f"retry_step{step.step_id}_attempt{attempt}_{timestamp}.png"
        diagnostics = {
            "step_id": step.step_id,
            "action": step.action,
            "attempt": attempt,
            "error": str(error),
            "args": dict(step.args),
            "selected_candidate": runtime_state.get("last_selected_candidate") if runtime_state else None,
            "locator_diagnostics": runtime_state.get("last_locator_diagnostics") if runtime_state else None,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        try:
            await page.screenshot(path=str(screenshot_path), full_page=True)
            diagnostics["screenshot_path"] = str(screenshot_path)
        except Exception as screenshot_error:  # noqa: BLE001
            diagnostics["screenshot_error"] = str(screenshot_error)

        if runtime_state is not None:
            runtime_state.setdefault("retry_artifacts", []).append(diagnostics)

    @staticmethod
    def _is_retryable_browser_error(message: str) -> bool:
        text = message.lower()
        return any(token in text for token in ["timeout", "net::", "navigation", "target closed", "detached"])

    @staticmethod
    def _is_technical_failure(message: str) -> bool:
        text = message.lower()
        return any(token in text for token in ["timeout", "net::", "navigation", "target closed", "detached", "browser"])

    @classmethod
    def _classify_failure_type(cls, message: str) -> str:
        lowered = message.lower()
        if "regex group reference is out of range" in lowered:
            return "regex_group_mismatch"
        if cls._is_technical_failure(message):
            return "browser_operation_failed"
        return "execution_step_failed"
