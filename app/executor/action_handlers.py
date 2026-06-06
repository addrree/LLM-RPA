import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, urlparse

from app.observer.page_observer import PageObserver
from app.interaction.action_grounder import ActionGrounder
from app.interaction.page_candidates import PageCandidateExtractor
from app.planner.action_vocab import casefold_with_mojibake_repairs, is_broad_semantic_click_target


class StructuredExtractionError(ValueError):
    def __init__(self, *, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


RATE_LIMIT_MARKERS = (
    "rate exceeded",
    "too many requests",
    "rate limit",
    "http 429",
    "quota exceeded",
)

CAPTCHA_MARKERS = (
    "captcha",
    "robot check",
    "cloudflare",
    "confirm that you are not a robot",
    "verify you are human",
    "are you a human",
    "just a moment",
    "client challenge",
    "browser challenge",
    "challenge page",
    "checking your browser",
    "security check",
    "ddos-guard",
    "perimeterx",
    "datadome",
)

ACCESS_DENIED_MARKERS = (
    "access denied",
    "403 forbidden",
    "forbidden access",
    "request forbidden",
    "not authorized",
    "unauthorized",
    "permission denied",
)


@dataclass(frozen=True)
class BlockDetectionResult:
    blocked: bool
    failure_stage: str | None = None
    markers: tuple[str, ...] = ()


def detect_blocked_or_limited_text(text: str, *, status_codes: list[int | None] | None = None) -> BlockDetectionResult:
    folded = str(text or "").casefold()
    codes = {int(code) for code in status_codes or [] if code is not None}
    if 429 in codes:
        return BlockDetectionResult(True, "skipped_rate_limited", ("429",))
    if 401 in codes:
        return BlockDetectionResult(True, "skipped_http_401", ("401",))
    if 403 in codes:
        return BlockDetectionResult(True, "skipped_http_403", ("403",))

    rate_markers = tuple(marker for marker in RATE_LIMIT_MARKERS if marker.casefold() in folded)
    if rate_markers:
        return BlockDetectionResult(True, "skipped_rate_limited", rate_markers)

    captcha_markers = tuple(marker for marker in CAPTCHA_MARKERS if marker.casefold() in folded)
    if captcha_markers:
        return BlockDetectionResult(True, "skipped_captcha_or_antibot", captcha_markers)

    access_markers = tuple(marker for marker in ACCESS_DENIED_MARKERS if marker.casefold() in folded)
    if access_markers:
        return BlockDetectionResult(True, "skipped_access_denied", access_markers)

    return BlockDetectionResult(False)


class ActionHandlers:
    _CLICK_META_LABELS = {
        "scenario id",
        "scenario",
        "category",
        "notes",
        "required fields",
        "expected result",
        "should succeed",
        "benchmark",
        "expected",
    }

    def __init__(self):
        self.page_observer = PageObserver()
        self.candidate_extractor = PageCandidateExtractor()
        self.action_grounder = ActionGrounder()

    async def open_url(self, page, args, runtime_state=None):
        wait_until = str(args.get("wait_until", "domcontentloaded"))
        timeout_ms = int(args.get("timeout_ms", 20000))
        await page.goto(args["url"], wait_until=wait_until, timeout=timeout_ms)
        await self._raise_if_page_blocked_or_limited(page, runtime_state=runtime_state, stage="after_open_url")

    async def click(self, page, args, runtime_state=None):
        locator, candidate = await self._resolve_ranked_click_locator(page=page, args=args, runtime_state=runtime_state)
        await self._wait_for_actionable(locator)
        await locator.first.click(no_wait_after=True)
        await self._wait_after_possible_navigation(page)
        args["_executor_note"] = (
            f"click used ranked locator strategy={candidate.get('strategy')}; selector={candidate.get('selector')!r}; "
            f"candidates_checked={candidate.get('candidates_checked', 0)}"
        )
        self._mark_used_skill(runtime_state, "semantic_click")

    async def type(self, page, args, runtime_state=None):
        await self.fill(page, args, runtime_state)

    async def fill(self, page, args, runtime_state=None):
        resolved = await self._resolve_interaction_args(page, "fill", args, runtime_state)
        selector = resolved["selector"]
        text = str(resolved.get("text", resolved.get("value", "")))
        await page.locator(selector).first.fill(text)
        self._record_interaction_note(args, "fill", resolved, runtime_state)

    async def focus(self, page, args, runtime_state=None):
        resolved = await self._resolve_interaction_args(page, "focus", args, runtime_state)
        await page.locator(resolved["selector"]).first.focus()
        self._record_interaction_note(args, "focus", resolved, runtime_state)

    async def clear(self, page, args, runtime_state=None):
        resolved = await self._resolve_interaction_args(page, "clear", args, runtime_state)
        await page.locator(resolved["selector"]).first.clear()
        self._record_interaction_note(args, "clear", resolved, runtime_state)

    async def press(self, page, args, runtime_state=None):
        key = str(args.get("key", "Enter"))
        if str(args.get("selector", "")).strip():
            await page.locator(args["selector"]).first.press(key)
            return
        resolved = await self._resolve_interaction_args(page, "press", args, runtime_state) if self._has_interaction_target(args) else dict(args)
        if resolved.get("selector"):
            await page.locator(resolved["selector"]).first.press(key)
        else:
            await page.keyboard.press(key)
        self._record_interaction_note(args, "press", resolved, runtime_state)

    async def hover(self, page, args, runtime_state=None):
        resolved = await self._resolve_interaction_args(page, "hover", args, runtime_state)
        await page.locator(resolved["selector"]).first.hover()
        self._record_interaction_note(args, "hover", resolved, runtime_state)

    async def select_option(self, page, args, runtime_state=None):
        resolved = await self._resolve_interaction_args(page, "select_option", args, runtime_state)
        if resolved.get("action") == "click":
            await page.locator(resolved["selector"]).first.click()
        else:
            option = resolved.get("option_value", resolved.get("option_text", resolved.get("value")))
            try:
                await page.locator(resolved["selector"]).first.select_option(label=str(option))
            except Exception:
                await page.locator(resolved["selector"]).first.select_option(str(option))
        self._record_interaction_note(args, "select_option", resolved, runtime_state)

    async def check(self, page, args, runtime_state=None):
        resolved = await self._resolve_interaction_args(page, "check", args, runtime_state)
        await page.locator(resolved["selector"]).first.check()
        self._record_interaction_note(args, "check", resolved, runtime_state)

    async def uncheck(self, page, args, runtime_state=None):
        resolved = await self._resolve_interaction_args(page, "uncheck", args, runtime_state)
        await page.locator(resolved["selector"]).first.uncheck()
        self._record_interaction_note(args, "uncheck", resolved, runtime_state)

    async def select_autocomplete(self, page, args, runtime_state=None):
        result = await self._ground_interaction(page, "select_autocomplete", args, runtime_state)
        for action in result.actions:
            if action.action == "fill":
                await page.locator(action.args["selector"]).first.fill(str(action.args.get("text", "")))
            elif action.action == "click":
                await page.locator(action.args["selector"]).first.click()
        self._record_grounding(args, result, runtime_state)
        self._mark_used_skill(runtime_state, "select_list_autocomplete")

    async def choose_date(self, page, args, runtime_state=None):
        resolved = await self._resolve_interaction_args(page, "choose_date", args, runtime_state)
        await page.locator(resolved["selector"]).first.fill(str(resolved.get("date", resolved.get("value", resolved.get("text", "")))))
        self._record_interaction_note(args, "choose_date", resolved, runtime_state)

    async def click_by_semantic_target(self, page, args, runtime_state=None):
        request = {
            "target_text": args.get("target_text") or args.get("text") or args.get("target"),
            "role": args.get("role"),
            "exact": args.get("exact", True),
            "scope": args.get("scope"),
            "allow_mouse_fallback": args.get("allow_mouse_fallback", False),
        }
        target_candidates = args.get("target_candidates")
        if not isinstance(target_candidates, list) or not target_candidates:
            target_candidates = [request.get("target_text")]
        first_result_navigation_request = self._looks_like_first_result_navigation_request(request)
        if first_result_navigation_request:
            opened_result = await self._click_first_result_like_link(page, args, runtime_state)
            if opened_result:
                self._mark_used_skill(runtime_state, "semantic_click")
                args["_executor_note"] = (
                    f"click_by_semantic_target target={request.get('target_text')!r}; "
                    "strategy=first_result_link_fallback"
                )
                return opened_result
            raise StructuredExtractionError(
                code="target_link_not_found",
                message=f"first result link not found for target: {request.get('target_text')!r}",
                details={"target_text": request.get("target_text"), "reason": "no_result_like_link_found"},
            )
        if self._should_prefer_search_result_target(page=page, request=request):
            opened_result = await self._click_search_result_matching_target(page, args, runtime_state)
            if opened_result:
                self._mark_used_skill(runtime_state, "semantic_click")
                args["_executor_note"] = (
                    f"click_by_semantic_target target={request.get('target_text')!r}; "
                    "strategy=search_result_target_fallback"
                )
                return opened_result
        if is_broad_semantic_click_target(request.get("target_text")):
            raise StructuredExtractionError(
                code="target_too_broad",
                message=f"semantic click target is too broad: {request.get('target_text')!r}",
                details={"target_text": request.get("target_text"), "role": request.get("role")},
            )
        result = None
        last_error = None
        fallback_clicked_text = None
        for target_text in [str(item).strip() for item in target_candidates if str(item).strip()]:
            candidate_request = dict(request)
            candidate_request["target_text"] = target_text
            try:
                result = await self._ground_interaction(page, "click", candidate_request, runtime_state)
                request = candidate_request
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                fallback_clicked_text = await self._click_semantic_text_fallback(
                    page,
                    target_text=target_text,
                    role=str(candidate_request.get("role") or ""),
                )
                if fallback_clicked_text:
                    request = candidate_request
                    break
        if result is None:
            if fallback_clicked_text:
                self._mark_used_skill(runtime_state, "semantic_click")
                args["_executor_note"] = (
                    f"click_by_semantic_target target={request.get('target_text')!r}; "
                    "strategy=semantic_text_locator_fallback"
                )
                return fallback_clicked_text
            if await self._page_has_antibot_challenge(page):
                raise StructuredExtractionError(
                    code="captcha_or_antibot",
                    message="captcha or anti-bot challenge detected after page interaction",
                    details={"target_text": request.get("target_text")},
                )
            if last_error is not None:
                raise StructuredExtractionError(
                    code="target_link_not_found",
                    message=f"semantic click target not found: {request.get('target_text')!r}",
                    details={
                        "target_text": request.get("target_text"),
                        "target_candidates": target_candidates,
                        "last_error": str(last_error),
                    },
                ) from last_error
            result = await self._ground_interaction(page, "click", request, runtime_state)
        for action in result.actions:
            try:
                await page.locator(action.args["selector"]).first.click(no_wait_after=True)
                await self._wait_after_possible_navigation(page)
            except Exception:
                if await self._page_has_antibot_challenge(page):
                    raise StructuredExtractionError(
                        code="captcha_or_antibot",
                        message="captcha or anti-bot challenge detected after page interaction",
                        details={"target_text": request.get("target_text")},
                    )
                raise
        self._mark_used_skill(runtime_state, "semantic_click")
        selected = result.selected_candidate or {}
        clicked_text = (
            selected.get("text")
            or selected.get("name")
            or selected.get("aria_label")
            or selected.get("ariaLabel")
            or request.get("target_text")
        )
        args["_executor_note"] = (
            f"click_by_semantic_target target={request.get('target_text')!r}; "
            f"strategy={result.grounding_strategy}; candidate_id={result.selected_candidate and result.selected_candidate.get('candidate_id')}"
        )
        return str(clicked_text or "").strip()

    async def _click_semantic_text_fallback(self, page, *, target_text: str, role: str = "") -> str | None:
        target = str(target_text or "").strip()
        if not target:
            return None
        pattern = re.compile(re.escape(target), flags=re.IGNORECASE)
        role_order = [role.strip().lower()] if role.strip() else []
        role_order.extend([item for item in ("link", "button") if item not in role_order])
        for role_name in role_order:
            try:
                locator = page.get_by_role(role_name, name=pattern).first
                if await locator.count():
                    await locator.click(no_wait_after=True)
                    await self._wait_after_possible_navigation(page)
                    try:
                        label = (await locator.inner_text(timeout=1000)).strip()
                    except Exception:
                        label = target
                    return label or target
            except Exception:
                continue
        return None

    @staticmethod
    def _looks_like_first_result_navigation_request(request: dict[str, Any]) -> bool:
        text = " ".join(str(request.get(key, "") or "") for key in ("target_text", "scope")).casefold()
        if not any(token in text for token in ("result", "results", "item", "link", "результат", "ссылка")):
            return False
        return any(token in text for token in ("first", "top", "best", "relevant", "1st", "перв", "релевант"))

    @classmethod
    def _should_prefer_search_result_target(cls, *, page, request: dict[str, Any]) -> bool:
        target = str(request.get("target_text") or "").strip()
        if not target or len(target) < 2:
            return False
        if cls._looks_like_search_ui_link(title=target, href=""):
            return False
        role = str(request.get("role") or "").strip().casefold()
        if role and role not in {"link", "button", "item", "result"}:
            return False
        url = str(getattr(page, "url", "") or "")
        title = ""
        try:
            title_value = getattr(page, "title", None)
            if isinstance(title_value, str):
                title = title_value
        except Exception:
            title = ""
        return cls._looks_like_search_results_context(url=url, title=title)

    @staticmethod
    def _looks_like_search_results_context(*, url: str, title: str = "") -> bool:
        folded_title = str(title or "").casefold()
        if any(token in folded_title for token in ("search results", "results for", "результаты поиска")):
            return True
        try:
            parsed = urlparse(str(url or ""))
        except Exception:
            return False
        path = parsed.path.casefold()
        query = parse_qs(parsed.query)
        if any(key.casefold() in {"q", "query", "search", "s", "keyword", "keywords", "term"} for key in query):
            return True
        return any(part in path for part in ("/search", "/find", "/results"))

    async def _click_search_result_matching_target(self, page, args: dict, runtime_state=None) -> str:
        target = str(args.get("target_text") or args.get("text") or args.get("target") or "").strip()
        if not target:
            return ""
        delegated = {
            "intent": "search_results",
            "output_key": "search_results",
            "limit": int(args.get("limit", 10) or 10),
        }
        items = await self._collect_search_results_generic(page=page, args=delegated, runtime_state=runtime_state)
        ranked: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            current = dict(item)
            current["_target_score"] = self._search_result_target_score(current, target=target)
            current["_target_rank"] = index
            ranked.append(current)
        ranked.sort(key=lambda item: (-float(item.get("_target_score", 0)), int(item.get("_target_rank", 9999))))
        for item in ranked:
            if float(item.get("_target_score", 0)) <= 0:
                continue
            href = str(item.get("href") or item.get("link") or "").strip()
            selector = str(item.get("selector") or "").strip()
            if href:
                await page.goto(href, wait_until="domcontentloaded", timeout=int(args.get("navigation_timeout_ms", 20000)))
                await self._wait_after_possible_navigation(page)
                await self._raise_if_page_blocked_or_limited(page, runtime_state=runtime_state, stage="after_search_result_target_navigation")
                if runtime_state is not None:
                    runtime_state["last_opened_result"] = item
                return str(item.get("title") or item.get("text") or href).strip()
            if selector:
                await page.locator(selector).first.click(no_wait_after=True)
                await self._wait_after_possible_navigation(page)
                await self._raise_if_page_blocked_or_limited(page, runtime_state=runtime_state, stage="after_search_result_target_click")
                if runtime_state is not None:
                    runtime_state["last_opened_result"] = item
                return str(item.get("title") or item.get("text") or selector).strip()
        return ""

    @classmethod
    def _search_result_target_score(cls, item: dict[str, Any], *, target: str) -> float:
        target_norm = cls._normalize_result_match_text(target)
        if not target_norm:
            return 0.0
        title = str(item.get("title") or item.get("text") or "")
        href = str(item.get("href") or item.get("link") or "")
        description = str(item.get("description") or item.get("snippet") or "")
        title_norm = cls._normalize_result_match_text(title)
        haystack = cls._normalize_result_match_text(f"{title} {description} {href}")
        score = float(item.get("score", 0) or 0)
        if title_norm == target_norm:
            score += 12.0
        elif title_norm.startswith(f"{target_norm} "):
            score += 7.0
        elif target_norm in title_norm:
            score += 4.0
        elif target_norm in haystack:
            score += 2.0
        else:
            return 0.0
        if re.search(rf"\b{re.escape(target_norm)}\s+(?:api|guide|overview|documentation|docs|reference)\b", title_norm):
            score += 3.0
        if target_norm and target_norm in cls._normalized_href_segments(href):
            score += 4.0
        simple_target = not re.search(r"[\s().:#/]", target_norm)
        if simple_target:
            if re.search(r"\(\)|\b(?:constructor|method|property|attribute|event|function|syntax|parameters?)\b", title_norm):
                score -= 6.0
            score -= min(max(len([part for part in urlparse(href).path.split("/") if part]) - 3, 0), 8) * 0.5
        return score

    @staticmethod
    def _normalized_href_segments(href: str) -> set[str]:
        try:
            path = urlparse(str(href or "")).path
        except Exception:
            path = str(href or "")
        segments = set()
        for part in path.split("/"):
            cleaned = re.sub(r"[-_]+", " ", part).strip()
            if cleaned:
                segments.add(re.sub(r"\s+", " ", cleaned.casefold()))
        return segments

    async def _click_first_result_like_link(self, page, args: dict, runtime_state=None) -> str:
        delegated = {
            "intent": "search_results",
            "output_key": "search_results",
            "limit": int(args.get("limit", 5) or 5),
        }
        items = []
        if isinstance(runtime_state, dict) and isinstance(runtime_state.get("last_search_results"), list):
            items = [item for item in runtime_state.get("last_search_results", []) if isinstance(item, dict)]
        if not items:
            items = await self._collect_search_results_generic(page=page, args=delegated, runtime_state=runtime_state)
        for item in items:
            if not isinstance(item, dict):
                continue
            href = str(item.get("href") or item.get("link") or "").strip()
            selector = str(item.get("selector") or "").strip()
            if href:
                await page.goto(href, wait_until="domcontentloaded", timeout=int(args.get("navigation_timeout_ms", 20000)))
                await self._wait_after_possible_navigation(page)
                await self._raise_if_page_blocked_or_limited(page, runtime_state=runtime_state, stage="after_first_result_navigation")
                if runtime_state is not None:
                    runtime_state["last_opened_result"] = item
                return str(item.get("title") or item.get("text") or href).strip()
            if selector:
                await page.locator(selector).first.click(no_wait_after=True)
                await self._wait_after_possible_navigation(page)
                await self._raise_if_page_blocked_or_limited(page, runtime_state=runtime_state, stage="after_first_result_click")
                if runtime_state is not None:
                    runtime_state["last_opened_result"] = item
                return str(item.get("title") or item.get("text") or selector).strip()
        return ""

    @staticmethod
    async def _page_has_antibot_challenge(page) -> bool:
        try:
            title = await page.title()
        except Exception:
            title = ""
        try:
            body_text = await page.locator("body").inner_text(timeout=1500)
        except Exception:
            body_text = ""
        haystack = f"{title}\n{body_text}".casefold()
        markers = [
            "captcha",
            "client challenge",
            "robot check",
            "cloudflare",
            "access denied",
            "verify you are human",
            "подтвердите, что вы не робот",
            "проверка безопасности",
            "доступ ограничен",
        ]
        return any(marker in haystack for marker in markers)

    @staticmethod
    async def _raise_if_page_blocked_or_limited(page, *, runtime_state=None, stage: str) -> None:
        try:
            title = await page.title()
        except Exception:
            title = ""
        try:
            body_text = await page.locator("body").inner_text(timeout=1200)
        except Exception:
            body_text = ""
        detection = detect_blocked_or_limited_text(f"{title}\n{body_text}")
        if not detection.blocked:
            return
        diagnostics = {
            "stage": stage,
            "failure_stage": detection.failure_stage,
            "markers": list(detection.markers),
        }
        if runtime_state is not None:
            runtime_state["blocked_or_limited_status"] = diagnostics
            runtime_state["rate_limit_detected"] = detection.failure_stage == "skipped_rate_limited"
        raise StructuredExtractionError(
            code=detection.failure_stage or "skipped_access_denied",
            message=f"blocked_or_limited_page_detected: {detection.failure_stage}",
            details=diagnostics,
        )

    @staticmethod
    async def _wait_after_possible_navigation(page) -> None:
        try:
            await page.wait_for_timeout(700)
        except Exception:
            pass
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            return
        try:
            await page.wait_for_load_state("networkidle", timeout=2500)
        except Exception:
            return

    @staticmethod
    async def _settle_page_for_read(page) -> None:
        for state, timeout in (("domcontentloaded", 5000), ("load", 2500), ("networkidle", 1200)):
            try:
                await page.wait_for_load_state(state, timeout=timeout)
            except Exception:
                pass
        try:
            await page.wait_for_timeout(100)
        except Exception:
            pass

    async def fill_by_semantic_target(self, page, args, runtime_state=None):
        request = {
            "target_text": args.get("target_text") or args.get("field_hint") or args.get("target") or args.get("role"),
            "label": args.get("field_hint"),
            "role": args.get("role"),
            "value": args.get("value", args.get("text", "")),
        }
        result = await self._ground_interaction(page, "fill", request, runtime_state)
        for action in result.actions:
            await page.locator(action.args["selector"]).first.fill(str(action.args.get("text", action.args.get("value", request["value"]))))
        self._mark_used_skill(runtime_state, "semantic_fill")
        args["_executor_note"] = (
            f"fill_by_semantic_target field_hint={args.get('field_hint')!r}; "
            f"strategy={result.grounding_strategy}; candidate_id={result.selected_candidate and result.selected_candidate.get('candidate_id')}"
        )

    async def select_by_semantic_target(self, page, args, runtime_state=None):
        request = {
            "target_text": args.get("control_hint") or args.get("target_text") or args.get("target"),
            "label": args.get("control_hint"),
            "option_text": args.get("option_text", args.get("value", "")),
            "exact": args.get("exact", True),
        }
        result = await self._ground_interaction(page, "select_option", request, runtime_state)
        for action in result.actions:
            if action.action == "click":
                await page.locator(action.args["selector"]).first.click()
            else:
                option = action.args.get("option_text", request["option_text"])
                try:
                    await page.locator(action.args["selector"]).first.select_option(label=str(option))
                except Exception:
                    await page.locator(action.args["selector"]).first.select_option(str(option))
        self._mark_used_skill(runtime_state, "select_list_autocomplete")
        args["_executor_note"] = (
            f"select_by_semantic_target option={request.get('option_text')!r}; "
            f"strategy={result.grounding_strategy}; candidate_id={result.selected_candidate and result.selected_candidate.get('candidate_id')}"
        )

    async def choose_autocomplete_suggestion(self, page, args, runtime_state=None):
        await self.select_autocomplete(page, args, runtime_state)
        self._mark_used_skill(runtime_state, "select_list_autocomplete")

    async def wait_for(self, page, args, runtime_state=None):
        timeout_ms = int(args.get("timeout_ms", 12000))
        if "selector" in args and str(args.get("selector", "")).strip():
            selector = str(args["selector"])
            try:
                await page.wait_for_selector(selector, state="visible", timeout=timeout_ms)
            except Exception:
                if bool(args.get("strict", False)):
                    raise
                args["_executor_note"] = (
                    f"wait_for selector={selector!r} timed out; continued because selector waits are advisory "
                    "unless strict=true"
                )
            return
        if "url_contains" in args and str(args.get("url_contains", "")).strip():
            await page.wait_for_url(f"**{args['url_contains']}**", timeout=timeout_ms)
            return
        if "text" in args and str(args.get("text", "")).strip():
            scope_selector = str(args.get("scope_selector", "")).strip()
            scope = page.locator(scope_selector) if scope_selector else page
            locator = scope.get_by_text(str(args["text"]), exact=bool(args.get("exact", False)))
            state = "visible" if bool(args.get("visible_only", True)) else "attached"
            await locator.first.wait_for(state=state, timeout=timeout_ms)
            return
        raise ValueError("wait_for requires one of selector | url_contains | text (optionally scoped)")

    async def navigate_to_relevant_section(self, page, args, runtime_state=None):
        await self.click(page, args, runtime_state)
        wait_args = args.get("wait_for") if isinstance(args.get("wait_for"), dict) else {}
        if wait_args:
            await self.wait_for(page, wait_args, runtime_state)
        elif args.get("post_click_wait_selector"):
            await self.wait_for(page, {"selector": args["post_click_wait_selector"]}, runtime_state)
        elif args.get("post_click_wait_ms"):
            await asyncio.sleep(float(args["post_click_wait_ms"]) / 1000.0)
        return {
            "navigated": True,
            "target": {
                "text": args.get("text"),
                "name": args.get("name"),
                "href_contains": args.get("href_contains"),
                "selector": args.get("selector"),
            },
            "final_url": page.url,
        }

    @staticmethod
    def _has_interaction_target(args: dict) -> bool:
        return any(str(args.get(key, "")).strip() for key in ("selector", "target", "target_text", "label", "candidate_id", "text", "name", "placeholder"))

    async def _ground_interaction(self, page, intent: str, args: dict, runtime_state=None):
        candidates = await self.candidate_extractor.extract(page)
        if runtime_state is not None:
            runtime_state["last_page_candidates"] = PageCandidateExtractor.compact(candidates)
        request = dict(args)
        request.setdefault("intent", intent)
        result = self.action_grounder.ground(
            request,
            candidates,
            user_goal=str((runtime_state or {}).get("user_goal", "")),
            page_snapshot=(runtime_state or {}).get("last_page_snapshot"),
        )
        self._record_grounding(args, result, runtime_state)
        return result

    async def _resolve_interaction_args(self, page, intent: str, args: dict, runtime_state=None) -> dict:
        if str(args.get("selector", "")).strip():
            resolved = dict(args)
            resolved["selector"] = str(args["selector"]).strip()
            return resolved
        result = await self._ground_interaction(page, intent, args, runtime_state)
        action = result.actions[0]
        resolved = dict(args)
        resolved.update(action.args)
        resolved["action"] = action.action
        return resolved

    @staticmethod
    def _record_grounding(args: dict, result, runtime_state=None) -> None:
        diagnostics = {
            "selected_candidate": result.selected_candidate,
            "grounding_strategy": result.grounding_strategy,
            "confidence": result.confidence,
            "rejected_candidates": result.rejected_candidates[:20],
        }
        args["_grounding"] = diagnostics
        if runtime_state is not None:
            runtime_state["last_selected_candidate"] = result.selected_candidate
            runtime_state["last_grounding"] = diagnostics

    @staticmethod
    def _mark_used_skill(runtime_state, skill_name: str) -> None:
        if runtime_state is None:
            return
        skills = runtime_state.setdefault("used_skills", [])
        if skill_name not in skills:
            skills.append(skill_name)

    def _record_interaction_note(self, args: dict, action: str, resolved: dict, runtime_state=None) -> None:
        args["_executor_note"] = f"{action} used selector={resolved.get('selector')!r}; candidate_id={resolved.get('candidate_id')!r}"
        skill = {
            "click": "semantic_click",
            "hover": "semantic_click",
            "focus": "semantic_fill",
            "fill": "semantic_fill",
            "type": "semantic_fill",
            "select_option": "select_list_autocomplete",
            "select_autocomplete": "select_list_autocomplete",
            "choose_date": "select_list_autocomplete",
            "check": "semantic_click",
            "uncheck": "semantic_click",
        }.get(action)
        if skill:
            self._mark_used_skill(runtime_state, skill)

    async def extract_value_from_section(self, page, args, runtime_state=None):
        section_selector = str(args.get("section_selector", "")).strip()
        if not section_selector:
            raise ValueError("extract_value_from_section requires non-empty 'section_selector'")
        section = page.locator(section_selector).first
        await section.wait_for(state="visible", timeout=int(args.get("timeout_ms", 15000)))

        field_selector = str(args.get("field_selector", "")).strip()
        pattern = str(args.get("pattern", "")).strip()
        if field_selector:
            target = section.locator(field_selector).first
            await target.wait_for(state="visible", timeout=int(args.get("timeout_ms", 15000)))
            text = (await target.inner_text()).strip()
        else:
            text = (await section.inner_text()).strip()

        if pattern:
            match = re.search(pattern, text, flags=re.IGNORECASE if bool(args.get("ignore_case", True)) else 0)
            if not match:
                raise ValueError(f"Pattern not found in section: {pattern}")
            value = self._extract_match_value(match, group_index=args.get("group_index"))
        else:
            value = text

        if bool(args.get("normalize_number", False)):
            value = self._normalize_number_token(
                value,
                number_type=str(args.get("number_type", "int")).lower(),
                strip_plus=bool(args.get("strip_plus", True)),
            )
        return value

    async def extract_structured_items_from_region(self, page, args, runtime_state=None):
        self._mark_used_skill(runtime_state, "row_list_extraction")
        region_selector = str(args.get("region_selector", "")).strip()
        if not region_selector:
            raise ValueError("extract_structured_items_from_region requires non-empty 'region_selector'")
        container_selector = str(args.get("container_selector", "")).strip()
        if not container_selector:
            raise ValueError("extract_structured_items_from_region requires non-empty 'container_selector'")

        delegated = dict(args)
        delegated["container_selector"] = f"{region_selector} {container_selector}"
        items = await self.extract_items(page, delegated, runtime_state)
        args["_executor_note"] = (
            f"extract_structured_items_from_region via region={region_selector!r}; "
            f"container={container_selector!r}; extracted={len(items)}"
        )
        return items

    async def compare_structured_values(self, page, args, runtime_state=None):
        runtime = runtime_state if runtime_state is not None else {}
        extracted = runtime.get("extracted_data", {})
        left_key = str(args.get("left_key", "section_a_data"))
        right_key = str(args.get("right_key", "section_b_data"))
        left = extracted.get(left_key)
        right = extracted.get(right_key)
        if left is None or right is None:
            raise ValueError(
                f"compare_structured_values requires data for '{left_key}' and '{right_key}' in extracted_data"
            )
        comparison = self._build_structured_comparison(left=left, right=right, label_left=left_key, label_right=right_key)
        extracted["structured_comparison"] = comparison
        extracted["comparison"] = comparison
        extracted["compare_status"] = comparison.get("status")
        extracted["comparison_left_summary"] = self._build_comparison_side_summary(value=left, label=left_key)
        extracted["comparison_right_summary"] = self._build_comparison_side_summary(value=right, label=right_key)
        runtime["extracted_data"] = extracted
        args["_executor_note"] = (
            f"compare_structured_values compared {left_key!r} vs {right_key!r}; "
            f"status={comparison.get('status')}; exact_match={comparison.get('exact_match')}"
        )
        return comparison

    async def assert_page_contains(self, page, args, runtime_state=None):
        text = str(args.get("text", "")).strip()
        pattern = str(args.get("pattern", "")).strip()
        selector = str(args.get("selector", "")).strip()
        if selector:
            count = await page.locator(selector).count()
            if count < 1:
                raise ValueError(f"assert_page_contains failed: selector not found {selector!r}")
            return {"assertion": "selector", "matched": count}
        body_text = await self._load_source_text(page=page, runtime_state=runtime_state)
        if text:
            matched = text.lower() in body_text.lower() if bool(args.get("ignore_case", True)) else text in body_text
            if not matched:
                raise ValueError(f"assert_page_contains failed: text not found {text!r}")
            return {"assertion": "text", "matched": True}
        if pattern:
            match = re.search(pattern, body_text, flags=re.IGNORECASE if bool(args.get("ignore_case", True)) else 0)
            if not match:
                raise ValueError(f"assert_page_contains failed: pattern not found {pattern!r}")
            return {"assertion": "pattern", "matched": True}
        raise ValueError("assert_page_contains requires selector | text | pattern")

    async def extract_text(self, page, args, runtime_state=None):
        self._mark_used_skill(runtime_state, "generic_text_extraction")
        selector = args["selector"]
        locator = page.locator(selector)
        match_count = await locator.count()

        index = args.get("index")
        if index is not None:
            target_locator = locator.nth(index)
            args["_executor_note"] = (
                f'Selector "{selector}" matched {match_count} elements; used index={index}.'
            )
        elif self._is_heading_selector(selector):
            extracted_value, note = await self._extract_text_with_heading_fallback(
                page=page,
                selector=selector,
                fallback_locator=locator,
            )
            args["_executor_note"] = note
            return extracted_value
        else:
            target_locator = locator.first
            if match_count > 1:
                args["_executor_note"] = (
                    f'Selector "{selector}" matched {match_count} elements; used first element.'
                )

        return (await target_locator.inner_text()).strip()

    async def extract_html(self, page, args, runtime_state=None):
        self._mark_used_skill(runtime_state, "generic_text_extraction")
        selector = args["selector"]
        locator = page.locator(selector)
        match_count = await locator.count()

        index = args.get("index")
        if index is not None:
            target_locator = locator.nth(index)
            args["_executor_note"] = (
                f'Selector "{selector}" matched {match_count} elements; used index={index}.'
            )
        else:
            target_locator = locator.first
            if match_count > 1:
                args["_executor_note"] = (
                    f'Selector "{selector}" matched {match_count} elements; used first element.'
                )

        return (await target_locator.inner_html()).strip()

    async def extract_items(self, page, args, runtime_state=None):
        self._mark_used_skill(runtime_state, "row_list_extraction")
        await self._raise_if_page_blocked_or_limited(page, runtime_state=runtime_state, stage="before_extract_items")
        container_selector = args["container_selector"]
        limit = int(args.get("limit", 20))
        fields = args.get("fields") or {}

        containers = page.locator(container_selector)
        count = min(await containers.count(), limit)
        items = []

        for idx in range(count):
            container = containers.nth(idx)
            item = {}
            for field_name, rule in fields.items():
                item[field_name] = await self._extract_field_value(container, field_name, rule)
            items.append(item)

        args["_executor_note"] = f'Extracted {len(items)} item(s) via "{container_selector}" (limit={limit}).'
        return items

    async def extract_structured_items(self, page, args, runtime_state=None):
        self._mark_used_skill(runtime_state, "row_list_extraction")
        await self._raise_if_page_blocked_or_limited(page, runtime_state=runtime_state, stage="before_extract_structured_items")
        limit = int(args.get("limit", 20))
        fields = args.get("fields") or {}
        pattern = str(args.get("pattern", "") or "").strip()
        flags = args.get("flags")
        benchmark_context = runtime_state.get("benchmark_context", {}) if isinstance(runtime_state, dict) else {}
        task_family = str(benchmark_context.get("task_family", "")).strip().lower()

        if not pattern:
            intent = str(args.get("intent", "") or args.get("item_type", "") or "").strip()
            if intent:
                delegated = dict(args)
                delegated["intent"] = intent
                return await self.extract_by_intent(page, delegated, runtime_state)
            table_rows = await self._extract_table_rows(page=page, limit=limit)
            if table_rows:
                projected = self._project_structured_rows_to_fields(rows=table_rows, fields=fields, limit=limit)
                args["_executor_note"] = (
                    "extract_structured_items fallback=table_rows_no_pattern; "
                    f"returned {len(projected)} row(s)"
                )
                return projected
            list_rows = await self._extract_repeated_link_or_list_items(page=page, limit=limit)
            if list_rows:
                projected = self._project_structured_rows_to_fields(rows=list_rows, fields=fields, limit=limit)
                args["_executor_note"] = (
                    "extract_structured_items fallback=list_items_no_pattern; "
                    f"returned {len(projected)} row(s)"
                )
                return projected
            raise StructuredExtractionError(
                code="structured_extraction_no_pattern_no_fallback",
                message="extract_structured_items requires a pattern or an available generic DOM/table/list fallback.",
                details={"intent": intent, "fields": fields},
            )

        if task_family == "repeated_structured_items" and self._is_overly_broad_repeated_pattern(pattern):
            projected, note = await self._extract_repeated_structured_with_quality_fallbacks(
                page=page,
                fields=fields,
                limit=limit,
                context_note=f"broad_pattern_rejected={pattern!r}",
            )
            if projected:
                args["_executor_note"] = note
                return projected
            raise StructuredExtractionError(
                code="broad_pattern_rejected_no_structured_fallback",
                message=(
                    "broad_pattern_rejected_no_structured_fallback: Regex pattern is too broad for repeated structured extraction and no high-quality DOM fallback was found."
                ),
                details={"pattern": pattern, "task_family": task_family, "fallback_note": note},
            )

        delegated_args = {
            "pattern": pattern,
            "limit": limit,
            "fields": fields,
            "occurrence": 1,
        }
        if flags is not None:
            delegated_args["flags"] = flags

        try:
            items = await self.extract_pattern_from_page_text(page, delegated_args, runtime_state)
            args["_executor_note"] = (
                f"extract_structured_items matched pattern={pattern!r}; returned {len(items)} repeated item(s)"
            )
            return items
        except Exception as pattern_error:  # noqa: BLE001
            if task_family == "repeated_structured_items":
                projected, note = await self._extract_repeated_structured_with_quality_fallbacks(
                    page=page,
                    fields=fields,
                    limit=limit,
                    context_note=f"regex_error={pattern_error}",
                )
                if projected:
                    args["_executor_note"] = note
                    return projected
                raise StructuredExtractionError(
                    code="repeated_structured_extraction_low_quality_fallback",
                    message=(
                        "Regex extraction failed and repeated-structured DOM fallbacks were low-quality or absent."
                    ),
                    details={"pattern": pattern, "task_family": task_family, "fallback_note": note},
                ) from pattern_error

            row_candidates = await self._collect_row_candidates_generic(page=page, limit=max(limit * 20, 120))
            matched_rows = self._filter_rows_by_pattern_literals(rows=row_candidates, pattern=pattern)
            if matched_rows:
                projected = [self._build_row_payload(row) for row in matched_rows[: max(limit, 1)]]
                args["_executor_note"] = (
                    "extract_structured_items fallback=row_candidates_by_pattern_literals; "
                    f"regex_error={pattern_error}; returned {len(projected)} row(s)"
                )
                return projected

            table_rows = await self._extract_table_rows(page=page, limit=limit)
            if table_rows:
                projected = self._project_structured_rows_to_fields(rows=table_rows, fields=fields, limit=limit)
                args["_executor_note"] = (
                    "extract_structured_items fallback=table_rows; "
                    f"regex_error={pattern_error}; returned {len(projected)} row(s)"
                )
                return projected

            list_rows = await self._extract_repeated_link_or_list_items(page=page, limit=limit)
            if list_rows:
                projected = self._project_structured_rows_to_fields(rows=list_rows, fields=fields, limit=limit)
                args["_executor_note"] = (
                    "extract_structured_items fallback=list_items; "
                    f"regex_error={pattern_error}; returned {len(projected)} row(s)"
                )
                return projected
            raise

    async def _extract_repeated_structured_with_quality_fallbacks(
        self,
        *,
        page,
        fields: Any,
        limit: int,
        context_note: str,
    ) -> tuple[list[dict[str, Any]], str]:
        table_rows = await self._extract_table_rows(page=page, limit=limit)
        if table_rows:
            projected = self._normalize_table_rows_to_objects(rows=table_rows, limit=limit)
            quality = self._score_structured_fallback_quality(items=projected, limit=limit, fallback_kind="table_rows")
            if quality["is_acceptable"]:
                return (
                    projected,
                    (
                        f"extract_structured_items fallback=table_rows; {context_note}; "
                        f"candidate_count={len(table_rows)}; returned_count={len(projected)}; "
                        f"quality_score={quality['score']:.3f}; quality={quality['grade']}"
                    ),
                )

        table_like_rows = await self._extract_table_like_rows(page=page, limit=limit)
        if table_like_rows:
            projected = self._normalize_table_rows_to_objects(rows=table_like_rows, limit=limit)
            quality = self._score_structured_fallback_quality(items=projected, limit=limit, fallback_kind="table_like_rows")
            if quality["is_acceptable"]:
                return (
                    projected,
                    (
                        f"extract_structured_items fallback=table_like_rows; {context_note}; "
                        f"candidate_count={len(table_like_rows)}; returned_count={len(projected)}; "
                        f"quality_score={quality['score']:.3f}; quality={quality['grade']}"
                    ),
                )

        entity_blocks = await self._extract_repeated_entity_blocks(page=page, limit=limit)
        if entity_blocks:
            projected_entities = entity_blocks[: max(limit, 1)]
            quality = self._score_structured_fallback_quality(
                items=projected_entities,
                limit=limit,
                fallback_kind="repeated_entity_blocks",
            )
            if quality["is_acceptable"]:
                return (
                    projected_entities,
                    (
                        f"extract_structured_items fallback=repeated_entity_blocks; {context_note}; "
                        f"candidate_count={len(entity_blocks)}; returned_count={len(projected_entities)}; "
                        f"quality_score={quality['score']:.3f}; quality={quality['grade']}"
                    ),
                )

        list_rows = await self._extract_repeated_link_or_list_items(page=page, limit=limit)
        if list_rows:
            projected_list = self._normalize_table_rows_to_objects(rows=list_rows, limit=limit)
            quality = self._score_structured_fallback_quality(items=projected_list, limit=limit, fallback_kind="list_items")
            if quality["is_acceptable"]:
                return (
                    projected_list,
                    (
                        f"extract_structured_items fallback=list_items; {context_note}; "
                        f"candidate_count={len(list_rows)}; returned_count={len(projected_list)}; "
                        f"quality_score={quality['score']:.3f}; quality={quality['grade']}"
                    ),
                )
            return (
                [],
                (
                    f"extract_structured_items fallback=list_items_rejected_low_quality; {context_note}; "
                    f"candidate_count={len(list_rows)}; returned_count={len(projected_list)}; "
                    f"quality_score={quality['score']:.3f}; quality={quality['grade']}"
                ),
            )
        return ([], f"extract_structured_items fallback=none; {context_note}; no candidates found")

    async def extract_section_lines(self, page, args, runtime_state=None):
        heading_text = str(args.get("heading_text", "")).strip()
        if not heading_text:
            raise ValueError("extract_section_lines requires non-empty 'heading_text'")
        limit = int(args.get("limit", 0))
        if limit <= 0:
            raise ValueError("extract_section_lines requires positive integer 'limit'")
        self._assert_section_heading_grounded(
            heading_text=heading_text,
            runtime_state=runtime_state,
            action_args=args,
        )

        source_text = await self._load_source_text(page=page, runtime_state=runtime_state, force_refresh=True)
        lines = self._split_visible_lines(source_text)
        heading_candidates = self._find_heading_indices(lines, heading_text=heading_text, ignore_case=bool(args.get("ignore_case", True)))
        if not heading_candidates:
            raise ValueError(f"Section heading not found: {heading_text!r}")

        stop_at_heading = bool(args.get("stop_at_heading", True))
        min_line_length = int(args.get("min_line_length", 1))
        if min_line_length <= 0:
            min_line_length = 1

        prioritized_candidates = self._prioritize_heading_indices(
            heading_indices=heading_candidates,
            heading_text=heading_text,
            runtime_state=runtime_state,
        )
        collected: list[str] = []
        heading_index = prioritized_candidates[0]
        for candidate_index in prioritized_candidates:
            heading_index = candidate_index
            collected = []
            for line in lines[candidate_index + 1 :]:
                if stop_at_heading and self._looks_like_heading_line(line):
                    break
                normalized = self._normalize_line(line)
                if len(normalized) < min_line_length:
                    continue
                if normalized in collected:
                    continue
                collected.append(normalized)
                if len(collected) >= limit:
                    break
            if collected:
                break

        args["_executor_note"] = (
            f"extract_section_lines heading={heading_text!r}; start_index={heading_index}; collected={len(collected)}"
        )
        if len(collected) == 0 and not bool(args.get("allow_empty", False)):
            diagnostics = self._build_empty_section_diagnostics(
                runtime_state=runtime_state,
                failed_heading=heading_text,
            )
            raise StructuredExtractionError(
                code="insufficient_section_data",
                message=(
                    "section heading found but extracted zero lines; choose another visible heading or "
                    "use page snapshot headings"
                ),
                details={
                    "reason": "empty_section",
                    "failed_heading": heading_text,
                    "heading_text": heading_text,
                    "collected": 0,
                    "allow_empty": False,
                    "instruction": (
                        "section heading found but extracted zero lines; choose another visible heading "
                        "or use page snapshot headings"
                    ),
                    "available_non_empty_headings": diagnostics["available_non_empty_headings"],
                    "suggested_next_headings": diagnostics["suggested_next_headings"],
                },
            )
        return collected

    async def observe_page(self, page, args, runtime_state=None):
        snapshot = await self.page_observer.observe_page(
            page,
            screenshot_path=args.get("path"),
            text_limit=int(args.get("text_limit", 5000)),
        )
        if runtime_state is not None:
            runtime_state["last_page_text"] = snapshot.page_text or ""
            runtime_state["last_page_snapshot"] = snapshot.model_dump(mode="json")
            runtime_state["last_page_candidates"] = snapshot.candidates
            runtime_state["last_visible_links"] = snapshot.links
            runtime_state["last_rows"] = snapshot.rows
            runtime_state["last_tables"] = snapshot.tables
            self._mark_used_skill(runtime_state, "observe_page")
        return snapshot.model_dump(mode="json")

    async def extract_visible_links(self, page, args, runtime_state=None):
        await self._settle_page_for_read(page)
        await self._raise_if_page_blocked_or_limited(page, runtime_state=runtime_state, stage="before_extract_visible_links")
        links = await self._collect_visible_links_generic(page=page, limit=int(args.get("limit", 80)))
        min_text_length = int(args.get("min_text_length", 1))
        if min_text_length > 1:
            links = [link for link in links if len(str(link.get("text", "")).strip()) >= min_text_length]
        if runtime_state is not None:
            runtime_state["last_visible_links"] = links
            self._mark_used_skill(runtime_state, "extract_visible_links")
        args["_executor_note"] = args.get("_executor_note") or f"extract_visible_links returned {len(links)} visible link(s)"
        return links


    @staticmethod
    def _filter_links_to_same_root_domain(links: list[dict[str, Any]], *, current_url: str) -> list[dict[str, Any]]:
        current_host = urlparse(str(current_url or "")).netloc.casefold()
        parts = [part for part in current_host.split(".") if part and part != "www"]
        if len(parts) < 2:
            return links
        root_domain = ".".join(parts[-2:])
        filtered = []
        for link in links:
            host = urlparse(str(link.get("href", ""))).netloc.casefold()
            if host == root_domain or host.endswith("." + root_domain):
                filtered.append(link)
        return filtered or links












    @staticmethod
    def _append_executor_note(args: dict, note: str | None) -> None:
        note = str(note or "").strip()
        if not note:
            return
        existing = str(args.get("_executor_note", "") or "").strip()
        args["_executor_note"] = f"{existing}; {note}" if existing else note

    @classmethod
    def _confident_filter_condition(cls, *, args: dict, allow_target: bool = False) -> tuple[Any | None, bool]:
        hint_present = False
        for key in ("condition", "filter", "where"):
            if key not in args:
                continue
            hint_present = True
            value = args.get(key)
            if cls._condition_term_groups(value):
                return value, True
        if "target" in args:
            hint_present = True
            target = args.get("target")
            target_is_condition = allow_target or any(
                bool(args.get(flag))
                for flag in ("filter_by_target", "target_is_condition", "condition_from_target")
            )
            if target_is_condition and cls._condition_term_groups(target):
                return target, True
        return None, hint_present

    @staticmethod
    def _record_condition_filter_diagnostic(runtime_state, *, output_key: str, reason: str, condition: Any) -> None:
        if runtime_state is None:
            return
        diagnostics = runtime_state.setdefault("condition_filter_diagnostics", [])
        diagnostics.append(
            {
                "output_key": output_key,
                "reason": reason,
                "condition": condition,
            }
        )

    @classmethod
    def _apply_confident_item_filter(
        cls,
        *,
        items: list[dict[str, Any]],
        args: dict,
        runtime_state=None,
        output_key: str,
    ) -> tuple[list[dict[str, Any]], str]:
        condition, filter_hint_present = cls._confident_filter_condition(args=args, allow_target=False)
        if not condition:
            if filter_hint_present:
                cls._record_condition_filter_diagnostic(
                    runtime_state,
                    output_key=output_key,
                    reason="condition_not_applied",
                    condition=args.get("condition") or args.get("filter") or args.get("where") or args.get("target"),
                )
                return items, "condition_not_applied"
            return items, ""
        filtered = cls._filter_structured_rows_by_condition(rows=items, condition=condition)
        if not filtered:
            cls._record_condition_filter_diagnostic(
                runtime_state,
                output_key=output_key,
                reason="no_matching_items",
                condition=condition,
            )
            return [], "no_matching_items"
        return filtered, "filter_applied"

    async def extract_by_intent(self, page, args, runtime_state=None):
        intent = str(args.get("intent", "")).strip().lower()
        args["intent"] = intent
        self._mark_used_skill(runtime_state, "extract_by_intent")
        await self._raise_if_page_blocked_or_limited(page, runtime_state=runtime_state, stage="before_extract_by_intent")
        if intent in {"current_url", "final_url", "url"}:
            return page.url
        if intent in {"page_title", "title"}:
            return await page.title()
        if intent in {"text_block", "page_summary", "section_summary", "first_paragraph"}:
            result = await self._extract_text_block_summary(page=page, args=args, runtime_state=runtime_state)
            args["_executor_note"] = (
                f"extract_by_intent intent=text_block status={result.get('status')}; "
                f"found_fields={result.get('found_fields', [])}"
            )
            return result
        if intent == "field_schema":
            result = await self._extract_fields_by_schema(page=page, args=args, runtime_state=runtime_state)
            args["_executor_note"] = (
                f"extract_by_intent intent=field_schema status={result.get('status')}; "
                f"found_fields={result.get('found_fields', [])}"
            )
            return result
        if intent in {"search_results", "search_result", "results"}:
            result = await self._collect_search_results_generic(page=page, args=args, runtime_state=runtime_state)
            args["_executor_note"] = f"extract_by_intent intent=search_results selected_count={len(result)}"
            return result
        if intent in {"row_fields", "row_details", "structured_row_details"}:
            row = (runtime_state or {}).get("last_row_by_condition")
            if isinstance(row, dict):
                requested = args.get("fields")
                if isinstance(requested, dict):
                    names = [str(field).strip() for field in requested if str(field).strip()]
                    return {field: row.get(field) for field in names if row.get(field) not in (None, "")}
                return dict(row)
        if isinstance(args.get("intents"), list):
            row = (runtime_state or {}).get("last_row_by_condition")
            if isinstance(row, dict):
                requested = [str(item).strip() for item in args.get("intents", []) if str(item).strip()]
                if requested:
                    return {field: row.get(field) for field in requested}
        if intent in {"visible_links", "extract_visible_links", "links"}:
            return await self.extract_visible_links(page, args, runtime_state)
        if intent in {"card_items", "cards"}:
            requested_fields = args.get("fields") if isinstance(args.get("fields"), dict) else {}
            collection_args = dict(args)
            if requested_fields:
                collection_args["_defer_projection"] = True
            result = await self._collect_card_items_generic(page=page, args=collection_args, runtime_state=runtime_state)
            result, filter_note = self._apply_confident_item_filter(
                items=result,
                args=args,
                runtime_state=runtime_state,
                output_key=str(args.get("output_key", "") or "items"),
            )
            if requested_fields:
                result = [self._project_item_to_schema(item=item, fields=requested_fields) for item in result]
            args["_executor_note"] = f"extract_by_intent intent=card_items selected_count={len(result)}"
            self._append_executor_note(args, filter_note)
            return result
        if intent in {"table_rows", "rows"}:
            rows = await self._extract_table_rows_as_dicts(page=page, limit=int(args.get("limit", 80)))
            requested_headers = self._requested_table_headers(args=args, runtime_state=runtime_state)
            if requested_headers:
                rows = self._filter_table_rows_by_requested_headers(rows=rows, requested_headers=requested_headers)
            condition, filter_hint_present = self._confident_filter_condition(args=args, allow_target=True)
            if condition:
                rows = self._filter_structured_rows_by_condition(rows=rows, condition=condition)
                filter_note = "filter_applied"
                if not rows:
                    filter_note = "no_matching_items"
                    self._record_condition_filter_diagnostic(
                        runtime_state,
                        output_key=str(args.get("output_key", "") or "rows"),
                        reason=filter_note,
                        condition=condition,
                    )
            elif filter_hint_present:
                filter_note = "condition_not_applied"
                self._record_condition_filter_diagnostic(
                    runtime_state,
                    output_key=str(args.get("output_key", "") or "rows"),
                    reason=filter_note,
                    condition=args.get("condition") or args.get("filter") or args.get("where") or args.get("target"),
                )
            else:
                filter_note = ""
            header_note = f" requested_headers={requested_headers}" if requested_headers else ""
            args["_executor_note"] = f"extract_by_intent intent=table_rows selected_count={len(rows)}{header_note}"
            self._append_executor_note(args, filter_note)
            return rows
        if intent in {"value_near_anchor", "extract_value_near_anchor", "anchor_value"}:
            delegated = dict(args)
            delegated.setdefault("anchor_text", args.get("target") or args.get("anchor"))
            delegated.setdefault("value_type", args.get("value_type", "number"))
            return await self.extract_value_near_anchor(page, delegated, runtime_state)
        if intent in {"anchor_object", "label_value_object"}:
            return await self._extract_anchor_object(page=page, args=args, runtime_state=runtime_state)
        if intent in {"row_by_condition", "table_row", "find_row", "structured_row"}:
            delegated = dict(args)
            delegated.setdefault("condition", args.get("target") or args.get("anchor") or args.get("condition"))
            return await self.find_row_by_condition(page, delegated, runtime_state)
        if intent in {"max_numeric", "find_max_numeric"}:
            source_text = await self._load_source_text(page=page, runtime_state=runtime_state, force_refresh=True)
            numbers = [float(match.replace(",", ".")) for match in re.findall(r"-?\d+(?:[.,]\d+)?", source_text)]
            if not numbers:
                raise StructuredExtractionError(
                    code="extract_by_intent_no_decision",
                    message="No numeric values found for max_numeric intent.",
                    details={"intent": intent},
                )
            return max(numbers)
        raise StructuredExtractionError(
            code="extract_by_intent_no_decision",
            message=f"No generic extraction decision for intent={intent!r}.",
            details={"intent": intent, "reason": "unsupported_intent"},
        )

    async def _extract_text_block_summary(self, *, page, args: dict, runtime_state=None) -> dict[str, Any]:
        fields = args.get("fields") if isinstance(args.get("fields"), dict) else {}
        requested = [str(field).strip() for field in fields if str(field).strip()]
        if not requested:
            requested = ["title", "description"]
        payload = await page.evaluate(
            """
            ({ requested }) => {
              const norm = (value) => String(value || "").replace(/\\s+/g, " ").trim();
              const visible = (el) => {
                if (!el) return false;
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style && style.visibility !== "hidden" && style.display !== "none"
                  && Number(style.opacity || 1) !== 0 && rect.width > 0 && rect.height > 0;
              };
              const ignored = (el) => !!el.closest(
                'nav,header,footer,aside,[role="navigation"],[aria-hidden="true"],'
                + '[class*="nav" i],[class*="menu" i],[class*="sidebar" i],[class*="footer" i],'
                + '[class*="cookie" i],[class*="banner" i],[id*="cookie" i],[id*="banner" i]'
              );
              const sentence = (text) => {
                const value = norm(text);
                const match = value.match(/^(.{40,}?\\.(?:\\s|$))/);
                return norm(match ? match[1] : value);
              };
              const meaningful = (text) => {
                const value = norm(text);
                if (value.length < 35 || value.length > 900) return false;
                if (/^(search|menu|navigation|next|previous|login|sign in|read more)$/i.test(value)) return false;
                return /[.!?]|\\s/.test(value);
              };
              const roots = Array.from(document.querySelectorAll(
                'main,article,[role="main"],#content,.content,section'
              )).filter((node) => visible(node) && !ignored(node));
              if (!roots.length && document.body) roots.push(document.body);
              let best = null;
              for (const root of roots) {
                const headingNode = root.querySelector('h1,h2,[role="heading"]');
                const heading = norm(headingNode?.innerText || headingNode?.textContent || document.title || "");
                const paragraphs = Array.from(root.querySelectorAll('p, [class*="summary" i], [class*="description" i]'))
                  .filter((node) => visible(node) && !ignored(node))
                  .map((node) => norm(node.innerText || node.textContent || ""))
                  .filter(meaningful);
                let description = paragraphs[0] || "";
                if (!description) {
                  const blocks = Array.from(root.querySelectorAll('section,div,li'))
                    .filter((node) => visible(node) && !ignored(node))
                    .map((node) => norm(node.innerText || node.textContent || ""))
                    .filter(meaningful)
                    .sort((a, b) => a.length - b.length);
                  description = blocks[0] || "";
                }
                const score = (heading ? 30 : 0) + (description ? Math.min(description.length, 500) : 0)
                  - (root.matches('body') ? 80 : 0);
                if (!best || score > best.score) {
                  best = { heading, description, score };
                }
              }
              const title = norm(best?.heading || document.title || "");
              const description = sentence(best?.description || "");
              const out = {
                title,
                heading: title,
                page_title: document.title || title,
                description,
                summary: description,
                snippet: description,
                first_sentence: sentence(description),
                paragraph: description,
                current_url: window.location.href,
                final_url: window.location.href
              };
              const projected = {};
              for (const field of requested || []) {
                const key = String(field || "").trim();
                if (!key) continue;
                const lower = key.toLowerCase();
                if (lower.includes("title") || lower.includes("heading") || lower.includes("name")) projected[key] = title;
                else if (lower.includes("url")) projected[key] = window.location.href;
                else if (lower.includes("sentence")) projected[key] = out.first_sentence;
                else if (lower.includes("paragraph")) projected[key] = out.paragraph;
                else if (lower.includes("description") || lower.includes("summary") || lower.includes("snippet")) projected[key] = description;
                else projected[key] = out[key] || description || title;
              }
              return { ...out, projected };
            }
            """,
            {"requested": requested},
        )
        data = dict(payload or {}) if isinstance(payload, dict) else {}
        projected = data.get("projected") if isinstance(data.get("projected"), dict) else {}
        result: dict[str, Any] = {}
        for field in requested:
            value = projected.get(field)
            if value not in (None, "", [], {}):
                result[field] = value
        for key in ("title", "heading", "description", "summary", "snippet", "first_sentence", "paragraph", "current_url", "final_url"):
            if key in requested and key not in result and data.get(key) not in (None, "", [], {}):
                result[key] = data[key]
        found = [field for field in requested if result.get(field) not in (None, "", [], {})]
        description_like = any(field for field in requested if self._field_is_description_like(field))
        if description_like and not any(result.get(field) for field in requested if self._field_is_description_like(field)):
            try:
                source_text = await self._load_source_text(page=page, runtime_state=runtime_state, force_refresh=True)
            except Exception:
                source_text = ""
            fallback = self._text_block_from_source_text(
                source_text=source_text,
                title=str(data.get("title") or data.get("page_title") or ""),
            )
            description = fallback.get("description")
            if description:
                for field in requested:
                    lower = field.casefold()
                    if self._field_is_description_like(field):
                        result[field] = fallback.get("first_sentence") if "sentence" in lower else description
                    elif any(token in lower for token in ("title", "heading", "name")) and fallback.get("title"):
                        result[field] = fallback["title"]
                    elif "url" in lower and data.get("current_url"):
                        result[field] = data["current_url"]
                found = [field for field in requested if result.get(field) not in (None, "", [], {})]
        if description_like and not any(result.get(field) for field in requested if self._field_is_description_like(field)):
            raise StructuredExtractionError(
                code="description_not_found",
                message="No meaningful description/paragraph text block was found.",
                details={"requested_fields": requested},
            )
        result["status"] = "success" if len(found) == len(requested) else ("partial_success" if found else "not_found")
        result["found_fields"] = found
        result["missing_fields"] = [field for field in requested if field not in found]
        self._mark_used_skill(runtime_state, "text_block_extraction")
        return result

    @staticmethod
    def _field_is_description_like(field: str) -> bool:
        normalized = str(field or "").strip().casefold()
        return any(token in normalized for token in ("description", "summary", "snippet", "sentence", "paragraph", "overview"))

    @classmethod
    def _text_block_from_source_text(cls, *, source_text: str, title: str = "") -> dict[str, str]:
        lines = [cls._normalize_line(line) for line in str(source_text or "").splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            return {}
        ui_exact = {
            "skip to main content",
            "skip to search",
            "search",
            "menu",
            "tools",
            "about",
            "blog",
            "all",
            "learn",
            "next",
            "previous",
            "sign in",
            "login",
            "theme",
            "light",
            "dark",
        }
        heading = cls._normalize_line(title)
        for line in lines:
            folded = line.casefold()
            if line.casefold() in ui_exact or len(line) < 4:
                continue
            if 4 <= len(line) <= 180 and not re.search(r"[.!?]", line):
                heading = heading or line
                break
        candidates: list[str] = []
        for line in lines:
            folded = line.casefold()
            if folded in ui_exact:
                continue
            if len(line) < 50 or len(line) > 900:
                continue
            if line == heading:
                continue
            if re.fullmatch(r"[\w\s/|+-]{1,80}", line) and not re.search(r"[.!?]", line):
                continue
            if not re.search(r"[.!?]", line):
                continue
            if folded.startswith(("note: this page", "was this page helpful", "this page was last modified")):
                continue
            candidates.append(line)
        if not candidates:
            return {"title": heading} if heading else {}
        description = candidates[0]
        sentence_match = re.match(r"^(.{40,}?[.!?])(?:\s|$)", description)
        first_sentence = cls._normalize_line(sentence_match.group(1) if sentence_match else description)
        return {
            "title": heading,
            "description": description,
            "summary": description,
            "snippet": description,
            "paragraph": description,
            "first_sentence": first_sentence,
        }

    async def _extract_fields_by_schema(self, *, page, args: dict, runtime_state=None) -> dict[str, Any]:
        fields = args.get("fields") if isinstance(args.get("fields"), dict) else {}
        if not fields:
            raise StructuredExtractionError(
                code="field_schema_missing_schema",
                message="field_schema extraction requires a non-empty fields schema.",
                details={"intent": args.get("intent")},
            )
        candidates = [str(item).strip() for item in args.get("region_candidates", []) if str(item).strip()]
        hint = str(args.get("region_hint", "") or "").strip()
        if hint:
            candidates.extend(part for part in re.split(r"[/,|]+", hint) if part.strip())
        candidates = self._unique_preserve_order(candidates)

        navigated = await self._navigate_to_candidate_region(page=page, candidates=candidates)
        if navigated:
            await self._wait_after_possible_navigation(page)
            if runtime_state is not None:
                runtime_state.pop("last_page_text", None)

        source_text = await self._load_source_text(page=page, runtime_state=runtime_state, force_refresh=navigated)
        region_text = self._select_candidate_region_text(source_text=source_text, candidates=candidates) or source_text
        href_values = await self._extract_href_values(page)
        page_title = await page.title()
        page_description = await self._extract_page_description(page)
        text_block_summary: dict[str, Any] | None = None
        payload: dict[str, Any] = {}

        table_payload = self._field_schema_table_row_from_source_text(
            fields=fields,
            source_text=source_text,
            runtime_state=runtime_state,
        )
        if table_payload:
            return table_payload
        visible_summary_payload = await self._field_schema_visible_object_summary(
            page=page,
            fields=fields,
            runtime_state=runtime_state,
        )
        if visible_summary_payload:
            return visible_summary_payload

        for field_name, rule in fields.items():
            field = str(field_name or "").strip()
            if not field:
                continue
            rule_dict = rule if isinstance(rule, dict) else {"type": str(rule or "text")}
            field_type = str(rule_dict.get("type", "text") or "text").strip().casefold()
            anchors = [str(item).strip() for item in rule_dict.get("anchors", []) if str(item).strip()] if isinstance(rule_dict.get("anchors"), list) else []
            value = None
            field_hint = casefold_with_mojibake_repairs(field)
            if field_type in {"current_url", "final_url", "url"}:
                value = getattr(page, "url", "")
            elif field_type in {"page_title", "title"}:
                value = page_title
            elif field_type in {"meta_description", "description", "summary", "snippet"}:
                if text_block_summary is None:
                    try:
                        text_block_summary = await self._extract_text_block_summary(
                            page=page,
                            args={"fields": {field: {"type": "description"}}},
                            runtime_state=runtime_state,
                        )
                    except Exception:
                        text_block_summary = {}
                value = (
                    text_block_summary.get(field)
                    or text_block_summary.get("description")
                    or page_description
                    or self._text_field_candidate(region_text=region_text, anchors=anchors)
                )
            elif field_type in {"email", "phone", "integer", "count", "number", "decimal", "float"}:
                value = self._extract_typed_scalar_candidate(
                    value_type=field_type,
                    text=region_text,
                    href_values=href_values,
                )
            elif field_type in {"address", "postal_address"} or any(token in field_hint for token in ("address", "адрес", "почтов")):
                value = self._extract_address_candidate(region_text=region_text, anchors=anchors)
            else:
                value = self._text_field_candidate(region_text=region_text, anchors=anchors)
            if value not in (None, "", [], {}):
                payload[field] = value

        requested = [str(field).strip() for field in fields if str(field).strip()]
        found = [field for field in requested if field in payload and payload[field] not in (None, "", [], {})]
        payload["status"] = "success" if len(found) == len(requested) else ("partial_success" if found else "not_found")
        payload["found_fields"] = found
        payload["missing_fields"] = [field for field in requested if field not in found]
        return payload

    async def _field_schema_visible_object_summary(self, *, page, fields: dict[str, Any], runtime_state=None) -> dict[str, Any]:
        goal = str((runtime_state or {}).get("user_goal", "") or "")
        if not self._goal_requests_visible_object_summary(goal=goal, fields=fields):
            return {}
        links = []
        if isinstance(runtime_state, dict) and isinstance(runtime_state.get("last_visible_links"), list):
            links = [dict(item) for item in runtime_state.get("last_visible_links", []) if isinstance(item, dict)]
        if not links:
            try:
                links = await self._collect_visible_links_generic(page=page, limit=80)
            except Exception:
                links = []
        return self._visible_object_summary_from_links(goal=goal, fields=fields, links=links)

    @classmethod
    def _goal_requests_visible_object_summary(cls, *, goal: str, fields: dict[str, Any]) -> bool:
        combined = " ".join([str(goal or ""), *[str(field or "") for field in fields.keys()]])
        text = casefold_with_mojibake_repairs(combined)
        wants_visible = any(token in text for token in ("visual", "visible", "central", "center", "large", "block", "blocks", "видим", "визуаль", "централь", "крупн", "блок"))
        wants_count = any(token in text for token in ("count", "number", "total", "колич", "числ", "посч"))
        wants_names = any(token in text for token in ("name", "names", "language", "languages", "назван", "язык"))
        wants_listable = any(token in text for token in ("link", "links", "block", "blocks", "item", "items", "язык", "блок"))
        return wants_visible and wants_count and (wants_names or wants_listable)

    @classmethod
    def _visible_object_summary_from_links(
        cls,
        *,
        goal: str,
        fields: dict[str, Any],
        links: list[dict[str, Any]],
    ) -> dict[str, Any]:
        requested = [str(field).strip() for field in fields if str(field).strip()]
        if not requested:
            return {}
        candidates = cls._visible_object_candidates_for_goal(goal=goal, links=links)
        if not candidates:
            return {}
        names = [str(item.get("name") or item.get("text") or "").strip() for item in candidates if str(item.get("name") or item.get("text") or "").strip()]
        count = len(names)
        payload: dict[str, Any] = {
            "count": count,
            "names": names,
            "items": candidates,
            "visible_count": count,
            "visible_names": names,
        }
        for field in requested:
            normalized = casefold_with_mojibake_repairs(field)
            if any(token in normalized for token in ("count", "number", "total", "колич", "числ")):
                payload[field] = count
            elif any(token in normalized for token in ("name", "names", "language", "languages", "назван", "язык")):
                payload[field] = names
            elif any(token in normalized for token in ("item", "items", "block", "blocks", "link", "links", "блок", "ссылк")):
                payload[field] = candidates
        found = [field for field in requested if payload.get(field) not in (None, "", [], {})]
        payload["status"] = "success" if len(found) == len(requested) else ("partial_success" if found else "not_found")
        payload["found_fields"] = found
        payload["missing_fields"] = [field for field in requested if field not in found]
        return payload

    @classmethod
    def _visible_object_candidates_for_goal(cls, *, goal: str, links: list[dict[str, Any]]) -> list[dict[str, Any]]:
        text = casefold_with_mojibake_repairs(goal)
        wants_language_blocks = any(token in text for token in ("language", "languages", "язык"))
        ranked: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for index, link in enumerate(links):
            raw_text = " ".join(str(link.get("text", "") or "").split()).strip()
            if not raw_text:
                continue
            name = cls._visible_object_name_from_text(raw_text)
            if not name:
                continue
            score = cls._visible_object_candidate_score(link=link, raw_text=raw_text, name=name, wants_language_blocks=wants_language_blocks)
            if score <= 0:
                continue
            dedupe_key = name.casefold()
            if dedupe_key in seen_names:
                continue
            seen_names.add(dedupe_key)
            current = dict(link)
            current["name"] = name
            current["text"] = raw_text
            current["_score"] = score
            current["_rank"] = index
            ranked.append(current)
        ranked.sort(key=lambda item: (-float(item.get("_score", 0)), int(item.get("_rank", 9999))))
        best = ranked[:50]
        if wants_language_blocks:
            best = [item for item in best if cls._text_has_count_unit(str(item.get("text") or ""))] or best
        if len(best) > 15:
            top_score = float(best[0].get("_score", 0) or 0)
            best = [item for item in best if float(item.get("_score", 0) or 0) >= top_score - 1.5][:15]
        return [
            {key: value for key, value in item.items() if key not in {"_score", "_rank"}}
            for item in best
        ]

    @staticmethod
    def _visible_object_name_from_text(text: str) -> str:
        raw = " ".join(str(text or "").split()).strip()
        if not raw:
            return ""
        value = re.sub(
            r"\s+[0-9][0-9\s,.\u00A0\u202F+]*(?:\s*(?:articles?|items?|results?|rows?|стат(?:ей|ьи|ья)?|artikel|artículos?|artigos?|voci|haseł|記事|条目|條目))?.*$",
            "",
            raw,
            flags=re.IGNORECASE,
        ).strip(" :;,.")
        return value if 1 <= len(value) <= 80 else ""

    @classmethod
    def _visible_object_candidate_score(cls, *, link: dict[str, Any], raw_text: str, name: str, wants_language_blocks: bool) -> float:
        text = str(raw_text or "")
        score = 0.0
        if cls._text_has_count_unit(text):
            score += 5.0
        if re.search(r"\d", text):
            score += 1.0
        if wants_language_blocks and cls._text_has_count_unit(text):
            score += 2.0
        bbox = link.get("bbox") if isinstance(link.get("bbox"), dict) else {}
        viewport = link.get("viewport") if isinstance(link.get("viewport"), dict) else {}
        try:
            vw = float(viewport.get("width") or 0)
            vh = float(viewport.get("height") or 0)
            cx = float(bbox.get("center_x") or 0)
            cy = float(bbox.get("center_y") or 0)
            width = float(bbox.get("width") or 0)
            height = float(bbox.get("height") or 0)
        except (TypeError, ValueError):
            vw = vh = cx = cy = width = height = 0.0
        if vw > 0 and vh > 0 and width > 0 and height > 0:
            center_distance = abs((cx / vw) - 0.5) + abs((cy / vh) - 0.35)
            score += max(0.0, 2.5 - center_distance * 4.0)
            if 0.05 <= cy / vh <= 0.75:
                score += 1.0
            if height >= 24 or width >= 80:
                score += 0.5
        selector = str(link.get("selector", "") or "").casefold()
        href = str(link.get("href", "") or "")
        if any(token in selector for token in ("footer", "app-badge", "other-project", "privacy", "donate")):
            score -= 5.0
        if any(token in href.casefold() for token in ("donate", "privacy", "itunes", "play.google", "wikimediafoundation")):
            score -= 3.0
        if name.casefold() in {"search", "menu", "google play store", "apple app store"}:
            score -= 5.0
        return score

    @staticmethod
    def _text_has_count_unit(text: str) -> bool:
        return bool(
            re.search(
                r"[0-9][0-9\s,.\u00A0\u202F+]*\s*(?:articles?|items?|results?|rows?|стат(?:ей|ьи|ья)?|artikel|artículos?|artigos?|voci|haseł|記事|条目|條目)",
                str(text or ""),
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    async def _extract_page_description(page) -> str:
        try:
            meta_description = await page.locator('meta[name="description"]').first.get_attribute("content")
        except Exception:
            meta_description = ""
        normalized_meta = " ".join(str(meta_description or "").split()).strip()
        if normalized_meta:
            return normalized_meta
        try:
            paragraphs = await page.locator(
                "main p, article p, [role='main'] p, main [class*='summary' i], "
                "article [class*='summary' i], [role='main'] [class*='summary' i]"
            ).evaluate_all(
                """
                els => els.map(el => {
                  const style = window.getComputedStyle(el);
                  const rect = el.getBoundingClientRect();
                  const visible = style
                    && style.visibility !== "hidden"
                    && style.display !== "none"
                    && Number(style.opacity || 1) !== 0
                    && rect.width > 0
                    && rect.height > 0;
                  return visible ? String(el.innerText || el.textContent || "").replace(/\\s+/g, " ").trim() : "";
                }).filter(Boolean)
                """
            )
        except Exception:
            return ""
        for paragraph in paragraphs or []:
            normalized = " ".join(str(paragraph or "").split()).strip()
            if 40 <= len(normalized) <= 700:
                return normalized
        return ""

    @staticmethod
    def _unique_preserve_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            key = item.casefold()
            if key and key not in seen:
                result.append(item)
                seen.add(key)
        return result

    async def _navigate_to_candidate_region(self, *, page, candidates: list[str]) -> bool:
        if not candidates:
            return False
        folded_candidates = [candidate.casefold() for candidate in candidates]
        try:
            links = await page.locator("a").evaluate_all(
                """els => els.slice(0, 200).map(a => ({text: (a.innerText || a.textContent || '').trim(), href: a.href || ''}))"""
            )
        except Exception:
            return False
        if not isinstance(links, list):
            return False
        for link in links:
            if not isinstance(link, dict):
                continue
            text = str(link.get("text", "") or "").strip()
            href = str(link.get("href", "") or "").strip()
            folded = f"{text} {href}".casefold()
            if href and any(candidate in folded for candidate in folded_candidates):
                try:
                    await page.goto(href, wait_until="domcontentloaded", timeout=20000)
                    return True
                except Exception:
                    return False
        return False

    @staticmethod
    def _select_candidate_region_text(*, source_text: str, candidates: list[str]) -> str:
        lines = [line.strip() for line in str(source_text or "").splitlines() if line.strip()]
        if not lines or not candidates:
            return ""
        folded_candidates = [candidate.casefold() for candidate in candidates]
        for index, line in enumerate(lines):
            if any(candidate in line.casefold() for candidate in folded_candidates):
                start = max(0, index - 2)
                end = min(len(lines), index + 12)
                return "\n".join(lines[start:end])
        return ""

    async def _extract_href_values(self, page) -> list[str]:
        try:
            hrefs = await page.locator("a[href]").evaluate_all("els => els.slice(0, 300).map(a => a.getAttribute('href') || '')")
        except Exception:
            return []
        if not isinstance(hrefs, list):
            return []
        return [str(href or "").strip() for href in hrefs if str(href or "").strip()]

    @classmethod
    def _extract_typed_scalar_candidate(cls, *, value_type: str, text: str, href_values: list[str]) -> str:
        pattern = cls._resolve_value_pattern(value_type)
        if not pattern:
            return ""
        candidates: list[str] = []
        for value in href_values:
            folded = value.casefold()
            if value_type == "email" and folded.startswith("mailto:"):
                candidates.append(value.split(":", 1)[1].split("?", 1)[0])
            elif value_type == "phone" and folded.startswith("tel:"):
                candidates.append(value.split(":", 1)[1])
            elif value_type == "url" and folded.startswith(("http://", "https://")):
                candidates.append(value)
        candidates.append(str(text or ""))
        for candidate in candidates:
            match = re.search(pattern, candidate, flags=re.IGNORECASE)
            if match:
                return str(match.group(1) if match.lastindex else match.group(0)).strip()
        return ""

    @staticmethod
    def _text_field_candidate(*, region_text: str, anchors: list[str]) -> str:
        lines = [line.strip() for line in str(region_text or "").splitlines() if line.strip()]
        folded_anchors = [anchor.casefold() for anchor in anchors]
        for index, line in enumerate(lines):
            folded = line.casefold()
            if folded_anchors and any(anchor in folded for anchor in folded_anchors):
                after_separator = re.split(r"[:：—-]", line, maxsplit=1)
                if len(after_separator) > 1 and after_separator[1].strip():
                    return after_separator[1].strip()
                if index + 1 < len(lines):
                    return lines[index + 1].strip()
        return ""

    @classmethod
    def _extract_address_candidate(cls, *, region_text: str, anchors: list[str]) -> str:
        lines = [line.strip() for line in str(region_text or "").splitlines() if line.strip()]
        if not lines:
            return ""
        folded_anchors = [casefold_with_mojibake_repairs(anchor) for anchor in anchors if str(anchor or "").strip()]
        address_label = re.compile(r"\baddress\b|адрес|почтов", flags=re.IGNORECASE)
        for index, line in enumerate(lines):
            folded = casefold_with_mojibake_repairs(line)
            if folded_anchors and not any(anchor in folded for anchor in folded_anchors):
                continue
            inline = cls._address_from_labeled_line(line)
            if inline:
                return inline
            for offset in range(1, 4):
                if index + offset < len(lines) and cls._looks_like_address_line(lines[index + offset]):
                    return lines[index + offset].strip()
        for index, line in enumerate(lines):
            if not address_label.search(line):
                continue
            inline = cls._address_from_labeled_line(line)
            if inline:
                return inline
            for offset in range(1, 4):
                if index + offset < len(lines) and cls._looks_like_address_line(lines[index + offset]):
                    return lines[index + offset].strip()
        for line in lines:
            if cls._looks_like_address_line(line):
                return line.strip()
        return ""

    @staticmethod
    def _address_from_labeled_line(line: str) -> str:
        parts = re.split(r"[:\u2013\u2014-]", str(line or ""), maxsplit=1)
        if len(parts) > 1 and ActionHandlers._looks_like_address_line(parts[1]):
            return parts[1].strip()
        return ""

    @staticmethod
    def _looks_like_address_line(line: str) -> bool:
        text = " ".join(str(line or "").split()).strip()
        if len(text) < 12 or len(text) > 240:
            return False
        has_postal_code = bool(re.search(r"\b\d{5,6}\b", text))
        has_street_marker = bool(
            re.search(
                r"\b(street|st\\.?|road|rd\\.?|avenue|ave\\.?|boulevard|blvd\\.?|lane|ln\\.?|drive|dr\\.?)\b|"
                r"\b(ул\\.?|улица|проспект|пр\\.?|наб\\.?|набережная|пер\\.?|переулок|д\\.?|дом|корпус)\b",
                text,
                flags=re.IGNORECASE,
            )
        )
        has_city_or_country = bool(re.search(r"\b(россия|russia|санкт|петербург|москва|moscow|city)\b", text, flags=re.IGNORECASE))
        comma_parts = len([part for part in text.split(",") if part.strip()])
        return (has_postal_code and (has_street_marker or has_city_or_country or comma_parts >= 3)) or (
            has_street_marker and has_city_or_country and comma_parts >= 2
        )

    async def find_row_by_condition(self, page, args, runtime_state=None):
        limit = int(args.get("limit", 120))
        table_rows = await self._extract_table_rows_as_dicts(page=page, limit=limit)
        generic_rows = await self._collect_row_candidates_generic(page=page, limit=limit)
        rows = [*table_rows, *generic_rows]
        condition = self._normalize_row_action_condition(args.get("condition"))
        if condition != args.get("condition"):
            args["condition"] = condition
        term_groups = self._condition_term_groups(condition)
        if not term_groups:
            raise StructuredExtractionError(
                code="missing_row_condition",
                message="find_row_by_condition requires non-empty condition terms",
                details={"condition": condition},
            )
        field_condition = self._condition_field_operator_value(condition)
        if field_condition:
            matches = self._filter_structured_rows_by_condition(rows=table_rows or rows, condition=condition)
            if not matches:
                raise StructuredExtractionError(
                    code="row_not_found",
                    message=f"No row matched field condition={condition!r}",
                    details={"condition": condition, "rows_checked": len(table_rows or rows)},
                )
            payload = self._build_row_payload(matches[0])
            if runtime_state is not None:
                runtime_state["last_row_ref"] = payload
                runtime_state["last_row_by_condition"] = payload
                self._mark_used_skill(runtime_state, "row_or_item_action")
            args["_executor_note"] = f"find_row_by_condition matched field_condition={condition!r}; selector={payload.get('selector')!r}"
            return payload
        matched_payloads: list[dict[str, Any]] = []
        missing_groups: list[list[str]] = []

        for terms in term_groups:
            matched_payload = None
            row = self._best_matching_row_by_terms(rows=rows, terms=terms)
            if row is not None:
                if self._row_candidate_is_too_broad(row=row, terms=terms):
                    row = None
                else:
                    matched_payload = self._build_row_payload(row)
            if matched_payload is None:
                missing_groups.append(terms)
            else:
                signature = (matched_payload.get("selector"), matched_payload.get("text"))
                if not any((item.get("selector"), item.get("text")) == signature for item in matched_payloads):
                    matched_payloads.append(matched_payload)

        if missing_groups:
            flat_terms = self._condition_terms(condition)
            raise StructuredExtractionError(
                code="row_not_found",
                message=f"No row matched condition terms={flat_terms}; missing_groups={missing_groups}",
                details={
                    "terms": flat_terms,
                    "term_groups": term_groups,
                    "missing_groups": missing_groups,
                    "rows_checked": len(rows),
                    "matched_count": len(matched_payloads),
                },
            )

        if len(term_groups) > 1:
            if runtime_state is not None:
                runtime_state["last_rows_by_condition"] = matched_payloads
                if matched_payloads:
                    runtime_state["last_row_ref"] = matched_payloads[0]
                    runtime_state["last_row_by_condition"] = matched_payloads[0]
                self._mark_used_skill(runtime_state, "row_or_item_action")
            args["_executor_note"] = (
                f"find_row_by_condition matched term_groups={term_groups}; "
                f"returned_count={len(matched_payloads)}"
            )
            return matched_payloads

        terms = term_groups[0]
        row = self._best_matching_row_by_terms(rows=rows, terms=terms)
        if row is not None:
            if self._row_candidate_is_too_broad(row=row, terms=terms):
                raise StructuredExtractionError(
                    code="row_not_found",
                    message="Only broad ancestor containers matched the row condition.",
                    details={"terms": terms, "rows_checked": len(rows), "reason": "broad_container_match"},
                )
            payload = self._build_row_payload(row)
            if runtime_state is not None:
                runtime_state["last_row_ref"] = payload
                runtime_state["last_row_by_condition"] = payload
                self._mark_used_skill(runtime_state, "row_or_item_action")
            args["_executor_note"] = f"find_row_by_condition matched terms={terms}; selector={payload.get('selector')!r}"
            return payload
        raise StructuredExtractionError(
            code="row_not_found",
            message=f"No row matched condition terms={terms}",
            details={"terms": terms, "rows_checked": len(rows)},
        )

    @classmethod
    def _best_matching_row_by_terms(cls, *, rows: list[dict[str, Any]], terms: list[str]) -> dict[str, Any] | None:
        best: tuple[int, int, dict[str, Any]] | None = None
        for index, row in enumerate(rows):
            text = str(row.get("text", "")).casefold()
            cells_text = " ".join(str(cell) for cell in row.get("cells", [])).casefold()
            haystack = f"{text} {cells_text}"
            if not all(term.casefold() in haystack for term in terms):
                continue
            score = cls._row_match_score(row=row, terms=terms)
            if best is None or (score, index) < (best[0], best[1]):
                best = (score, index, row)
        return best[2] if best else None

    @staticmethod
    def _row_match_score(*, row: dict[str, Any], terms: list[str]) -> int:
        text = str(row.get("text", "") or "")
        tag = str(row.get("tag", "") or "").casefold()
        role = str(row.get("role", "") or "").casefold()
        cells = row.get("cells") if isinstance(row.get("cells"), list) else []
        score = min(len(text), 5000)
        if tag in {"tr", "li"} or role in {"row", "listitem"}:
            score -= 1200
        if cells:
            score -= 300
        folded_text = text.casefold()
        normalized_terms = [term.casefold() for term in terms if str(term).strip()]
        if normalized_terms and any(folded_text.strip() == term for term in normalized_terms):
            score -= 600
        if normalized_terms and all(any(term in str(cell).casefold() for cell in cells) for term in normalized_terms):
            score -= 400
        return score

    @staticmethod
    def _row_candidate_is_too_broad(*, row: dict[str, Any], terms: list[str]) -> bool:
        text = str(row.get("text", "") or "")
        if len(text) < 1200:
            return False
        tag = str(row.get("tag", "") or "").casefold()
        role = str(row.get("role", "") or "").casefold()
        cells = row.get("cells") if isinstance(row.get("cells"), list) else []
        if cells or tag in {"tr", "li"} or role in {"row", "listitem"}:
            return False
        folded = text.casefold()
        return all(str(term or "").casefold() in folded for term in terms if str(term or "").strip())

    @staticmethod
    def _row_action_control_matches(*, haystack: str, target_words: list[str]) -> bool:
        text = str(haystack or "").casefold()
        for word in target_words:
            token = str(word or "").casefold().strip()
            if not token:
                continue
            if re.search(rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])", text):
                return True
        return False

    @classmethod
    def _normalize_row_action_condition(cls, condition: Any) -> Any:
        if isinstance(condition, dict):
            normalized = dict(condition)
            for key in ("value", "contains", "text", "target_text", "label", "name"):
                value = normalized.get(key)
                if not isinstance(value, str):
                    continue
                row_text = cls._row_text_from_action_phrase(value)
                if not row_text or row_text == value:
                    continue
                if key in {"contains", "text", "target_text", "label", "name"} and not str(normalized.get("field", "") or "").strip():
                    return {"field": None, "operator": "contains", "value": row_text}
                normalized[key] = row_text
                normalized.setdefault("operator", "contains")
                return normalized
            return condition
        if isinstance(condition, str):
            row_text = cls._row_text_from_action_phrase(condition)
            if row_text and row_text != condition.strip():
                return {"field": None, "operator": "contains", "value": row_text}
        return condition

    @staticmethod
    def _row_text_from_action_phrase(value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip(" :;,.\"'")
        if not text:
            return ""
        row_subject = r"(?:table\s+row|list\s+item|row|item|record)"
        relation = (
            r"(?:named|called|labeled|labelled|with\s+text|containing\s+text|"
            r"containing|contains|whose\s+text\s+(?:is|contains))"
        )
        patterns = [
            rf"\b(?:for|inside|within|in)\s+(?:the\s+)?{row_subject}\s+{relation}\s+(.+)$",
            rf"\b{row_subject}\s+{relation}\s+(.+)$",
            rf"\b(?:for|inside|within|in)\s+(?:the\s+)?{row_subject}\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = re.sub(r"\s+", " ", match.group(1)).strip(" :;,.\"'")
            candidate = re.split(
                r"\s+(?:then|afterwards)\b|\s+and\s+(?:return|extract|show)\b|[.;]",
                candidate,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" :;,.\"'")
            return candidate
        return text

    async def click_row_action(self, page, args, runtime_state=None):
        row_ref = args.get("row_ref") or (runtime_state or {}).get("last_row_ref")
        if not isinstance(row_ref, dict) or not row_ref.get("selector"):
            condition = self._normalize_row_action_condition(args.get("condition"))
            row_ref = await self.find_row_by_condition(page, {"condition": condition, "limit": args.get("limit", 120)}, runtime_state)
        selector = str(row_ref.get("selector", "")).strip()
        if not selector:
            raise ValueError("click_row_action requires resolved row selector")
        explicit_candidates = args.get("target_candidates")
        explicit_selector = str(args.get("action_selector", "") or "").strip()
        action_name = str(args.get("action_name") or args.get("target_text") or "").strip().lower()
        if not action_name and not explicit_candidates and not explicit_selector:
            action_name = "open"
        row_context = self._row_context_for_ref(page=page, row_ref=row_ref)
        row = await self._row_locator_for_ref(row_context=row_context, selector=selector, row_ref=row_ref)
        if action_name in {"open", "select", "click", "choose"} and not explicit_candidates and not explicit_selector:
            await row.click()
        else:
            if isinstance(explicit_candidates, list):
                target_words = [str(item).strip() for item in explicit_candidates if str(item).strip()]
            else:
                target_words = []
            target_words.extend(word for word in re.split(r"[_\-\s]+", action_name) if word)
            default_action_selector = (
                "button,a,[role='button'],input,[title],[aria-label],"
                "[class*='close' i],[class*='delete' i],[class*='remove' i]"
            )
            locator = row.locator(explicit_selector or default_action_selector)
            count = await locator.count()
            selected = locator.first if explicit_selector and count else None
            if selected is None:
                for index in range(count):
                    item = locator.nth(index)
                    text_parts = [
                        await item.inner_text(timeout=500) if hasattr(item, "inner_text") else "",
                        await item.get_attribute("aria-label") or "",
                        await item.get_attribute("title") or "",
                        await item.get_attribute("class") or "",
                        await item.get_attribute("id") or "",
                    ]
                    haystack = " ".join(text_parts).casefold()
                    delete_like = action_name in {"delete", "remove", "close"}
                    if self._row_action_control_matches(haystack=haystack, target_words=target_words) or (
                        delete_like and any(token in haystack for token in ("×", "close", "delete", "remove"))
                    ):
                        selected = item
                        break
            if selected is None:
                if await self._retry_row_action_in_runnable_example(
                    page=page,
                    args=args,
                    runtime_state=runtime_state,
                    row_ref=row_ref,
                    action_name=action_name,
                ):
                    retry_args = dict(args)
                    retry_args.pop("row_ref", None)
                    retry_args["_runnable_example_retry_done"] = True
                    if runtime_state is not None:
                        runtime_state.pop("last_row_ref", None)
                    return await self.click_row_action(page, retry_args, runtime_state)
                raise StructuredExtractionError(
                    code="row_action_target_not_found",
                    message=f"No row action control found for action_name={action_name!r}",
                    details={"action_name": action_name, "row_selector": selector},
                )
            await self._click_row_action_control(selected)
        self._mark_used_skill(runtime_state, "row_or_item_action")
        args["_executor_note"] = f"click_row_action action_name={action_name!r}; row_selector={selector!r}"
        return {"clicked": True, "action_name": action_name, "row_ref": row_ref}

    async def _retry_row_action_in_runnable_example(
        self,
        *,
        page,
        args: dict[str, Any],
        runtime_state,
        row_ref: dict[str, Any],
        action_name: str,
    ) -> bool:
        if bool(args.get("_runnable_example_retry_done")):
            return False
        if str(args.get("action_selector", "") or "").strip():
            return False
        if isinstance(args.get("target_candidates"), list) and args.get("target_candidates"):
            return False
        condition_text = ""
        condition = self._normalize_row_action_condition(args.get("condition"))
        if isinstance(condition, dict):
            condition_text = str(condition.get("value") or condition.get("contains") or "").strip()
        if not condition_text:
            condition_text = str(row_ref.get("text") or "").strip()
        href = await self._find_runnable_example_link_for_context(
            page=page,
            context_text=condition_text,
            action_name=action_name,
        )
        if not href:
            return False
        await page.goto(href, wait_until="domcontentloaded", timeout=int(args.get("navigation_timeout_ms", 20000)))
        await self._wait_after_possible_navigation(page)
        await self._raise_if_page_blocked_or_limited(page, runtime_state=runtime_state, stage="after_runnable_example_navigation")
        if runtime_state is not None:
            runtime_state["last_runnable_example_url"] = href
        return True

    async def _find_runnable_example_link_for_context(self, *, page, context_text: str, action_name: str) -> str:
        try:
            payload = await page.evaluate(
                """
                ({ contextText, actionName }) => {
                  const norm = (value) => String(value || "").replace(/\\s+/g, " ").trim();
                  const lower = (value) => norm(value).toLowerCase();
                  const isVisible = (el) => {
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return style && style.visibility !== "hidden" && style.display !== "none"
                      && Number(style.opacity || 1) !== 0 && rect.width > 0 && rect.height > 0;
                  };
                  const ignored = (el) => !!el.closest('nav,header,footer,aside,[role="navigation"],[aria-hidden="true"]');
                  const samePageJump = (href) => {
                    try {
                      const url = new URL(href, document.baseURI);
                      return url.origin === location.origin && url.pathname === location.pathname && !!url.hash;
                    } catch {
                      return false;
                    }
                  };
                  const badLabel = (text, href) => {
                    const haystack = `${text} ${href}`.toLowerCase();
                    return /\\b(login|sign in|signup|register|pricing|upgrade|certificate|certified|course|academy|advert|privacy|terms|download app|share)\\b/.test(haystack);
                  };
                  const runnableTokens = [
                    "try", "demo", "example", "run", "live", "preview",
                    "playground", "sandbox", "editor", "open in", "launch"
                  ];
                  const actionTokens = lower(actionName).split(/[^a-z0-9]+/).filter(Boolean);
                  const wantedContext = lower(contextText);
                  const orderedAnchors = [];
                  const seenAnchors = new Set();
                  for (const selector of [
                    'main a[href], article a[href], [role="main"] a[href], #main a[href], #maincontent a[href], .content a[href], section a[href]',
                    'a[href]'
                  ]) {
                    for (const anchor of Array.from(document.querySelectorAll(selector))) {
                      if (seenAnchors.has(anchor)) continue;
                      seenAnchors.add(anchor);
                      orderedAnchors.push(anchor);
                      if (orderedAnchors.length >= 1200) break;
                    }
                    if (orderedAnchors.length >= 1200) break;
                  }
                  const candidates = [];
                  for (const anchor of orderedAnchors) {
                    if (!isVisible(anchor) || ignored(anchor)) continue;
                    const href = anchor.href || anchor.getAttribute("href") || "";
                    if (!href || /^javascript:/i.test(href) || samePageJump(href)) continue;
                    const text = norm(anchor.innerText || anchor.textContent || anchor.getAttribute("aria-label") || anchor.getAttribute("title") || "");
                    if (!text || badLabel(text, href)) continue;
                    const haystack = `${text} ${href}`.toLowerCase();
                    let score = 0;
                    for (const token of runnableTokens) {
                      if (haystack.includes(token)) score += token === "try" || token === "demo" || token === "run" ? 4 : 3;
                    }
                    const container = anchor.closest('main,article,section,[role="main"],div[class*="example" i],div[class*="demo" i],div[class*="preview" i],div[class*="try" i]') || anchor.parentElement || anchor;
                    const containerText = lower(container.innerText || container.textContent || "");
                    if (wantedContext && containerText.includes(wantedContext)) score += 3;
                    if (actionTokens.some((token) => token && haystack.includes(token))) score += 1;
                    if (anchor.closest('main,article,section,[role="main"]')) score += 1;
                    if (anchor.closest('[class*="example" i],[class*="demo" i],[class*="preview" i],[class*="try" i]')) score += 2;
                    if (text.length > 80) score -= 1;
                    if (score >= 4) candidates.push({ href, text, score });
                  }
                  candidates.sort((a, b) => b.score - a.score);
                  return candidates[0] || null;
                }
                """,
                {"contextText": str(context_text or ""), "actionName": str(action_name or "")},
            )
        except Exception:
            return ""
        if not isinstance(payload, dict):
            return ""
        return str(payload.get("href") or "").strip()

    @classmethod
    async def _click_row_action_control(cls, selected) -> None:
        try:
            await selected.click()
            return
        except Exception as click_error:
            if not cls._row_action_click_can_use_programmatic_fallback(str(click_error)):
                raise
            last_error = click_error
        try:
            await selected.click(force=True, timeout=5000)
            return
        except TypeError:
            pass
        except Exception as force_error:
            last_error = force_error
        if hasattr(selected, "evaluate"):
            try:
                await selected.evaluate("(el) => el.click()")
                return
            except Exception as evaluate_error:
                last_error = evaluate_error
        raise last_error

    @staticmethod
    def _row_action_click_can_use_programmatic_fallback(message: str) -> bool:
        text = str(message or "").casefold()
        return any(
            marker in text
            for marker in (
                "outside of the viewport",
                "not visible",
                "not stable",
                "intercepts pointer events",
                "element is detached",
                "timeout",
            )
        )

    @staticmethod
    def _row_context_for_ref(*, page, row_ref: dict[str, Any]):
        frame_index = row_ref.get("frame_index")
        if isinstance(frame_index, int):
            frames = getattr(page, "frames", None)
            if isinstance(frames, list) and 0 <= frame_index < len(frames):
                return frames[frame_index]
        return page

    @staticmethod
    async def _row_locator_for_ref(*, row_context, selector: str, row_ref: dict[str, Any]):
        locator = row_context.locator(selector)
        text = str(row_ref.get("text", "") or "").strip()
        if text and hasattr(locator, "filter"):
            try:
                filtered = locator.filter(has_text=text)
                if await filtered.count() > 0:
                    return filtered.first
            except Exception:
                pass
        return locator.first

    async def visual_observe(self, page, args, runtime_state=None):
        snapshot = await self.observe_page(page, args, runtime_state)
        svg_summary = await self._extract_svg_summary(page=page)
        if runtime_state is not None:
            runtime_state["last_visual_summary"] = svg_summary
            self._mark_used_skill(runtime_state, "visual_svg_recognition")
        return {**snapshot, "svg_summary": svg_summary}

    async def visual_extract_object_count(self, page, args, runtime_state=None):
        target = str(args.get("object", args.get("shape", args.get("target", "")))).strip().lower()
        if not target or target in {"__unspecified__", "count"}:
            raise StructuredExtractionError(
                code="visual_target_not_specified",
                message="visual object count requires a generic target such as link, button, item, or visible object type.",
                details={"target": target},
            )
        summary = (runtime_state or {}).get("last_visual_summary")
        if not isinstance(summary, dict):
            summary = await self._extract_svg_summary(page=page)
        counts = summary.get("shape_counts", {}) if isinstance(summary, dict) else {}
        count = counts.get(target)
        if count is None and target.endswith("s"):
            count = counts.get(target[:-1])
        if count is None:
            dom_count = await self._count_visible_dom_objects_by_geometry(page=page, target=target, args=args)
            if dom_count is not None:
                self._mark_used_skill(runtime_state, "visual_dom_geometry")
                args["_executor_note"] = f"visual_extract_object_count target={target!r}; dom_geometry_count={dom_count}"
                return dom_count
        if count is None:
            raise StructuredExtractionError(
                code="visual_spatial_no_decision",
                message="visual controller required: object count is not exposed by reliable SVG/DOM geometry.",
                details={"target": target, "available_counts": counts},
            )
        self._mark_used_skill(runtime_state, "visual_svg_recognition")
        args["_executor_note"] = f"visual_extract_object_count target={target!r}; count={count}"
        return count

    async def visual_click_by_geometry(self, page, args, runtime_state=None):
        if "x" not in args or "y" not in args:
            raise StructuredExtractionError(
                code="visual_spatial_no_decision",
                message="visual controller required: missing reliable x/y coordinates.",
                details={"reason": "missing_coordinates"},
            )
        await page.mouse.click(float(args["x"]), float(args["y"]))
        self._mark_used_skill(runtime_state, "visual_click_by_geometry")
        return {"clicked": True, "x": float(args["x"]), "y": float(args["y"])}

    async def extract_pattern_from_page_text(self, page, args, runtime_state=None):
        self._mark_used_skill(runtime_state, "numeric_extraction")
        flags_value = 0
        for token in str(args.get("flags", "")).split("|"):
            key = token.strip().upper()
            if not key:
                continue
            flags_value |= getattr(re, key, 0)

        pattern = args["pattern"]
        occurrence = int(args.get("occurrence", 1))
        limit = args.get("limit")
        fields = args.get("fields")
        group_index = args.get("group_index")
        normalize_number = bool(args.get("normalize_number", False))
        number_type = str(args.get("number_type", "int")).lower()
        strip_plus = bool(args.get("strip_plus", True))

        source_text = await self._load_source_text(page=page, runtime_state=runtime_state, force_refresh=True)

        matches = list(re.finditer(pattern, source_text, flags=flags_value))
        if not matches:
            self._record_pattern_artifact(
                runtime_state=runtime_state,
                pattern=pattern,
                success=False,
                reason="pattern_not_found",
            )
            raise ValueError(f"Pattern not found: {pattern}")
        if occurrence < 1 or occurrence > len(matches):
            self._record_pattern_artifact(
                runtime_state=runtime_state,
                pattern=pattern,
                success=False,
                reason=f"occurrence_out_of_range:{occurrence}/{len(matches)}",
            )
            raise ValueError(f"Occurrence {occurrence} out of range, found {len(matches)} matches")

        match = matches[occurrence - 1]
        if limit is not None:
            extracted_items = self._extract_pattern_repeated(
                matches=matches,
                limit=int(limit),
                fields=fields,
                group_index=group_index,
            )
            args["_executor_note"] = (
                f"extract_pattern_from_page_text matched pattern={pattern!r}; "
                f"returned {len(extracted_items)} repeated item(s)"
            )
            return extracted_items

        extracted_value = self._extract_match_value(match, group_index=group_index)
        benchmark_context = runtime_state.get("benchmark_context", {}) if isinstance(runtime_state, dict) else {}
        task_family = str(benchmark_context.get("task_family", "")).strip().lower()
        if task_family == "negative_or_ambiguous_case" and self._looks_like_broad_prose_match(
            value=extracted_value,
            pattern=str(pattern),
        ):
            raise ValueError(
                "Weak regex extraction: matched broad prose fragment instead of a concrete value token"
            )

        normalized_value = None
        if normalize_number:
            normalized_value = self._normalize_number_token(
                extracted_value,
                number_type=number_type,
                strip_plus=strip_plus,
            )
            result_value = normalized_value
        else:
            result_value = extracted_value

        self._record_pattern_artifact(
            runtime_state=runtime_state,
            pattern=pattern,
            success=True,
            raw_match=extracted_value,
            normalized_value=normalized_value,
            group_index=group_index,
            occurrence=occurrence,
        )
        args["_executor_note"] = (
            f"extract_pattern_from_page_text matched pattern={pattern!r}; "
            f"raw_match={extracted_value!r}; normalized_value={normalized_value!r}"
        )
        return result_value

    @classmethod
    def _extract_pattern_repeated(
        cls,
        *,
        matches: list[re.Match[str]],
        limit: int,
        fields: Any,
        group_index: int | None,
    ) -> list[Any]:
        if limit < 1:
            return []
        repeated = matches[:limit]

        if fields is not None:
            return [cls._build_structured_match_item(match=match, fields=fields) for match in repeated]

        return [cls._extract_match_value(match, group_index=group_index) for match in repeated]

    @classmethod
    def _build_structured_match_item(cls, *, match: re.Match[str], fields: Any) -> dict[str, Any]:
        item: dict[str, Any] = {}
        if isinstance(fields, dict):
            field_entries = fields.items()
        else:
            field_entries = [(field_name, idx + 1) for idx, field_name in enumerate(list(fields))]

        for default_group_index, (field_name, spec) in enumerate(field_entries, start=1):
            field = str(field_name)
            extracted = cls._extract_match_group_by_field(match=match, field=field, spec=spec, default_index=default_group_index)
            item[field] = extracted
        return item

    @classmethod
    def _extract_match_group_by_field(
        cls,
        *,
        match: re.Match[str],
        field: str,
        spec: Any,
        default_index: int,
    ) -> Any:
        if isinstance(spec, dict):
            group = int(spec.get("group_index", default_index))
            normalize_number = bool(spec.get("normalize_number", False))
            number_type = str(spec.get("number_type", "int")).lower()
            strip_plus = bool(spec.get("strip_plus", True))
        else:
            group = int(spec) if isinstance(spec, int) else default_index
            normalize_number = False
            number_type = "int"
            strip_plus = True

        extracted = cls._extract_match_value(match, group_index=group)
        if normalize_number:
            try:
                return cls._normalize_number_token(
                    extracted,
                    number_type=number_type,
                    strip_plus=strip_plus,
                )
            except ValueError:
                return extracted
        return extracted

    async def extract_text_near_text(self, page, args, runtime_state=None):
        self._mark_used_skill(runtime_state, "extract_value_near_anchor")
        anchor_text = str(args["anchor_text"])
        pattern = str(args["pattern"])
        window_chars = int(args.get("window_chars", 200))
        flags_value = re.IGNORECASE if bool(args.get("ignore_case", True)) else 0

        source_text = await self._load_source_text(page=page, runtime_state=runtime_state, force_refresh=True)
        anchor_match = re.search(re.escape(anchor_text), source_text, flags=flags_value)
        if not anchor_match:
            raise ValueError(f"Anchor text not found: {anchor_text}")

        start = max(0, anchor_match.start() - window_chars)
        end = min(len(source_text), anchor_match.end() + window_chars)
        window_text = source_text[start:end]

        match = re.search(pattern, window_text, flags=flags_value)
        if not match:
            raise ValueError(
                f"Pattern not found near anchor_text={anchor_text!r} within {window_chars} chars"
            )

        group_index = args.get("group_index")
        extracted_value = self._extract_match_value(match, group_index=group_index)

        normalized_value = None
        if bool(args.get("normalize_number", False)):
            normalized_value = self._normalize_number_token(
                extracted_value,
                number_type=str(args.get("number_type", "int")).lower(),
                strip_plus=bool(args.get("strip_plus", True)),
            )
            args["_executor_note"] = (
                f"extract_text_near_text matched near anchor={anchor_text!r}; "
                f"raw_match={extracted_value!r}; normalized_value={normalized_value!r}"
            )
            return normalized_value

        args["_executor_note"] = f"extract_text_near_text matched near anchor={anchor_text!r}; raw={extracted_value!r}"
        return extracted_value

    async def extract_value_near_anchor(self, page, args, runtime_state=None):
        self._mark_used_skill(runtime_state, "extract_value_near_anchor")
        anchor_text = str(args.get("anchor_text", "")).strip()
        anchor_candidates = [str(item).strip() for item in args.get("anchor_candidates", []) if str(item).strip()]
        anchor_matching_mode = str(args.get("anchor_matching_mode", "auto")).strip().lower()
        page_language = str(args.get("page_language", "")).strip().lower()
        value_pattern = args.get("value_pattern")
        value_type = str(args.get("value_type", "")).strip().lower()
        source_text = await self._load_source_text(page=page, runtime_state=runtime_state, force_refresh=True)
        visible_anchor_texts = await self._collect_visible_anchor_texts(page)
        effective_page_language = await self._resolve_page_language(
            page=page,
            source_text=source_text,
            visible_anchor_texts=visible_anchor_texts,
            provided_language=page_language,
        )
        if not value_pattern:
            value_pattern = self._resolve_value_pattern(value_type)
        if anchor_matching_mode not in {"auto", "exact", "contains"}:
            anchor_matching_mode = "auto"
        provided_candidates_ranked = list(
            dict.fromkeys(
                [*([anchor_text] if anchor_text else []), *anchor_candidates]
            )
        )
        resolved_candidates = list(provided_candidates_ranked)
        if resolved_candidates:
            args["anchor_candidates"] = resolved_candidates
        if resolved_candidates:
            enforce_anchor_language_filter = bool(args.get("enforce_anchor_language_filter", True))
            resolved_anchor_text = ""
            if provided_candidates_ranked:
                try:
                    resolved_anchor_text = await self._resolve_anchor_text(
                        page=page,
                        preferred_anchor="",
                        anchor_candidates=provided_candidates_ranked,
                        anchor_matching_mode=anchor_matching_mode,
                        page_language=effective_page_language,
                        enforce_anchor_language_filter=False,
                        value_pattern=str(value_pattern) if value_pattern else None,
                        runtime_state=runtime_state,
                    )
                except ValueError:
                    resolved_anchor_text = ""

            if not resolved_anchor_text:
                plain_match = self._plain_text_anchor_fallback(
                    source_text=source_text,
                    anchor_candidates=resolved_candidates,
                    value_pattern=str(value_pattern) if value_pattern else "",
                    search_direction=str(args.get("search_direction", "after")).lower(),
                    required_right_context=args.get("required_right_context"),
                    required_left_context=args.get("required_left_context"),
                    group_index=args.get("group_index"),
                    flags_value=re.IGNORECASE if bool(args.get("ignore_case", True)) else 0,
                )
                if plain_match is None:
                    raise ValueError(f"Anchor text not found for candidates={resolved_candidates}")
                anchor_text = plain_match["anchor_text"]
                args["_page_text_anchor_fallback_match"] = plain_match
                args["anchor_text"] = anchor_text
                args["_executor_note"] = "extract_value_near_anchor resolved via page_text_anchor_fallback"
                return self._finalize_anchor_fallback_value(plain_match, args)
            anchor_text = resolved_anchor_text
            args["anchor_text"] = anchor_text
        if not value_pattern:
            raise ValueError("extract_value_near_anchor requires value_pattern or supported value_type")
        value_pattern = str(value_pattern)
        prefer_local_match = bool(args.get("prefer_local_match", False))
        search_direction = str(args.get("search_direction", "after")).lower()
        same_block_only = bool(args.get("same_block_only", True))
        required_right_context = args.get("required_right_context")
        required_left_context = args.get("required_left_context")
        max_distance_chars = args.get("max_distance_chars")
        group_index = args.get("group_index")
        normalize_number = bool(args.get("normalize_number", False))
        number_type = args.get("number_type")
        strip_plus = bool(args.get("strip_plus", False))
        flags_value = re.IGNORECASE if bool(args.get("ignore_case", True)) else 0

        if search_direction not in {"after", "before", "around"}:
            raise ValueError(f"Unsupported search_direction: {search_direction}")

        if required_right_context is not None:
            required_right_context = str(required_right_context)
        if required_left_context is not None:
            required_left_context = str(required_left_context)
        if max_distance_chars is not None:
            max_distance_chars = int(max_distance_chars)
        elif prefer_local_match:
            max_distance_chars = 280

        candidates = await self._collect_anchor_candidates(
            page=page,
            anchor_text=anchor_text,
            search_direction=search_direction,
            same_block_only=same_block_only,
            max_distance_chars=max_distance_chars,
            matching_mode=anchor_matching_mode,
            runtime_state=runtime_state,
        )

        fallback_used = False
        best_match = self._select_best_value_near_anchor(
            candidates=candidates,
            value_pattern=value_pattern,
            search_direction=search_direction,
            required_right_context=required_right_context,
            required_left_context=required_left_context,
            max_distance_chars=max_distance_chars,
            flags_value=flags_value,
            group_index=group_index,
            prefer_local_match=prefer_local_match,
        )

        strict_context_disabled = False
        if best_match is None and (required_left_context or required_right_context):
            strict_context_disabled = True
            best_match = self._select_best_value_near_anchor(
                candidates=candidates,
                value_pattern=value_pattern,
                search_direction=search_direction,
                required_right_context=None,
                required_left_context=None,
                max_distance_chars=max_distance_chars,
                flags_value=flags_value,
                group_index=group_index,
                prefer_local_match=prefer_local_match,
            )

        if best_match is None and same_block_only:
            fallback_used = True
            relaxed_candidates = await self._collect_anchor_candidates(
                page=page,
                anchor_text=anchor_text,
                search_direction=search_direction,
                same_block_only=False,
                max_distance_chars=max_distance_chars,
                matching_mode=anchor_matching_mode,
                runtime_state=runtime_state,
            )
            best_match = self._select_best_value_near_anchor(
                candidates=relaxed_candidates,
                value_pattern=value_pattern,
                search_direction=search_direction,
                required_right_context=required_right_context,
                required_left_context=required_left_context,
                max_distance_chars=max_distance_chars,
                flags_value=flags_value,
                group_index=group_index,
                prefer_local_match=prefer_local_match,
            )
            if best_match is None and (required_left_context or required_right_context):
                strict_context_disabled = True
                best_match = self._select_best_value_near_anchor(
                    candidates=relaxed_candidates,
                    value_pattern=value_pattern,
                    search_direction=search_direction,
                    required_right_context=None,
                    required_left_context=None,
                    max_distance_chars=max_distance_chars,
                    flags_value=flags_value,
                    group_index=group_index,
                    prefer_local_match=prefer_local_match,
                )

        if best_match is None:
            plain_match = self._plain_text_anchor_fallback(
                source_text=source_text,
                anchor_candidates=[anchor_text],
                value_pattern=value_pattern,
                search_direction=search_direction,
                required_right_context=required_right_context,
                required_left_context=required_left_context,
                group_index=group_index,
                flags_value=flags_value,
            )
            if plain_match is not None:
                args["_page_text_anchor_fallback_match"] = plain_match
                args["_executor_note"] = "extract_value_near_anchor matched via page_text_anchor_fallback"
                return self._finalize_anchor_fallback_value(plain_match, args)
            raise ValueError(
                f"Value not found near anchor_text={anchor_text!r}; pattern={value_pattern!r}; "
                f"required_left_context={required_left_context!r}; required_right_context={required_right_context!r}"
            )
        extracted_value = best_match["value"]
        if normalize_number:
            result = self._normalize_number_token(
                extracted_value,
                number_type=str(number_type or "int").lower(),
                strip_plus=strip_plus,
            )
        else:
            result = extracted_value

        args["_executor_note"] = (
            f"extract_value_near_anchor matched near anchor={anchor_text!r}; "
            f"raw_match={extracted_value!r}; distance={best_match['distance']}; "
            f"source={best_match['source']}; fallback_used={fallback_used}; "
            f"scope={best_match.get('match_scope')}; confidence={best_match.get('confidence')}; "
            f"strict_context_disabled={strict_context_disabled}"
        )
        return result

    @classmethod
    def _normalize_anchor_fallback_text(cls, value: str) -> str:
        return re.sub(r"[ \t\r\n\u00A0\u202F]+", " ", str(value or "")).strip()

    @classmethod
    def _plain_text_anchor_fallback(
        cls,
        *,
        source_text: str,
        anchor_candidates: list[str],
        value_pattern: str,
        search_direction: str,
        required_right_context: str | None,
        required_left_context: str | None,
        group_index: int | None,
        flags_value: int,
    ) -> dict[str, Any] | None:
        if not source_text or not value_pattern:
            return None
        normalized = cls._normalize_anchor_fallback_text(source_text)
        window_chars = 800
        for anchor in anchor_candidates or []:
            anchor_text = cls._normalize_anchor_fallback_text(str(anchor))
            if not anchor_text:
                continue
            match_anchor = re.search(re.escape(anchor_text), normalized, flags=flags_value | re.IGNORECASE)
            if not match_anchor:
                continue
            if search_direction == "before":
                start = max(0, match_anchor.start() - window_chars)
                end = match_anchor.end()
            elif search_direction == "around":
                start = max(0, match_anchor.start() - window_chars)
                end = min(len(normalized), match_anchor.end() + window_chars)
            else:
                start = match_anchor.start()
                end = min(len(normalized), match_anchor.end() + window_chars)
            window = normalized[start:end]
            for value_match in re.finditer(value_pattern, window, flags=flags_value):
                raw_value = cls._extract_match_value(value_match, group_index=group_index)
                value_start = value_match.start()
                value_end = value_match.end()
                left = window[max(0, value_start - 80):value_start]
                right = window[value_end:min(len(window), value_end + 80)]
                if required_left_context and cls._normalize_anchor_fallback_text(str(required_left_context)).casefold() not in left.casefold():
                    continue
                if required_right_context and cls._normalize_anchor_fallback_text(str(required_right_context)).casefold() not in right.casefold():
                    continue
                return {
                    "source": "page_text_anchor_fallback",
                    "fallback_used": True,
                    "anchor_text": anchor_text,
                    "value": raw_value,
                    "window_text": window[:500],
                }
        return None

    def _finalize_anchor_fallback_value(self, match: dict[str, Any], args: dict) -> Any:
        raw_value = match["value"]
        if bool(args.get("normalize_number", False)):
            result = self._normalize_number_token(
                raw_value,
                number_type=str(args.get("number_type") or "int").lower(),
                strip_plus=bool(args.get("strip_plus", False)),
            )
        else:
            result = raw_value
        args["_executor_note"] = (
            f"extract_value_near_anchor matched near anchor={match.get('anchor_text')!r}; "
            f"raw_match={raw_value!r}; source=page_text_anchor_fallback; fallback_used=True"
        )
        return result

    async def _resolve_anchor_text(
        self,
        *,
        page,
        preferred_anchor: str,
        anchor_candidates: list[str],
        anchor_matching_mode: str,
        page_language: str,
        enforce_anchor_language_filter: bool,
        value_pattern: str | None,
        runtime_state=None,
    ) -> str:
        source_text = await self._load_source_text(page=page, runtime_state=runtime_state, force_refresh=True)
        visible_anchors = await self._collect_visible_anchor_texts(page)
        ranked = [preferred_anchor] + anchor_candidates if preferred_anchor else list(anchor_candidates)
        ranked = [item for item in dict.fromkeys(ranked) if str(item).strip()]
        score_best: tuple[int, str] | None = None

        for candidate in ranked:
            visible_match = self._find_visible_anchor_match(
                visible_anchors=visible_anchors,
                candidate=candidate,
                matching_mode=anchor_matching_mode,
            )
            candidate_to_use = visible_match or candidate
            if not self._anchor_present(
                source_text=source_text,
                visible_anchors=visible_anchors,
                candidate=candidate_to_use,
                matching_mode=anchor_matching_mode,
                page_language=page_language,
                enforce_language_filter=enforce_anchor_language_filter,
            ):
                continue
            if not value_pattern:
                return candidate_to_use
            candidate_score = await self._score_anchor_candidate_block_match(
                page=page,
                anchor_text=candidate_to_use,
                value_pattern=value_pattern,
            )
            if score_best is None or candidate_score > score_best[0]:
                score_best = (candidate_score, candidate_to_use)

        if score_best is not None:
            return score_best[1]
        raise ValueError(f"Anchor text not found for candidates={ranked}")

    @staticmethod
    def _find_visible_anchor_match(*, visible_anchors: list[str], candidate: str, matching_mode: str) -> str | None:
        needle = str(candidate).strip().lower()
        if not needle:
            return None
        for anchor in visible_anchors:
            normalized = str(anchor).strip()
            if not normalized:
                continue
            haystack = normalized.lower()
            if matching_mode == "exact" and haystack == needle:
                return normalized
            if matching_mode == "contains" and needle in haystack:
                return normalized
            if matching_mode == "auto" and (needle in haystack or (haystack in needle and len(haystack) >= 4)):
                return normalized
        return None

    async def _score_anchor_candidate_block_match(self, *, page, anchor_text: str, value_pattern: str) -> int:
        candidates = await self._collect_anchor_candidates(
            page=page,
            anchor_text=anchor_text,
            search_direction="around",
            same_block_only=True,
            max_distance_chars=None,
        )
        if not candidates:
            return 0
        score = 0
        for candidate in candidates:
            scope_text = str(candidate.get("scope_text") or "")
            if re.search(value_pattern, scope_text, flags=re.IGNORECASE):
                score += 2
            elif re.search(value_pattern, str(candidate.get("window_text") or ""), flags=re.IGNORECASE):
                score += 1
        return score

    def _detect_script_language(self, text: str) -> str:
        latin = len(re.findall(r"[A-Za-z]", text or ""))
        cyrillic = len(re.findall(r"[А-Яа-яЁё]", text or ""))
        total = latin + cyrillic
        if not total:
            return ""
        dominant_ratio = max(latin, cyrillic) / total
        if (latin == 0 and cyrillic >= 6) or (cyrillic > 0 and dominant_ratio >= 0.6 and cyrillic > latin):
            return "ru"
        if (cyrillic == 0 and latin >= 6) or (latin > 0 and dominant_ratio >= 0.6 and latin > cyrillic):
            return "en"
        return ""

    async def _resolve_page_language(
        self,
        *,
        page,
        source_text: str,
        visible_anchor_texts: list[str],
        provided_language: str,
    ) -> str:
        detected_from_source = self._detect_script_language(source_text)
        if detected_from_source:
            return detected_from_source

        detected_from_anchors = self._detect_script_language(" ".join(visible_anchor_texts or []))
        if detected_from_anchors:
            return detected_from_anchors

        html_lang = ""
        try:
            lang_attr = await page.evaluate(
                """
                () => (document.documentElement && document.documentElement.lang) || ""
                """
            )
            html_lang = str(lang_attr or "").strip().lower()
        except Exception:
            html_lang = ""

        if html_lang.startswith("en"):
            return "en"
        if html_lang.startswith("ru"):
            return "ru"

        # provided_language is only a weak fallback hint.
        provided = str(provided_language or "").strip().lower()
        if provided.startswith("en"):
            return "en"
        if provided.startswith("ru"):
            return "ru"
        return ""

    async def _collect_visible_anchor_texts(self, page) -> list[str]:
        anchors = await page.evaluate(
            """
            () => {
              const nodes = document.querySelectorAll("a, button, dt, dd, th, td, h1, h2, h3, h4, label, p, li, span");
              const result = [];
              for (const node of nodes) {
                const text = (node.innerText || node.textContent || "").replace(/\\s+/g, " ").trim();
                if (!text) continue;
                result.push(text);
                if (result.length >= 300) break;
              }
              return result;
            }
            """
        )
        return [item.strip() for item in anchors if isinstance(item, str) and item.strip()]

    @classmethod
    def _anchor_present(
        cls,
        *,
        source_text: str,
        visible_anchors: list[str],
        candidate: str,
        matching_mode: str,
        page_language: str,
        enforce_language_filter: bool,
    ) -> bool:
        normalized_candidate = candidate.strip()
        if not normalized_candidate:
            return False
        if enforce_language_filter:
            if page_language in {"en", "english"} and cls._contains_cyrillic(normalized_candidate):
                return False
            if page_language in {"ru", "russian"} and cls._contains_latin(normalized_candidate):
                return False

        candidate_lower = normalized_candidate.lower()
        corpus = [source_text] + visible_anchors
        for text in corpus:
            haystack = str(text).lower()
            if matching_mode == "exact" and candidate_lower == haystack:
                return True
            if matching_mode == "contains" and candidate_lower in haystack:
                return True
            if matching_mode == "auto":
                if candidate_lower in haystack:
                    return True
                if haystack in candidate_lower and len(haystack) >= 4:
                    return True
        return False

    @staticmethod
    def _contains_cyrillic(text: str) -> bool:
        return bool(re.search(r"[А-Яа-яЁё]", text))

    @staticmethod
    def _contains_latin(text: str) -> bool:
        return bool(re.search(r"[A-Za-z]", text))

    @staticmethod
    def _resolve_value_pattern(value_type: str) -> str | None:
        if value_type in {"integer", "count", "number"}:
            return r"([0-9][0-9\s,\.\u00A0\u202F\+]*)"
        if value_type in {"decimal", "float"}:
            return r"([0-9]+(?:[.,][0-9]+)?)"
        if value_type == "email":
            return r"([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,63})"
        if value_type == "phone":
            return r"(\+?\d[\d\-\u2010-\u2015\(\)\s\.]{6,}\d)"
        if value_type == "url":
            return r"(https?://[^\s<>'\"]+)"
        return None

    @staticmethod
    def _looks_like_broad_prose_match(*, value: Any, pattern: str) -> bool:
        text = str(value or "").strip()
        if len(text) < 90:
            return False
        has_capture_group = False
        try:
            has_capture_group = re.compile(pattern).groups > 0
        except re.error:
            has_capture_group = False
        word_count = len(re.findall(r"\b\w+\b", text))
        sentence_like = bool(re.search(r"[.;:]\s", text))
        return word_count >= 14 and sentence_like and not has_capture_group

    async def _collect_visible_links_generic(self, *, page, limit: int) -> list[dict[str, Any]]:
        payload = await page.evaluate(
            """
            ({ limit }) => {
              const norm = (value) => String(value || "").replace(/\\s+/g, " ").trim();
              const isVisible = (el) => {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style && style.visibility !== "hidden" && style.display !== "none" && Number(style.opacity || 1) !== 0 && rect.width > 0 && rect.height > 0;
              };
              const cssEscape = (value) => {
                if (window.CSS && CSS.escape) return CSS.escape(value);
                return String(value).replace(/[^a-zA-Z0-9_-]/g, "\\\\$&");
              };
              const cssPath = (el) => {
                if (el.id) return `#${cssEscape(el.id)}`;
                const parts = [];
                let cur = el;
                while (cur && cur.nodeType === Node.ELEMENT_NODE && cur !== document.body && parts.length < 6) {
                  const tag = cur.tagName.toLowerCase();
                  let part = tag;
                  const cls = Array.from(cur.classList || []).slice(0, 2).map(cssEscape);
                  if (cls.length) part += "." + cls.join(".");
                  const parent = cur.parentElement;
                  if (parent) {
                    const siblings = Array.from(parent.children).filter((sib) => sib.tagName === cur.tagName);
                    if (siblings.length > 1) part += `:nth-of-type(${siblings.indexOf(cur) + 1})`;
                  }
                  parts.unshift(part);
                  cur = parent;
                }
                return parts.join(" > ");
              };
              return Array.from(document.querySelectorAll("a[href]"))
                .filter(isVisible)
                .slice(0, limit)
                .map((el) => {
                  const rect = el.getBoundingClientRect();
                  return {
                    text: norm(el.innerText || el.textContent || el.getAttribute("aria-label") || el.getAttribute("title") || ""),
                    href: el.href || el.getAttribute("href") || "",
                    selector: cssPath(el),
                    bbox: {
                      x: rect.x,
                      y: rect.y,
                      width: rect.width,
                      height: rect.height,
                      center_x: rect.x + rect.width / 2,
                      center_y: rect.y + rect.height / 2
                    },
                    viewport: { width: window.innerWidth, height: window.innerHeight }
                  };
                })
                .filter((item) => item.text && item.href);
            }
            """,
            {"limit": max(limit, 1)},
        )
        return [dict(item) for item in payload or [] if isinstance(item, dict)]







    async def _collect_result_like_items_generic(self, *, page, limit: int) -> list[dict[str, Any]]:
        payload = await page.evaluate(
            """
            ({ limit }) => {
              const norm = (value) => String(value || "").replace(/\\s+/g, " ").trim();
              const current = new URL(window.location.href);
              const query = norm(
                current.searchParams.get("q")
                || current.searchParams.get("query")
                || current.searchParams.get("search")
                || ""
              ).toLowerCase();
              const queryTerms = query.split(/[^\\p{L}\\p{N}_-]+/u).filter((term) => term.length >= 3);
              const isVisible = (el) => {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style && style.visibility !== "hidden" && style.display !== "none" && Number(style.opacity || 1) !== 0 && rect.width > 0 && rect.height > 0;
              };
              const deepQueryAll = (selector, root = document, out = [], seen = new Set()) => {
                let nodes = [];
                try {
                  nodes = Array.from(root.querySelectorAll(selector));
                } catch (_) {
                  nodes = [];
                }
                for (const node of nodes) {
                  if (!seen.has(node)) {
                    seen.add(node);
                    out.push(node);
                  }
                }
                let all = [];
                try {
                  all = Array.from(root.querySelectorAll("*"));
                } catch (_) {
                  all = [];
                }
                for (const node of all) {
                  if (node.shadowRoot) deepQueryAll(selector, node.shadowRoot, out, seen);
                }
                return out;
              };
              const inIgnoredRegion = (el) => !!el.closest('nav, header, footer, aside, [role="navigation"], [aria-hidden="true"], .breadcrumbs, .breadcrumb, .a11y-menu, [class*="footer" i], [class*="menu" i]');
              const cssEscape = (value) => {
                if (window.CSS && CSS.escape) return CSS.escape(value);
                return String(value).replace(/[^a-zA-Z0-9_-]/g, "\\\\$&");
              };
              const cssPath = (el) => {
                if (!el || !el.tagName) return "";
                if (el.id) return `#${cssEscape(el.id)}`;
                const parts = [];
                let cur = el;
                while (cur && cur.nodeType === Node.ELEMENT_NODE && cur !== document.body && parts.length < 6) {
                  const tag = cur.tagName.toLowerCase();
                  let part = tag;
                  const cls = Array.from(cur.classList || []).slice(0, 2).map(cssEscape);
                  if (cls.length) part += "." + cls.join(".");
                  const parent = cur.parentElement;
                  if (parent) {
                    const siblings = Array.from(parent.children).filter((sib) => sib.tagName === cur.tagName);
                    if (siblings.length > 1) part += `:nth-of-type(${siblings.indexOf(cur) + 1})`;
                  }
                  parts.unshift(part);
                  cur = parent;
                }
                return parts.join(" > ");
              };
              const descriptionFrom = (container, title) => {
                const lines = String(container?.innerText || container?.textContent || "").split(/\\n+/).map(norm).filter(Boolean);
                for (const line of lines) {
                  if (!line || line === title || line.length < 20 || line.length > 360) continue;
                  if (/^(search|results?|login|sign in|next|previous|share|menu)$/i.test(line)) continue;
                  return line;
                }
                return "";
              };
              const samePageJump = (href) => {
                try {
                  const url = new URL(href, window.location.href);
                  return url.origin === current.origin
                    && url.pathname === current.pathname
                    && url.search === current.search
                    && !!url.hash;
                } catch (_) {
                  return false;
                }
              };
              const queryMatches = (title, description, href) => {
                if (!queryTerms.length) return true;
                let path = "";
                try {
                  const url = new URL(href, window.location.href);
                  path = decodeURIComponent(url.pathname || "");
                } catch (_) {}
                const haystack = `${title} ${description} ${path}`.toLowerCase();
                return queryTerms.some((term) => haystack.includes(term));
              };
              const uiLabel = (title) => {
                const label = norm(title).replace(/^[«‹<\\s]+|[»›>\\s]+$/g, "");
                return /^(skip to .+|search|results?|login|sign in|sign up|menu|next|previous|read more|about|careers?|advertise|community|privacy|legal|settings|rss feed)$/i.test(label);
              };
              const seen = new Set();
              const results = [];
              const resultContainers = deepQueryAll('main li, [role="main"] li, main article, [role="main"] article, main section, [role="main"] section, [class*="result" i], [class*="search-result" i], [class*="item" i]').filter((node) => isVisible(node) && !inIgnoredRegion(node));
              for (const node of resultContainers) {
                const text = norm(node.innerText || node.textContent || "");
                if (!text || text.length < 20 || text.length > 1800) continue;
                const selector = cssPath(node);
                const resultishContainer = /\b(result|search-result|list|item)\b/i.test(selector) || !!node.closest('[class*="result" i], [role="listitem"]');
                const lines = String(node.innerText || node.textContent || "").split(/\\n+/).map(norm).filter(Boolean);
                const heading = node.querySelector('h1,h2,h3,h4,[role="heading"]');
                let title = norm(heading?.innerText || heading?.textContent || "");
                if (!title) {
                  title = lines.find((line) => line.length >= 4 && line.length <= 220 && !uiLabel(line) && !/^(version|released):/i.test(line)) || "";
                }
                if (!title || uiLabel(title)) continue;
                const anchor = Array.from(node.querySelectorAll('a[href]')).find((link) => {
                  const label = norm(link.innerText || link.textContent || link.getAttribute("aria-label") || link.getAttribute("title") || "");
                  const href = link.href || link.getAttribute("href") || "";
                  return href && !samePageJump(href) && !uiLabel(label);
                });
                const href = anchor ? (anchor.href || anchor.getAttribute("href") || "") : "";
                const description = descriptionFrom(node, title);
                const key = href || `${title}|${description}`;
                if (!key || seen.has(key)) continue;
                if (!resultishContainer && !queryMatches(title, description, href)) continue;
                seen.add(key);
                results.push({
                  title,
                  text: title,
                  href,
                  link: href,
                  description,
                  selector
                });
                if (results.length >= limit) break;
              }
              if (results.length >= limit) return results;
              const anchors = deepQueryAll('main a[href], [role="main"] a[href], article a[href], section a[href], [class*="result" i] a[href], [class*="search" i] a[href], a[href]').filter((anchor) => isVisible(anchor) && !inIgnoredRegion(anchor));
              for (const anchor of anchors) {
                const href = anchor.href || anchor.getAttribute("href") || "";
                const title = norm(anchor.innerText || anchor.textContent || anchor.getAttribute("aria-label") || anchor.getAttribute("title") || "");
                if (!href || !title || title.length < 4 || title.length > 220) continue;
                if (samePageJump(href)) continue;
                if (uiLabel(title)) continue;
                if (seen.has(href)) continue;
                const container = anchor.closest('article,li,section,div[class*="result" i],div[class*="item" i],div[class*="card" i],div[class*="search" i],div') || anchor.parentElement || anchor;
                if (inIgnoredRegion(container)) continue;
                const selector = cssPath(anchor);
                const resultishSelector = /\\b(result|search|item|card)\\b/i.test(selector) || !!anchor.closest('article,[role="listitem"],[class*="result" i],[class*="search" i]');
                const description = descriptionFrom(container, title);
                if (!description && uiLabel(title)) continue;
                if (!description && !resultishSelector && !queryMatches(title, description, href)) continue;
                if (queryTerms.length && !queryMatches(title, description, href) && !description) continue;
                seen.add(href);
                results.push({
                  title,
                  text: title,
                  href,
                  link: href,
                  description,
                  selector
                });
                if (results.length >= limit) break;
              }
              return results;
            }
            """,
            {"limit": max(limit, 1)},
        )
        return [dict(item) for item in payload or [] if isinstance(item, dict)]

    async def _collect_search_results_generic(self, *, page, args: dict, runtime_state=None) -> list[dict[str, Any]]:
        limit = int(args.get("limit", 10) or 10)
        await self._settle_page_for_read(page)
        items = await self._collect_result_like_items_generic(page=page, limit=max(limit, 1))
        if not items and await self._submit_search_query_from_current_url(page=page, runtime_state=runtime_state):
            await self._settle_page_for_read(page)
            items = await self._collect_result_like_items_generic(page=page, limit=max(limit, 1))
        if not items:
            try:
                source_text = await self._load_source_text(page=page, runtime_state=runtime_state, force_refresh=True)
            except Exception:
                source_text = ""
            try:
                links = await self.extract_visible_links(page, {"limit": 80}, runtime_state)
            except Exception:
                links = []
            items = self._collect_result_like_items_from_text(
                source_text=source_text,
                links=[dict(item) for item in links if isinstance(item, dict)] if isinstance(links, list) else [],
                limit=max(limit, 1),
            )
        ranked: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            current = dict(item)
            title = str(current.get("title") or current.get("text") or "").strip()
            href = str(current.get("href") or current.get("link") or "").strip()
            description = str(current.get("description") or current.get("snippet") or "").strip()
            if not title or not href:
                continue
            if self._looks_like_search_ui_link(title=title, href=href):
                continue
            current["rank"] = index + 1
            current["score"] = self._search_result_score(current, query=self._search_query_from_url(str(getattr(page, "url", "") or "")))
            ranked.append(current)
        ranked.sort(key=lambda item: (-float(item.get("score", 0)), int(item.get("rank", 9999))))
        result = ranked[: max(limit, 1)]
        if runtime_state is not None:
            runtime_state["last_search_results"] = result
            if not result:
                runtime_state["search_result_diagnostics"] = {
                    "reason": "no_result_like_link_found",
                    "candidate_count": len(items),
                }
        if result:
            self._mark_used_skill(runtime_state, "search_result_extraction")
        return result

    async def _submit_search_query_from_current_url(self, *, page, runtime_state=None) -> bool:
        query = self._search_query_from_url(str(getattr(page, "url", "") or ""))
        if not query:
            return False
        flag = f"search_query_submitted:{str(getattr(page, 'url', '') or '')}:{query}"
        if isinstance(runtime_state, dict):
            submitted = runtime_state.setdefault("submitted_search_queries", [])
            if flag in submitted:
                return False
        selectors = [
            "main input[type='search']",
            "main input[name='q']",
            "main input[name='query']",
            "input[type='search']",
            "input[role='searchbox']",
            "input[name='q']",
            "input[name='query']",
            "input[name='search']",
            "input[placeholder*='Search' i]",
            "input[aria-label*='Search' i]",
        ]
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                if await locator.count() <= 0:
                    continue
                await locator.fill(query)
                await locator.press("Enter")
                await self._wait_after_possible_navigation(page)
                if isinstance(runtime_state, dict):
                    runtime_state["submitted_search_queries"].append(flag)
                return True
            except Exception:
                continue
        return False

    @staticmethod
    def _search_query_from_url(url: str) -> str:
        try:
            parsed = urlparse(str(url or ""))
            params = parse_qs(parsed.query)
        except Exception:
            return ""
        for key in ("q", "query", "search", "s", "keyword", "keywords", "term"):
            values = params.get(key)
            if values:
                value = str(values[0] or "").strip()
                if value:
                    return value
        return ""

    @staticmethod
    def _looks_like_search_ui_link(*, title: str, href: str) -> bool:
        label = re.sub(r"\s+", " ", str(title or "").strip().casefold())
        if not label:
            return True
        blocked_labels = {
            "search",
            "next",
            "previous",
            "prev",
            "login",
            "sign in",
            "menu",
            "filters",
            "filter",
            "settings",
            "pagination",
            "home",
        }
        if label in blocked_labels:
            return True
        try:
            path = str(href or "").casefold()
        except Exception:
            path = ""
        return any(token in path for token in ("page=", "pagination", "#"))

    @staticmethod
    def _search_result_score(item: dict[str, Any], *, query: str = "") -> float:
        score = 0.0
        if item.get("href") or item.get("link"):
            score += 2.0
        if item.get("description") or item.get("snippet"):
            score += 1.0
        if item.get("selector"):
            selector = str(item.get("selector", "")).casefold()
            if any(token in selector for token in ("result", "article", "li", "item", "card")):
                score += 1.0
        title_len = len(str(item.get("title") or item.get("text") or ""))
        if 6 <= title_len <= 160:
            score += 0.5
        query_norm = ActionHandlers._normalize_result_match_text(query)
        if query_norm:
            title_norm = ActionHandlers._normalize_result_match_text(str(item.get("title") or item.get("text") or ""))
            href = str(item.get("href") or item.get("link") or "")
            haystack = ActionHandlers._normalize_result_match_text(f"{title_norm} {item.get('description') or item.get('snippet') or ''} {href}")
            if title_norm == query_norm:
                score += 9.0
            elif title_norm.startswith(f"{query_norm} "):
                score += 6.0
            elif query_norm in title_norm:
                score += 3.0
            elif query_norm in haystack:
                score += 1.0
            if query_norm in ActionHandlers._normalized_href_segments(href):
                score += 3.0
            simple_query = not re.search(r"[\s().:#/]", query_norm)
            if simple_query and query_norm in haystack and not title_norm.startswith(query_norm):
                if re.search(r"\b(?:writing|server|servers|client|clients|tutorial|guide|how to|example|examples|constructor|method|property|event|syntax)\b", title_norm):
                    score -= 4.0
        return score

    async def _extract_anchor_object(self, *, page, args: dict, runtime_state=None) -> dict[str, Any]:
        fields = args.get("fields") if isinstance(args.get("fields"), dict) else {}
        requested = [str(field).strip() for field in fields if str(field).strip()]
        if not requested:
            requested = ["name", "count"]
        anchors = [
            str(item).strip()
            for item in args.get("anchor_candidates", [])
            if str(item).strip()
        ] if isinstance(args.get("anchor_candidates"), list) else []
        for key in ("anchor_text", "anchor", "target", "label"):
            value = str(args.get(key, "") or "").strip()
            if value:
                anchors.append(value)
        payload = await page.evaluate(
            """
            ({ anchors, limit }) => {
              const norm = (value) => String(value || "").replace(/\\s+/g, " ").trim();
              const isVisible = (el) => {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style && style.visibility !== "hidden" && style.display !== "none"
                  && Number(style.opacity || 1) !== 0 && rect.width > 0 && rect.height > 0;
              };
              const ignored = (el) => !!el.closest('nav,header,footer,aside,[role="navigation"],[aria-hidden="true"]');
              const numberPattern = /[0-9][0-9\\s.,\\u00A0\\u202F+]*(?:\\s*(?:articles?|items?|results?|rows?))?/i;
              const cssEscape = (value) => {
                if (window.CSS && CSS.escape) return CSS.escape(value);
                return String(value).replace(/[^a-zA-Z0-9_-]/g, "\\\\$&");
              };
              const cssPath = (el) => {
                if (!el || !el.tagName) return "";
                if (el.id) return `#${cssEscape(el.id)}`;
                const parts = [];
                let cur = el;
                while (cur && cur.nodeType === Node.ELEMENT_NODE && cur !== document.body && parts.length < 6) {
                  const tag = cur.tagName.toLowerCase();
                  let part = tag;
                  const cls = Array.from(cur.classList || []).slice(0, 2).map(cssEscape);
                  if (cls.length) part += "." + cls.join(".");
                  const parent = cur.parentElement;
                  if (parent) {
                    const siblings = Array.from(parent.children).filter((sib) => sib.tagName === cur.tagName);
                    if (siblings.length > 1) part += `:nth-of-type(${siblings.indexOf(cur) + 1})`;
                  }
                  parts.unshift(part);
                  cur = parent;
                }
                return parts.join(" > ");
              };
              const wanted = (anchors || []).map((item) => norm(item).toLowerCase()).filter(Boolean);
              const candidates = Array.from(document.querySelectorAll('main a, main li, main div, article a, article li, article div, [role="main"] a, [role="main"] li, [role="main"] div, body a, body li'))
                .filter((node) => isVisible(node));
              const rows = [];
              const seen = new Set();
              for (const node of candidates) {
                const text = norm(node.innerText || node.textContent || "");
                if (!text || text.length < 3 || text.length > 500) continue;
                if (wanted.length && !wanted.some((anchor) => text.toLowerCase().includes(anchor))) continue;
                if (ignored(node) && !wanted.length) continue;
                const numberMatch = text.match(numberPattern);
                const heading = norm(node.querySelector?.('h1,h2,h3,h4,[role="heading"],a')?.innerText || node.querySelector?.('a')?.textContent || "");
                let label = heading || text.replace(numberPattern, "").trim();
                if (numberPattern.test(label)) label = norm(label.replace(numberPattern, "").trim());
                label = norm(label.split(/\\n/)[0] || label);
                if (!label || label.length > 160) {
                  const lines = text.split(/\\s{2,}|\\n+/).map(norm).filter(Boolean);
                  label = lines.find((line) => !numberPattern.test(line) && line.length >= 2 && line.length <= 160) || label;
                }
                const value = norm(numberMatch ? numberMatch[0] : "");
                if (!label || !value) continue;
                const href = node.href || node.querySelector?.('a[href]')?.href || "";
                const key = `${label}|${value}|${href}`;
                if (seen.has(key)) continue;
                seen.add(key);
                rows.push({ label, name: label, title: label, value, count: value, number: value, href, link: href, selector: cssPath(node), raw_text: text });
                if (rows.length >= limit) break;
              }
              return rows;
            }
            """,
            {"anchors": anchors, "limit": int(args.get("limit", 20) or 20)},
        )
        rows = [dict(item) for item in payload or [] if isinstance(item, dict)]
        if not rows:
            raise StructuredExtractionError(
                code="anchor_object_not_found",
                message="No label/value object was found near the requested anchor.",
                details={"anchors": anchors, "requested_fields": requested},
            )
        item = rows[0]
        result: dict[str, Any] = {}
        for field in requested:
            normalized = field.casefold()
            if any(token in normalized for token in ("name", "title", "label", "language")):
                result[field] = item.get("label") or item.get("name") or item.get("title")
            elif any(token in normalized for token in ("count", "number", "total", "value")):
                result[field] = item.get("count") or item.get("value") or item.get("number")
            elif any(token in normalized for token in ("href", "url", "link")):
                result[field] = item.get("href") or item.get("link")
            else:
                result[field] = item.get(field) or item.get("value") or item.get("label")
        result["status"] = "success"
        result["found_fields"] = [field for field in requested if result.get(field) not in (None, "", [], {})]
        result["missing_fields"] = [field for field in requested if field not in result["found_fields"]]
        result["raw_item"] = item
        if runtime_state is not None:
            runtime_state["last_anchor_objects"] = rows
        self._mark_used_skill(runtime_state, "anchor_object_extraction")
        return result

    @classmethod
    def _collect_result_like_items_from_text(
        cls,
        *,
        source_text: str,
        links: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        lines = [" ".join(line.split()).strip() for line in str(source_text or "").splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            return []
        start = 0
        for idx, line in enumerate(lines):
            if re.search(r"\bresults?\b|\bрезультат", line, flags=re.IGNORECASE):
                start = idx + 1
                break
        footer_markers = {
            "privacy policy",
            "about",
            "disclaimers",
            "developers",
            "statistics",
            "mobile view",
            "see all results",
        }
        ui_pattern = re.compile(
            r"^(search|results?|advanced search|search in:|help|tools|appearance|small|standard|large|wide|automatic|light|dark|next|previous|login|log in|create account|donate)$",
            flags=re.IGNORECASE,
        )
        metadata_pattern = re.compile(
            r"\b\d+\s*(?:kb|mb|words?)\b|released?:|version:|\b\d{1,2}:\d{2}\b|\b\d{4}\b",
            flags=re.IGNORECASE,
        )
        by_text: dict[str, dict[str, Any]] = {}
        for link in links:
            text = cls._normalize_result_match_text(str(link.get("title") or link.get("text") or ""))
            if text and text not in by_text:
                by_text[text] = link

        items: list[dict[str, Any]] = []
        idx = start
        while idx < len(lines) and len(items) < max(limit, 1):
            title = lines[idx]
            folded = title.casefold()
            if any(marker == folded or marker in folded for marker in footer_markers):
                break
            idx += 1
            if ui_pattern.match(title):
                continue
            if re.match(r"^results?\s+\d", title, flags=re.IGNORECASE):
                continue
            if title.endswith(":") or len(title) < 3 or len(title) > 140:
                continue
            if metadata_pattern.search(title):
                continue
            if title.lower().startswith(("the page ", "did you mean", "jump to", "main menu")):
                continue
            description_parts: list[str] = []
            lookahead = idx
            while lookahead < len(lines) and len(description_parts) < 2:
                candidate = lines[lookahead]
                candidate_folded = candidate.casefold()
                candidate_link_key = cls._normalize_result_match_text(candidate)
                if candidate_link_key in by_text and candidate_link_key != cls._normalize_result_match_text(title):
                    break
                if ui_pattern.match(candidate) or any(marker == candidate_folded for marker in footer_markers):
                    break
                if metadata_pattern.search(candidate):
                    lookahead += 1
                    break
                if len(candidate) >= 20 and candidate != title:
                    description_parts.append(candidate)
                lookahead += 1
            if not description_parts:
                continue
            link = cls._link_for_result_title(title=title, links_by_text=by_text)
            href = str((link or {}).get("href") or (link or {}).get("link") or "")
            items.append(
                {
                    "title": title,
                    "text": title,
                    "href": href,
                    "link": href,
                    "description": " ".join(description_parts),
                    "selector": "",
                    "source": "text_result_fallback",
                }
            )
            idx = max(idx, lookahead)
        return items

    @classmethod
    def _link_for_result_title(cls, *, title: str, links_by_text: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        wanted = cls._normalize_result_match_text(title)
        if not wanted:
            return None
        if wanted in links_by_text:
            return links_by_text[wanted]
        for text, link in links_by_text.items():
            if wanted in text or text in wanted:
                return link
        return None

    @staticmethod
    def _normalize_result_match_text(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().casefold())


    @staticmethod
    def _merge_card_item_candidates(
        primary: list[dict[str, Any]],
        secondary: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in [*primary, *secondary]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("name") or item.get("text") or "").strip()
            description = str(item.get("description") or item.get("snippet") or "").strip()
            href = str(item.get("href") or item.get("link") or "").strip()
            key = href or f"{title}|{description}"
            if not key or key in seen:
                continue
            seen.add(key)
            current = dict(item)
            if title:
                current.setdefault("title", title)
                current.setdefault("name", title)
                current.setdefault("text", title)
            if description:
                current.setdefault("description", description)
                current.setdefault("snippet", description)
            if href:
                current.setdefault("href", href)
                current.setdefault("link", href)
            merged.append(current)
        return merged












    async def _collect_card_items_generic(self, *, page, args: dict, runtime_state=None) -> list[dict[str, Any]]:
        limit = int(args.get("limit", 20))
        collection_limit = max(limit * 4, limit, 80)
        numeric_value_required = bool(args.get("numeric_value_required", False))
        payload = await page.evaluate(
            """
            ({ limit, numericValueRequired }) => {
              const norm = (value) => String(value || "").replace(/\\s+/g, " ").trim();
              const numericPattern = /[-+]?\\d[\\d\\s.,]*/;
              const isVisible = (el) => {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style && style.visibility !== "hidden" && style.display !== "none" && Number(style.opacity || 1) !== 0 && rect.width > 0 && rect.height > 0;
              };
              const inIgnoredRegion = (el) => !!el.closest('nav, header, footer, aside, [role="navigation"], [aria-hidden="true"], [class*="footer" i], [class*="menu" i]');
              const hasRepeatedSiblingShape = (node) => {
                const parent = node.parentElement;
                if (!parent) return false;
                const ownClasses = Array.from(node.classList || []).filter(Boolean);
                const peers = Array.from(parent.children).filter((peer) => {
                  if (peer.tagName !== node.tagName || !isVisible(peer)) return false;
                  const peerClasses = Array.from(peer.classList || []).filter(Boolean);
                  if (!ownClasses.length) return !peerClasses.length;
                  return ownClasses.some((name) => peerClasses.includes(name));
                });
                return peers.length >= 2;
              };
              const hasNestedRepeatedChildren = (node) => {
                const nested = Array.from(node.querySelectorAll(':scope > article, :scope > li, :scope > [role="listitem"], :scope > div[class*="card" i], :scope > div[class*="item" i], :scope > div[class*="listing" i], :scope > div[class*="tile" i]'))
                  .filter((child) => isVisible(child) && !inIgnoredRegion(child));
                if (nested.length < 2) return false;
                const withLinksOrHeadings = nested.filter((child) => child.querySelector('a[href],h1,h2,h3,h4,[role="heading"]'));
                return withLinksOrHeadings.length >= 2;
              };
              const cssEscape = (value) => {
                if (window.CSS && CSS.escape) return CSS.escape(value);
                return String(value).replace(/[^a-zA-Z0-9_-]/g, "\\\\$&");
              };
              const cssPath = (el) => {
                if (!el || !el.tagName) return "";
                if (el.id) return `#${cssEscape(el.id)}`;
                const parts = [];
                let cur = el;
                while (cur && cur.nodeType === Node.ELEMENT_NODE && cur !== document.body && parts.length < 6) {
                  const tag = cur.tagName.toLowerCase();
                  let part = tag;
                  const cls = Array.from(cur.classList || []).slice(0, 2).map(cssEscape);
                  if (cls.length) part += "." + cls.join(".");
                  const parent = cur.parentElement;
                  if (parent) {
                    const siblings = Array.from(parent.children).filter((sib) => sib.tagName === cur.tagName);
                    if (siblings.length > 1) part += `:nth-of-type(${siblings.indexOf(cur) + 1})`;
                  }
                  parts.unshift(part);
                  cur = parent;
                }
                return parts.join(" > ");
              };
              const titleFrom = (node, lines) => {
                const heading = node.querySelector('h1,h2,h3,h4,[role="heading"],[class*="title" i],[class*="name" i]');
                const headingText = norm(heading?.innerText || heading?.textContent || "");
                if (headingText && headingText.length >= 3 && headingText.length <= 220) return headingText;
                const anchor = node.querySelector('a[href]');
                const anchorText = norm(anchor?.innerText || anchor?.textContent || anchor?.getAttribute("aria-label") || anchor?.getAttribute("title") || "");
                if (anchorText && anchorText.length >= 3 && anchorText.length <= 220) return anchorText;
                return lines.find((line) => line.length >= 3 && line.length <= 220 && !/^(menu|navigation|search|login|sign in|next|previous|read more)$/i.test(line)) || "";
              };
              const descriptionFrom = (lines, title) => {
                for (const line of lines) {
                  if (!line || line === title) continue;
                  if (line.length < 18 || line.length > 420) continue;
                  if (/^(menu|navigation|search|login|sign in|next|previous|read more)$/i.test(line)) continue;
                  return line;
                }
                return "";
              };
              const explicitBlocks = Array.from(document.querySelectorAll(
                'main article, [role="main"] article, main li, [role="main"] li, [role="listitem"], main div[class*="card" i], main div[class*="item" i], main div[class*="listing" i], main div[class*="tile" i], div[class*="card" i], div[class*="item" i], div[class*="listing" i], div[class*="tile" i]'
              ));
              const repeatedBlocks = Array.from(document.querySelectorAll(
                'main section, [role="main"] section, main div, [role="main"] div'
              )).filter(hasRepeatedSiblingShape);
              const candidates = Array.from(new Set([...explicitBlocks, ...repeatedBlocks]))
                .filter((node) => isVisible(node) && !inIgnoredRegion(node));
              const seen = new Set();
              const cards = [];
              for (const node of candidates) {
                if (hasNestedRepeatedChildren(node) && !node.matches('article,li,[role="listitem"],[class*="card" i],[class*="item" i],[class*="listing" i],[class*="tile" i]')) continue;
                const raw = String(node.innerText || node.textContent || "");
                const text = norm(raw);
                if (!text || text.length < 18 || text.length > 1800) continue;
                if (numericValueRequired && !numericPattern.test(text)) continue;
                const lines = raw.split(/\\n+/).map(norm).filter(Boolean);
                if (lines.length < 2 && !node.querySelector('a[href]')) continue;
                const title = titleFrom(node, lines);
                if (!title) continue;
                const description = descriptionFrom(lines, title);
                const anchor = node.querySelector('a[href]');
                const href = anchor ? (anchor.href || anchor.getAttribute('href') || "") : "";
                const numericValueText = numericValueRequired ? (lines.find((line) => numericPattern.test(line)) || "") : "";
                const imageNode = node.querySelector("img");
                const key = href || `${title}|${description}`;
                if (!key || seen.has(key)) continue;
                seen.add(key);
                cards.push({
                  title,
                  name: title,
                  description,
                  snippet: description,
                  href,
                  link: href,
                  numeric_value_text: numericValueText,
                  image: imageNode ? (imageNode.currentSrc || imageNode.src || imageNode.getAttribute("src") || "") : "",
                  selector: cssPath(node),
                  raw_text: text
                });
                if (cards.length >= Math.max(limit, 1)) break;
              }
              return cards;
            }
            """,
            {"limit": max(collection_limit, 1), "numericValueRequired": numeric_value_required},
        )
        cards = [dict(item) for item in payload or [] if isinstance(item, dict)]
        try:
            link_cards = await self._collect_result_like_items_generic(page=page, limit=max(limit * 4, limit, 80))
        except Exception:
            link_cards = []
        if link_cards:
            prioritize_link_cards = any(args.get(key) for key in ("condition", "filter", "where", "target"))
            cards = (
                self._merge_card_item_candidates(link_cards, cards)
                if prioritize_link_cards
                else self._merge_card_item_candidates(cards, link_cards)
            )
        if not cards:
            cards = await self._collect_result_like_items_generic(page=page, limit=limit)
        if not cards:
            rows = await self._collect_row_candidates_generic(page=page, limit=max(limit * 3, limit, 20))
            cards = self._card_items_from_row_candidates(rows=rows, limit=limit, numeric_value_required=numeric_value_required)
        cards = self._remove_shared_item_descriptions(cards)
        cards = self._rank_module_like_items_if_requested(cards=cards, args=args, runtime_state=runtime_state)
        if cards:
            self._mark_used_skill(runtime_state, "row_list_extraction")
        if numeric_value_required:
            for item in cards:
                numeric_value = self._normalize_numeric_token(str(item.get("numeric_value_text", "")))
                if numeric_value is not None:
                    item["numeric_value"] = numeric_value
            upper_bound = self._extract_numeric_upper_bound(args=args, runtime_state=runtime_state)
            if upper_bound is not None:
                numeric_items = [item for item in cards if isinstance(item.get("numeric_value"), (int, float))]
                if numeric_items:
                    cards = [item for item in numeric_items if float(item["numeric_value"]) <= float(upper_bound)]
                elif cards and runtime_state is not None:
                    runtime_state["partial_success_reason"] = "numeric_filter_not_applied"
        fields = args.get("fields") if isinstance(args.get("fields"), dict) else {}
        if fields and not bool(args.get("_defer_projection")):
            cards = [self._project_item_to_schema(item=item, fields=fields) for item in cards]
        return cards[: max(limit, 1)]

    @classmethod
    def _rank_module_like_items_if_requested(
        cls,
        *,
        cards: list[dict[str, Any]],
        args: dict[str, Any],
        runtime_state=None,
    ) -> list[dict[str, Any]]:
        if not cards:
            return cards
        fields = args.get("fields") if isinstance(args.get("fields"), dict) else {}
        goal = str((runtime_state or {}).get("user_goal", "") or "")
        combined = " ".join(
            [
                goal,
                str(args.get("output_key", "") or ""),
                str(args.get("target", "") or ""),
                *[str(field or "") for field in fields.keys()],
            ]
        )
        text = casefold_with_mojibake_repairs(combined)
        if not any(token in text for token in ("module", "modules", "модул")):
            return cards
        ranked: list[dict[str, Any]] = []
        for index, item in enumerate(cards):
            if not isinstance(item, dict):
                continue
            current = cls._normalize_module_like_item(dict(item))
            score = cls._module_like_item_score(current)
            current["_module_score"] = score
            current["_module_rank"] = index
            ranked.append(current)
        strong = [item for item in ranked if float(item.get("_module_score", 0)) >= 3.0]
        if len(strong) >= 3:
            ranked = strong
        ranked.sort(key=lambda item: (-float(item.get("_module_score", 0)), int(item.get("_module_rank", 9999))))
        return [
            {key: value for key, value in item.items() if key not in {"_module_score", "_module_rank"}}
            for item in ranked
        ]

    @classmethod
    def _normalize_module_like_item(cls, item: dict[str, Any]) -> dict[str, Any]:
        current = dict(item)
        for source_key in ("title", "name", "text", "raw_text"):
            source = str(current.get(source_key) or "").strip()
            module_name, description = cls._split_module_name_description(source)
            if not module_name:
                continue
            current["title"] = module_name
            current["name"] = module_name
            current["text"] = module_name
            if description and not str(current.get("description") or "").strip():
                current["description"] = description
                current["snippet"] = description
            elif description and cls._module_like_item_score({"title": module_name, "description": description}) > cls._module_like_item_score(current):
                current["description"] = description
                current["snippet"] = description
            return current
        return current

    @staticmethod
    def _split_module_name_description(value: str) -> tuple[str, str]:
        text = " ".join(str(value or "").split()).strip()
        if not text:
            return "", ""
        match = re.match(r"^([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)?)\s+(?:[\u2013\u2014-])\s+(.{3,240})$", text)
        if not match:
            return "", ""
        return match.group(1).strip(), match.group(2).strip()

    @classmethod
    def _module_like_item_score(cls, item: dict[str, Any]) -> float:
        title = str(item.get("title") or item.get("name") or item.get("text") or "").strip()
        description = str(item.get("description") or item.get("snippet") or "").strip()
        raw_text = str(item.get("raw_text") or "").strip()
        href = str(item.get("href") or item.get("link") or "").strip()
        if not title:
            return -10.0
        folded_title = title.casefold()
        score = 0.0
        if re.fullmatch(r"[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)?", title):
            score += 5.0
        elif re.fullmatch(r"[a-z_][a-z0-9_.-]{1,40}", title):
            score += 3.0
        if description and description != title and 8 <= len(description) <= 220:
            score += 2.0
        if href and re.search(rf"/{re.escape(title)}(?:\.|/|$)", href, flags=re.IGNORECASE):
            score += 1.0
        if raw_text and len(raw_text) <= 260:
            score += 0.5
        if raw_text and len(raw_text) >= 700:
            score -= 3.0
        if re.search(r"\b(?:introduction|overview|tutorial|guide|reference|index|contents?|notes?)\b", folded_title):
            score -= 3.0
        if " " in title and not re.search(r"[._-]", title):
            score -= 1.0
        if title[:1].isupper() and any(char.islower() for char in title[1:]) and " " in title:
            score -= 1.0
        return score

    @classmethod
    def _card_items_from_row_candidates(
        cls,
        *,
        rows: list[dict[str, Any]],
        limit: int,
        numeric_value_required: bool = False,
    ) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        seen: set[str] = set()
        numeric_pattern = re.compile(r"[-+]?\d[\d\s.,]*")
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = cls._clean_row_candidate_item_text(str(row.get("text") or ""))
            if not text or len(text) > 320:
                continue
            if numeric_value_required and not numeric_pattern.search(text):
                continue
            lines = [cls._clean_row_candidate_item_text(part) for part in str(row.get("text") or "").splitlines()]
            lines = [line for line in lines if line]
            title = lines[0] if lines else text
            if len(title) > 220:
                title = text[:220].strip()
            key = title.casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            item = {
                "title": title,
                "name": title,
                "text": title,
                "description": "",
                "snippet": "",
                "href": "",
                "link": "",
                "selector": row.get("selector", ""),
                "raw_text": text,
            }
            if isinstance(row.get("frame_index"), int):
                item["frame_index"] = row["frame_index"]
            if row.get("frame_url"):
                item["frame_url"] = row["frame_url"]
            cards.append(item)
            if len(cards) >= max(limit, 1):
                break
        return cards

    @staticmethod
    def _clean_row_candidate_item_text(value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        text = re.sub(r"\s*(?:[×✕✖✗]|Г—)\s*$", "", text).strip()
        text = re.sub(r"\s+\b(?:close|delete|remove|dismiss)\b\s*$", "", text, flags=re.IGNORECASE).strip()
        return text

    @staticmethod
    def _remove_shared_item_descriptions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        description_titles: dict[str, set[str]] = {}
        for item in items:
            description = " ".join(str(item.get("description") or item.get("snippet") or "").split()).strip()
            title = " ".join(str(item.get("title") or item.get("name") or item.get("text") or "").split()).strip()
            if description:
                description_titles.setdefault(description.casefold(), set()).add(title.casefold())
        shared = {
            description
            for description, titles in description_titles.items()
            if len({title for title in titles if title}) > 1
        }
        if not shared:
            return items
        cleaned: list[dict[str, Any]] = []
        for item in items:
            current = dict(item)
            description = " ".join(str(current.get("description") or current.get("snippet") or "").split()).strip()
            if description.casefold() in shared:
                current.pop("description", None)
                current.pop("snippet", None)
            cleaned.append(current)
        return cleaned

    @classmethod
    def _project_item_to_schema(cls, *, item: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
        aliases = {
            "title": ("title", "name", "text"),
            "name": ("name", "title", "text"),
            "description": ("description", "snippet"),
            "snippet": ("snippet", "description"),
            "url": ("href", "link"),
            "href": ("href", "link"),
            "link": ("link", "href"),
        }
        projected: dict[str, Any] = {}
        raw_text = str(item.get("raw_text", "") or "")
        for field_name, rule in fields.items():
            field = str(field_name or "").strip()
            if not field:
                continue
            rule_dict = rule if isinstance(rule, dict) else {}
            field_type = str(rule_dict.get("type", "") or "").strip().casefold()
            candidates = aliases.get(field_type) or aliases.get(field.casefold()) or (field,)
            value = next((item.get(key) for key in candidates if item.get(key) not in (None, "", [], {})), None)
            pattern = str(rule_dict.get("value_pattern", "") or "").strip()
            if value in (None, "", [], {}) and pattern and raw_text:
                try:
                    match = re.search(pattern, raw_text, flags=re.IGNORECASE)
                except re.error:
                    match = None
                if match:
                    group_index = int(rule_dict.get("group_index", 1 if match.lastindex else 0))
                    value = match.group(group_index)
            if value not in (None, "", [], {}):
                projected[field] = value
        for diagnostic in ("selector", "raw_text"):
            if item.get(diagnostic) not in (None, ""):
                projected[diagnostic] = item[diagnostic]
        return projected




    async def _collect_row_candidates_generic(self, *, page, limit: int) -> list[dict[str, Any]]:
        rows = await self._collect_row_candidates_from_context(context=page, limit=limit)
        frames = getattr(page, "frames", None)
        if isinstance(frames, list) and len(rows) < limit:
            main_frame = getattr(page, "main_frame", None)
            for index, frame in enumerate(frames):
                if frame is main_frame:
                    continue
                remaining = max(limit - len(rows), 1)
                frame_rows = await self._collect_row_candidates_from_context(
                    context=frame,
                    limit=remaining,
                    frame_index=index,
                )
                rows.extend(frame_rows)
                if len(rows) >= limit:
                    break
        return rows[:limit]

    async def _collect_row_candidates_from_context(self, *, context, limit: int, frame_index: int | None = None) -> list[dict[str, Any]]:
        try:
            payload = await context.evaluate(
            """
            ({ limit }) => {
              const norm = (value) => String(value || "").replace(/\\s+/g, " ").trim();
              const isVisible = (el) => {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style && style.visibility !== "hidden" && style.display !== "none" && Number(style.opacity || 1) !== 0 && rect.width > 0 && rect.height > 0;
              };
              const cssEscape = (value) => {
                if (window.CSS && CSS.escape) return CSS.escape(value);
                return String(value).replace(/[^a-zA-Z0-9_-]/g, "\\\\$&");
              };
              const cssPath = (el) => {
                if (el.id) return `#${cssEscape(el.id)}`;
                const parts = [];
                let cur = el;
                while (cur && cur.nodeType === Node.ELEMENT_NODE && cur !== document.body && parts.length < 6) {
                  const tag = cur.tagName.toLowerCase();
                  let part = tag;
                  const cls = Array.from(cur.classList || []).slice(0, 2).map(cssEscape);
                  if (cls.length) part += "." + cls.join(".");
                  const parent = cur.parentElement;
                  if (parent) {
                    const siblings = Array.from(parent.children).filter((sib) => sib.tagName === cur.tagName);
                    if (siblings.length > 1) part += `:nth-of-type(${siblings.indexOf(cur) + 1})`;
                  }
                  parts.unshift(part);
                  cur = parent;
                }
                return parts.join(" > ");
              };
              const headersFor = (row) => {
                const table = row.closest("table,[role='table'],[role='grid']");
                if (!table) return [];
                return Array.from(table.querySelectorAll("thead th,thead [role='columnheader'],tr:first-child th,tr:first-child [role='columnheader']")).map((cell) => norm(cell.innerText || cell.textContent || "")).filter(Boolean);
              };
              const specificRows = Array.from(document.querySelectorAll("tr,[role='row'],li,[role='listitem'],article"));
              const broadRows = Array.from(document.querySelectorAll("[class*='row'],[class*='item'],[class*='card']"));
              const seen = new Set();
              const candidates = [];
              for (const el of [...specificRows, ...broadRows]) {
                if (!el || seen.has(el)) continue;
                seen.add(el);
                candidates.push(el);
              }
              return candidates
                .filter(isVisible)
                .map((el, index) => ({
                  row_id: `row_${index + 1}`,
                  tag: (el.tagName || "").toLowerCase(),
                  role: el.getAttribute("role") || "",
                  className: el.getAttribute("class") || "",
                  text: norm(el.innerText || el.textContent || ""),
                  selector: cssPath(el),
                  headers: headersFor(el),
                  cells: Array.from(el.querySelectorAll("th,td,[role='cell'],[role='gridcell']")).map((cell) => norm(cell.innerText || cell.textContent || "")).filter(Boolean),
                  links: Array.from(el.querySelectorAll("a[href]")).map((a) => ({ text: norm(a.innerText || a.textContent || ""), href: a.href || a.getAttribute("href") || "" })).filter((link) => link.text && link.href)
                }))
                .filter((row) => row.text)
                .slice(0, limit);
            }
            """,
            {"limit": max(limit, 1)},
            )
        except Exception:
            return []
        rows = [dict(item) for item in payload or [] if isinstance(item, dict)]
        if frame_index is not None:
            frame_url = str(getattr(context, "url", "") or "")
            for row in rows:
                row["frame_index"] = frame_index
                if frame_url:
                    row["frame_url"] = frame_url
        return rows

    async def _extract_svg_summary(self, *, page) -> dict[str, Any]:
        payload = await page.evaluate(
            """
            () => {
              const shapeTags = ["circle", "rect", "ellipse", "polygon", "path", "line", "polyline"];
              const counts = {};
              const items = [];
              for (const tag of shapeTags) {
                const nodes = Array.from(document.querySelectorAll(`svg ${tag}`));
                if (nodes.length) counts[tag] = nodes.length;
                for (const node of nodes.slice(0, 80)) {
                  const rect = node.getBoundingClientRect();
                  items.push({ tag, bbox: { x: Math.round(rect.x), y: Math.round(rect.y), width: Math.round(rect.width), height: Math.round(rect.height) } });
                }
              }
              return { shape_counts: counts, shapes: items };
            }
            """
        )
        return dict(payload or {}) if isinstance(payload, dict) else {"shape_counts": {}, "shapes": []}

    async def _count_visible_dom_objects_by_geometry(self, *, page, target: str, args: dict) -> int | None:
        target_text = str(target or "").casefold()
        countable_targets = {
            "link": ("link", "links", "anchor", "anchors"),
            "button": ("button", "buttons"),
            "input": ("input", "inputs", "field", "fields"),
            "item": ("item", "items", "card", "cards", "row", "rows"),
        }
        target_kind = ""
        for kind, tokens in countable_targets.items():
            if any(token in target_text for token in tokens):
                target_kind = kind
                break
        if not target_kind:
            return None

        payload = await page.evaluate(
            """
            ({ targetKind, region }) => {
              const norm = (value) => String(value || "").replace(/\\s+/g, " ").trim();
              const viewport = { width: window.innerWidth || 0, height: window.innerHeight || 0 };
              const isVisible = (el) => {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style
                  && style.visibility !== "hidden"
                  && style.display !== "none"
                  && Number(style.opacity || 1) !== 0
                  && rect.width > 0
                  && rect.height > 0
                  && rect.bottom >= 0
                  && rect.right >= 0
                  && rect.top <= viewport.height
                  && rect.left <= viewport.width;
              };
              const selectors = {
                link: "a[href],[role='link']",
                button: "button,[role='button'],input[type='button'],input[type='submit']",
                input: "input,textarea,select,[contenteditable='true']",
                item: "li,article,[role='listitem'],[class*='item' i],[class*='card' i],[class*='row' i]"
              };
              const normalizeRegion = (raw) => {
                if (!raw || typeof raw !== "object") return null;
                let x = Number(raw.x);
                let y = Number(raw.y);
                let width = Number(raw.width);
                let height = Number(raw.height);
                if (![x, y, width, height].every(Number.isFinite) || width <= 0 || height <= 0) return null;
                const ratioLike = Math.max(Math.abs(x), Math.abs(y), Math.abs(width), Math.abs(height)) <= 1.5;
                if (ratioLike) {
                  const pad = 0.08;
                  x = Math.max(0, x - pad) * viewport.width;
                  y = Math.max(0, y - pad) * viewport.height;
                  width = Math.min(1, width + pad * 2) * viewport.width;
                  height = Math.min(1, height + pad * 2) * viewport.height;
                }
                return { x, y, width, height };
              };
              const regionBox = normalizeRegion(region);
              const inRegion = (rect) => {
                if (!regionBox) return true;
                const cx = rect.left + rect.width / 2;
                const cy = rect.top + rect.height / 2;
                return cx >= regionBox.x
                  && cx <= regionBox.x + regionBox.width
                  && cy >= regionBox.y
                  && cy <= regionBox.y + regionBox.height;
              };
              const seen = new Set();
              const items = [];
              for (const el of Array.from(document.querySelectorAll(selectors[targetKind] || selectors.item))) {
                if (!isVisible(el)) continue;
                const rect = el.getBoundingClientRect();
                if (!inRegion(rect)) continue;
                const text = norm(el.innerText || el.textContent || el.getAttribute("aria-label") || el.getAttribute("title") || "");
                if (targetKind !== "input" && !text) continue;
                const key = `${Math.round(rect.left)}:${Math.round(rect.top)}:${Math.round(rect.width)}:${Math.round(rect.height)}:${text}`;
                if (seen.has(key)) continue;
                seen.add(key);
                items.push({ text, bbox: { x: rect.left, y: rect.top, width: rect.width, height: rect.height } });
              }
              return { count: items.length, items: items.slice(0, 20) };
            }
            """,
            {"targetKind": target_kind, "region": args.get("region")},
        )
        if not isinstance(payload, dict):
            return None
        count = payload.get("count")
        if isinstance(count, int) and count >= 0:
            return count
        return None

    @staticmethod
    def _condition_terms(condition: Any) -> list[str]:
        if isinstance(condition, dict):
            values: list[str] = []
            metadata_keys = {"field", "operator", "mode", "match", "case_sensitive"}
            for key, value in condition.items():
                if str(key).casefold() in metadata_keys:
                    continue
                values.extend(ActionHandlers._flatten_condition_values(value))
            return list(dict.fromkeys(values))
        if isinstance(condition, list):
            return ActionHandlers._flatten_condition_values(condition)
        condition_text = str(condition or "").strip()
        quoted = re.findall(r"['\"]([^'\"]{1,120})['\"]", condition_text)
        if quoted:
            return list(dict.fromkeys(item.strip() for item in quoted if item.strip()))
        if "==" in condition_text:
            condition_text = condition_text.split("==", 1)[1].strip()
        return [part.strip() for part in re.split(r"\s*\|\s*|\s*,\s*", condition_text) if part.strip()]

    @staticmethod
    def _flatten_condition_values(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, dict):
            flattened: list[str] = []
            for item in value.values():
                flattened.extend(ActionHandlers._flatten_condition_values(item))
            return flattened
        if isinstance(value, (list, tuple, set)):
            flattened = []
            for item in value:
                flattened.extend(ActionHandlers._flatten_condition_values(item))
            return [item for item in flattened if item]
        text = str(value).strip()
        return [text] if text else []

    @classmethod
    def _condition_term_groups(cls, condition: Any) -> list[list[str]]:
        if not isinstance(condition, dict):
            if isinstance(condition, str) and re.search(r"\b(or|или)\b|\|\|", condition, flags=re.IGNORECASE):
                quoted_terms = [
                    item.strip()
                    for item in re.findall(r"['\"]([^'\"]{1,120})['\"]", condition)
                    if item.strip()
                ]
                if quoted_terms:
                    return [[term] for term in list(dict.fromkeys(quoted_terms))]
                split_terms = [
                    item.strip()
                    for item in re.split(r"\s+(?:or|или)\s+|\|\|", condition, flags=re.IGNORECASE)
                    if item.strip()
                ]
                if split_terms:
                    return [[term] for term in list(dict.fromkeys(split_terms))]
            terms = cls._condition_terms(condition)
            return [terms] if terms else []

        required_terms: list[str] = []
        alternative_terms: list[str] = []
        metadata_keys = {"field", "operator", "mode", "match", "case_sensitive", "all"}
        for key, value in condition.items():
            if str(key).casefold() in metadata_keys:
                continue
            terms = cls._flatten_condition_values(value)
            if not terms:
                continue
            if isinstance(value, (list, tuple, set)):
                alternative_terms.extend(terms)
            else:
                required_terms.extend(terms)
        required_terms.extend(cls._flatten_condition_values(condition.get("all")))

        required_terms = list(dict.fromkeys(required_terms))
        alternative_terms = list(dict.fromkeys(alternative_terms))
        if alternative_terms:
            return [list(dict.fromkeys([*required_terms, term])) for term in alternative_terms]
        return [required_terms] if required_terms else []

    @classmethod
    def _filter_structured_rows_by_condition(cls, *, rows: list[dict[str, Any]], condition: Any) -> list[dict[str, Any]]:
        field_condition = cls._condition_field_operator_value(condition)
        if field_condition:
            field, operator, value = field_condition
            return [
                row for row in rows
                if cls._row_matches_field_condition(row=row, field=field, operator=operator, value=value)
            ]
        term_groups = cls._condition_term_groups(condition)
        if not term_groups:
            return rows
        matched: list[dict[str, Any]] = []
        for row in rows:
            haystack = " ".join(
                [
                    str(row.get("text", "")),
                    " ".join(str(cell) for cell in row.get("cells", []) or []),
                    " ".join(str(value) for value in (row.get("fields_by_header") or {}).values()),
                    " ".join(str(value) for key, value in row.items() if key not in {"cells", "headers", "fields_by_header"}),
                ]
            ).casefold()
            if any(all(term.casefold() in haystack for term in terms) for terms in term_groups):
                matched.append(row)
        return matched

    @staticmethod
    def _condition_field_operator_value(condition: Any) -> tuple[str, str, str] | None:
        if not isinstance(condition, dict):
            return None
        field = str(condition.get("field") or "").strip()
        value = condition.get("value")
        if value is None and "contains" in condition:
            value = condition.get("contains")
            operator = "contains"
        else:
            operator = str(condition.get("operator") or condition.get("match") or "contains").strip().casefold()
        if not field and value is not None and "value" in condition:
            return None
        if not field or value is None:
            metadata_keys = {"field", "operator", "mode", "match", "case_sensitive", "all", "contains", "value"}
            data_items = [
                (str(key).strip(), raw_value)
                for key, raw_value in condition.items()
                if str(key).casefold() not in metadata_keys and raw_value not in (None, "", [], {})
            ]
            if len(data_items) != 1:
                return None
            field, value = data_items[0]
            operator = "contains"
        text_value = str(value).strip()
        if not text_value:
            return None
        return field, operator, text_value

    @classmethod
    def _row_matches_field_condition(cls, *, row: dict[str, Any], field: str, operator: str, value: str) -> bool:
        cell = cls._value_for_row_field(row=row, field=field)
        if cell is None:
            return False
        haystack = str(cell).casefold()
        needle = str(value).casefold()
        if operator in {"equals", "eq", "=", "==", "exact"}:
            return haystack.strip() == needle.strip()
        if operator in {"starts_with", "prefix"}:
            return haystack.strip().startswith(needle.strip())
        if operator in {"ends_with", "suffix"}:
            return haystack.strip().endswith(needle.strip())
        return needle in haystack

    @classmethod
    def _value_for_row_field(cls, *, row: dict[str, Any], field: str) -> Any:
        wanted = cls._normalize_header_match_key(field)
        if not wanted:
            return None
        direct_keys = [field, field.casefold(), cls._header_alias(field)]
        for key in direct_keys:
            if key in row and row.get(key) not in (None, ""):
                return row.get(key)
        fields_by_header = row.get("fields_by_header") if isinstance(row.get("fields_by_header"), dict) else {}
        for header, value in fields_by_header.items():
            normalized = cls._normalize_header_match_key(str(header))
            if normalized and (wanted == normalized or wanted in normalized or normalized in wanted):
                return value
        headers = [str(header or "") for header in row.get("headers", []) or []]
        cells = list(row.get("cells", []) or [])
        for idx, header in enumerate(headers):
            normalized = cls._normalize_header_match_key(header)
            if normalized and (wanted == normalized or wanted in normalized or normalized in wanted):
                return cells[idx] if idx < len(cells) else None
        return None

    @staticmethod
    def _extract_numeric_upper_bound(*, args: dict, runtime_state=None) -> float | None:
        for key in ("numeric_upper_bound", "max_value", "budget"):
            value = args.get(key)
            if value is None:
                continue
            normalized = ActionHandlers._normalize_numeric_token(str(value))
            if normalized is not None:
                return float(normalized)
        goal = str((runtime_state or {}).get("user_goal", "")) if isinstance(runtime_state, dict) else ""
        match = re.search(r"(?:до|under|below|at most|<=|less than)\s*([0-9][0-9\s.,]*)", goal, flags=re.IGNORECASE)
        if match:
            normalized = ActionHandlers._normalize_numeric_token(match.group(1))
            return float(normalized) if normalized is not None else None
        return None

    @staticmethod
    def _normalize_numeric_token(value: str) -> float | None:
        text = str(value or "")
        match = re.search(r"([0-9][0-9\s.,]*)", text)
        if not match:
            return None
        token = match.group(1).replace("\u00a0", " ").replace("\u202f", " ")
        token = re.sub(r"\s+", "", token)
        if "," in token and "." in token:
            token = token.replace(".", "").replace(",", ".")
        elif "," in token:
            token = token.replace(",", ".")
        try:
            return float(token)
        except ValueError:
            return None

    @classmethod
    def _filter_rows_by_pattern_literals(cls, *, rows: list[dict[str, Any]], pattern: str) -> list[dict[str, Any]]:
        terms = cls._literal_terms_from_pattern(pattern)
        if not terms:
            return []
        matched: list[dict[str, Any]] = []
        for row in rows:
            haystack = " ".join(
                [
                    str(row.get("text", "")),
                    " ".join(str(cell) for cell in row.get("cells", []) or []),
                ]
            ).casefold()
            if all(term.casefold() in haystack for term in terms):
                matched.append(row)
        return matched

    @staticmethod
    def _literal_terms_from_pattern(pattern: str) -> list[str]:
        cleaned = re.sub(r"\\[dDsSwWbBAZz]|\[[^\]]+\]|\(\?[:=!<].*?\)|[{}+*?.^$|()]", " ", str(pattern or ""))
        words = re.findall(r"[A-Za-zА-Яа-яЁё]{2,}", cleaned)
        stop = {"and", "or", "the", "www", "http", "https"}
        return list(dict.fromkeys(word for word in words if word.casefold() not in stop))

    @classmethod
    def _build_row_payload(cls, row: dict[str, Any]) -> dict[str, Any]:
        cells = [str(cell).strip() for cell in row.get("cells", []) if str(cell).strip()]
        headers = [str(header).strip() for header in row.get("headers", []) if str(header).strip()]
        fields_by_header = {}
        for idx, cell in enumerate(cells):
            if idx < len(headers):
                fields_by_header[headers[idx]] = cell
        aliases: dict[str, Any] = {}
        for header, value in fields_by_header.items():
            key = cls._header_alias(header)
            if key and key not in aliases:
                aliases[key] = value
        payload = {
            "row_id": row.get("row_id"),
            "selector": row.get("selector"),
            "text": row.get("text"),
            "cells": cells,
            "headers": headers,
            "fields_by_header": fields_by_header,
            "links": row.get("links", []),
        }
        if isinstance(row.get("frame_index"), int):
            payload["frame_index"] = row["frame_index"]
        if str(row.get("frame_url", "") or "").strip():
            payload["frame_url"] = str(row.get("frame_url")).strip()
        payload.update(aliases)
        return payload

    @classmethod
    def _requested_table_headers(cls, *, args: dict, runtime_state=None) -> list[str]:
        explicit = (
            args.get("columns")
            or args.get("headers")
            or args.get("required_headers")
            or args.get("fields")
        )
        headers = cls._normalize_requested_header_list(explicit)
        if headers:
            return headers
        goal = str((runtime_state or {}).get("user_goal", "") or "")
        if not goal:
            return []
        marker_match = re.search(
            r"(?:columns?|headers?|fields?|колонк\w*|столбц\w*|пол\w*)\s*[:：]?\s+([^.;\n]+)",
            goal,
            flags=re.IGNORECASE,
        )
        if not marker_match:
            return []
        return cls._normalize_requested_header_list(marker_match.group(1))

    @staticmethod
    def _normalize_requested_header_list(value: Any) -> list[str]:
        if isinstance(value, dict):
            raw_items = list(value.keys())
        elif isinstance(value, list):
            raw_items = value
        elif isinstance(value, str):
            raw_items = re.split(r",|;|\band\b|\bor\b|\bи\b|\bили\b", value, flags=re.IGNORECASE)
        else:
            raw_items = []
        stop_words = {
            "with",
            "columns",
            "column",
            "headers",
            "header",
            "fields",
            "field",
            "колонками",
            "колонки",
            "колонка",
            "полями",
            "поля",
        }
        headers: list[str] = []
        for item in raw_items:
            header = str(item or "").strip().strip("`'\"“”«»()[]{}")
            header = re.sub(r"\s+", " ", header)
            header = header.strip(" .,:;-")
            if not header:
                continue
            if header.casefold() in stop_words:
                continue
            if len(header) > 60:
                continue
            if not re.search(r"[A-Za-zА-Яа-яЁё0-9]", header):
                continue
            if header not in headers:
                headers.append(header)
        return headers

    @classmethod
    def _field_schema_table_row_from_source_text(
        cls,
        *,
        fields: dict[str, Any],
        source_text: str,
        runtime_state=None,
    ) -> dict[str, Any]:
        if not isinstance(fields, dict) or not fields:
            return {}
        goal = str((runtime_state or {}).get("user_goal", "") or "")
        condition = cls._table_field_condition_from_goal(goal)
        if not condition:
            return {}
        field_name, expected_value = condition
        rows = cls._table_like_rows_from_source_text(source_text)
        if not rows:
            return {}
        requested_fields = [str(field).strip() for field in fields if str(field).strip()]
        requested_keys = [cls._normalize_header_match_key(field) for field in requested_fields]
        condition_key = cls._normalize_header_match_key(field_name)
        header_index = -1
        header_cells: list[str] = []
        for index, cells in enumerate(rows):
            normalized = [cls._normalize_header_match_key(cell) for cell in cells]
            if condition_key and not any(condition_key == cell or condition_key in cell or cell in condition_key for cell in normalized):
                continue
            requested_matches = sum(
                1
                for wanted in requested_keys
                if wanted and any(wanted == cell or wanted in cell or cell in wanted for cell in normalized)
            )
            if requested_matches >= 1:
                header_index = index
                header_cells = cells
                break
        if header_index < 0 or not header_cells:
            return {}
        normalized_headers = [cls._normalize_header_match_key(header) for header in header_cells]
        condition_idx = next(
            (
                idx
                for idx, header in enumerate(normalized_headers)
                if condition_key and header and (condition_key == header or condition_key in header or header in condition_key)
            ),
            None,
        )
        if condition_idx is None:
            return {}
        matched_cells: list[str] = []
        for cells in rows[header_index + 1:]:
            if len(cells) <= condition_idx:
                continue
            if str(expected_value).casefold() in str(cells[condition_idx]).casefold():
                matched_cells = cells
                break
        if not matched_cells:
            return {}
        row_payload = cls._build_row_payload(
            {
                "headers": header_cells,
                "cells": matched_cells,
                "text": " ".join(matched_cells),
                "row_id": "source_text_table_row",
            }
        )
        payload: dict[str, Any] = {}
        for field in requested_fields:
            value = cls._value_for_row_field(row=row_payload, field=field)
            if value not in (None, "", [], {}):
                payload[field] = value
        for header, value in row_payload.get("fields_by_header", {}).items():
            if value not in (None, "", [], {}):
                payload.setdefault(header, value)
                alias = cls._header_alias(header)
                if alias:
                    payload.setdefault(alias, value)
        found = [field for field in requested_fields if field in payload and payload[field] not in (None, "", [], {})]
        payload["status"] = "success" if found else "not_found"
        payload["found_fields"] = found
        payload["missing_fields"] = [field for field in requested_fields if field not in found]
        return payload if found else {}

    @staticmethod
    def _table_field_condition_from_goal(goal: str) -> tuple[str, str] | None:
        text = str(goal or "")
        patterns = [
            r"(?:where|when|if|где|если)\s+([A-Za-zА-Яа-яЁё0-9 _-]{2,60}?)\s+(?:contains|includes|has|содержит|включает)\s+([A-Za-zА-Яа-яЁё0-9 _.,+-]{1,80})",
            r"([A-Za-zА-Яа-яЁё0-9 _-]{2,60}?)\s+(?:contains|includes|has|содержит|включает)\s+([A-Za-zА-Яа-яЁё0-9 _.,+-]{1,80})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            field = re.sub(r"\s+", " ", match.group(1)).strip(" :;,.\"'")
            value = re.sub(r"\s+", " ", match.group(2)).strip(" :;,.\"'")
            value = re.split(r"\s+(?:верни|return|and return)\b", value, maxsplit=1, flags=re.IGNORECASE)[0].strip(" :;,.\"'")
            if field and value:
                return field, value
        return None

    @classmethod
    def _table_like_rows_from_source_text(cls, source_text: str) -> list[list[str]]:
        rows: list[list[str]] = []
        for line in str(source_text or "").splitlines():
            normalized = cls._normalize_line(line)
            if not normalized:
                continue
            if "\t" in line:
                cells = [cls._normalize_line(cell) for cell in line.split("\t")]
            else:
                cells = [cls._normalize_line(cell) for cell in re.split(r"\s{2,}", normalized)]
            cells = [cell for cell in cells if cell]
            if len(cells) >= 2:
                rows.append(cells)
        return rows

    @classmethod
    def _filter_table_rows_by_requested_headers(
        cls,
        *,
        rows: list[dict[str, Any]],
        requested_headers: list[str],
    ) -> list[dict[str, Any]]:
        requested = [cls._normalize_header_match_key(header) for header in requested_headers]
        requested = [header for header in requested if header]
        if not requested:
            return rows
        matched: list[dict[str, Any]] = []
        for row in rows:
            raw_headers = [str(header or "").strip() for header in row.get("headers", []) or []]
            raw_cells = list(row.get("cells", []) or [])
            normalized_headers = [cls._normalize_header_match_key(header) for header in raw_headers]
            selected_indices: list[int] = []
            for required in requested:
                index = next(
                    (
                        idx
                        for idx, header in enumerate(normalized_headers)
                        if header and (required == header or required in header or header in required)
                    ),
                    None,
                )
                if index is None:
                    selected_indices = []
                    break
                selected_indices.append(index)
            if not selected_indices:
                continue
            selected_headers = [raw_headers[index] for index in selected_indices]
            selected_cells = [raw_cells[index] if index < len(raw_cells) else "" for index in selected_indices]
            projected = {
                key: row.get(key)
                for key in ("row_id", "selector", "text", "links")
                if row.get(key) not in (None, "")
            }
            projected["headers"] = selected_headers
            projected["cells"] = selected_cells
            projected["fields_by_header"] = dict(zip(selected_headers, selected_cells))
            for header, cell in zip(selected_headers, selected_cells):
                alias = cls._header_alias(header)
                if alias:
                    projected[alias] = cell
            matched.append(projected)
        return matched or rows

    @staticmethod
    def _normalize_header_match_key(value: str) -> str:
        return re.sub(r"[^\wА-Яа-яЁё]+", "", str(value or "").casefold(), flags=re.UNICODE)

    @staticmethod
    def _header_alias(header: str) -> str:
        normalized = re.sub(r"[^\w]+", "_", str(header or "").strip().casefold(), flags=re.UNICODE).strip("_")
        if not normalized:
            return ""
        if normalized[0].isdigit():
            normalized = f"column_{normalized}"
        return normalized[:80]

    async def _load_source_text(self, page, runtime_state=None, force_refresh: bool = False) -> str:
        source_text = ""
        if runtime_state is not None and not force_refresh:
            source_text = runtime_state.get("last_page_text") or ""
        if not source_text:
            source_text = (await page.locator("body").inner_text()).strip()
            if runtime_state is not None:
                runtime_state["last_page_text"] = source_text
        return source_text

    async def _resolve_ranked_click_locator(self, *, page, args, runtime_state=None):
        selector = str(args.get("selector", "")).strip()
        text = str(args.get("text", "")).strip()
        anchor = str(args.get("anchor", "")).strip()
        role = str(args.get("role", "")).strip()
        name = str(args.get("name", "")).strip()
        href_contains = str(args.get("href_contains", "")).strip()
        scope_selector = str(args.get("scope_selector", "")).strip()
        label = str(args.get("label", "")).strip()
        placeholder = str(args.get("placeholder", "")).strip()
        exact = bool(args.get("exact", False))
        visible_only = bool(args.get("visible_only", True))

        if selector and self._selector_looks_like_plain_text_click_target(selector):
            if not text and not self._is_meta_click_label(selector):
                text = selector
                args["text"] = text
                args.setdefault("exact", True)
            if not href_contains and text:
                inferred_href = self._infer_href_slug_from_text(text)
                if inferred_href:
                    args["href_contains"] = inferred_href
                    href_contains = inferred_href
            args["_selector_canonicalized_from_plain_text"] = selector
            args.pop("selector", None)
            selector = ""

        text_is_meta = self._is_meta_click_label(text)
        anchor_is_meta = self._is_meta_click_label(anchor)
        name_is_meta = self._is_meta_click_label(name)

        canonical_text = ""
        if text and not text_is_meta:
            canonical_text = text
        elif anchor and not anchor_is_meta:
            canonical_text = anchor
        elif name and not name_is_meta:
            canonical_text = name

        if canonical_text:
            text = canonical_text
            args["text"] = canonical_text
            args.pop("anchor", None)
        elif text_is_meta:
            text = ""
            args.pop("text", None)

        if text and not href_contains:
            inferred_href = self._infer_href_slug_from_text(text)
            if inferred_href:
                href_contains = inferred_href
                args["href_contains"] = inferred_href

        scope = page.locator(scope_selector) if scope_selector else page
        candidates: list[dict[str, Any]] = []
        if text:
            candidates.append({"strategy": "role_link_name", "locator": scope.get_by_role("link", name=text, exact=exact), "selector": f"role=link, name={text}"})
        if anchor and not anchor_is_meta:
            candidates.append({"strategy": "role_link_anchor", "locator": scope.get_by_role("link", name=anchor, exact=exact), "selector": f"role=link, name={anchor}"})
        if name and not name_is_meta:
            candidates.append({"strategy": "role_link_name_field", "locator": scope.get_by_role("link", name=name, exact=exact), "selector": f"role=link, name={name}"})
        if role and name and not name_is_meta:
            candidates.append({"strategy": "role_name", "locator": scope.get_by_role(role, name=name, exact=exact), "selector": f"role={role}, name={name}"})
        if label:
            candidates.append({"strategy": "label", "locator": scope.get_by_label(label, exact=exact), "selector": f"label={label}"})
        if placeholder:
            candidates.append({"strategy": "placeholder", "locator": scope.get_by_placeholder(placeholder), "selector": f"placeholder={placeholder}"})
        if text:
            candidates.append({"strategy": "visible_text", "locator": scope.get_by_text(text, exact=exact), "selector": f"text={text}"})
            candidates.append({"strategy": "visible_text_fuzzy", "locator": scope.get_by_text(text, exact=False), "selector": f"text~={text}"})
            normalized_text = re.sub(r"\s+", " ", text).strip()
            if normalized_text and normalized_text != text:
                candidates.append({"strategy": "visible_text_normalized", "locator": scope.get_by_text(normalized_text, exact=False), "selector": f"text_norm~={normalized_text}"})
            escaped = re.escape(text).replace("\\ ", r"\s+")
            candidates.append({"strategy": "visible_text_casefold", "locator": scope.get_by_text(re.compile(escaped, flags=re.IGNORECASE)), "selector": f"text~/(?i){text}/"})
            inferred_slug = self._infer_href_slug_from_text(text)
            if inferred_slug:
                candidates.append({"strategy": "href_from_text_slug", "locator": scope.locator(f'a[href*="{inferred_slug}"]'), "selector": f'a[href*=\"{inferred_slug}\"]'})
                discovered_href = await self._discover_href_from_visible_links(page=page, text=text)
                if discovered_href:
                    candidates.append({"strategy": "href_from_visible_links", "locator": scope.locator(f'a[href*="{discovered_href}"]'), "selector": f'a[href*=\"{discovered_href}\"]'})
        if href_contains:
            href_selector = f'a[href*="{href_contains}"]'
            candidates.append({"strategy": "href_filter", "locator": scope.locator(href_selector), "selector": href_selector})
        if scope_selector and text:
            scoped_selector = f"{scope_selector} a, {scope_selector} button, {scope_selector} [role='button'], {scope_selector} [role='link']"
            candidates.append({"strategy": "scoped_selector", "locator": page.locator(scoped_selector).filter(has_text=text), "selector": scoped_selector})
        if selector:
            if self._is_too_broad_click_selector(selector):
                raise ValueError(
                    f"Click selector is too broad: {selector!r}. Use specific selector or text/role/href filter."
                )
            candidates.append({"strategy": "generic_selector_fallback", "locator": page.locator(selector), "selector": selector})

        if not candidates:
            if text_is_meta:
                raise ValueError(
                    "Invalid click target: args.text is benchmark/meta label and no valid fallback target found."
                )
            raise ValueError("click requires selector or text/role+name/href_contains/label/placeholder")

        diagnostics: list[dict[str, Any]] = []
        for candidate in candidates:
            locator = candidate["locator"]
            count = await locator.count()
            diagnostic_item = {
                "strategy": candidate["strategy"],
                "selector": candidate["selector"],
                "count": count,
            }
            diagnostics.append(diagnostic_item)
            if count > 0:
                selected = {
                    "strategy": candidate["strategy"],
                    "selector": candidate["selector"],
                    "count": count,
                    "candidates_checked": len(diagnostics),
                }
                if runtime_state is not None:
                    runtime_state["last_locator_diagnostics"] = diagnostics
                    runtime_state["last_selected_candidate"] = selected
                return locator, selected

        if runtime_state is not None:
            runtime_state["last_locator_diagnostics"] = diagnostics
        raise ValueError(f"No locator candidates matched for click target. diagnostics={diagnostics}")

    async def _wait_for_actionable(self, locator) -> None:
        target = locator.first
        await target.wait_for(state="visible", timeout=10000)
        await target.scroll_into_view_if_needed(timeout=10000)
        enabled = await target.is_enabled()
        if not enabled:
            raise ValueError("Resolved click target is disabled")
        await asyncio.sleep(0.08)

    @staticmethod
    def _build_structured_comparison(*, left: Any, right: Any, label_left: str, label_right: str) -> dict[str, Any]:
        left_is_empty = ActionHandlers._is_empty_structured_value(left)
        right_is_empty = ActionHandlers._is_empty_structured_value(right)
        left_keys = sorted(left.keys()) if isinstance(left, dict) else []
        right_keys = sorted(right.keys()) if isinstance(right, dict) else []
        differing_keys: list[str] = []
        differing_values: dict[str, dict[str, Any]] = {}
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left.keys()).union(right.keys())):
                if left.get(key) != right.get(key):
                    differing_keys.append(str(key))
                    differing_values[str(key)] = {
                        label_left: left.get(key),
                        label_right: right.get(key),
                    }

        comparison = {
            "compared": True,
            "compared_at": datetime.now(timezone.utc).isoformat(),
            "left_key": label_left,
            "right_key": label_right,
            "left_present": left is not None,
            "right_present": right is not None,
            "left_type": type(left).__name__,
            "right_type": type(right).__name__,
            "left_keys": left_keys,
            "right_keys": right_keys,
            "shared_keys": sorted(set(left_keys).intersection(right_keys)),
            "exact_match": left == right,
            "differing_keys": differing_keys,
            "differing_values": differing_values,
            "status": "equal" if left == right else "different",
        }
        if left_is_empty or right_is_empty:
            comparison["status"] = "insufficient_data"
            comparison["exact_match"] = False
            comparison["reason"] = "empty_source"
            comparison["left_is_empty"] = left_is_empty
            comparison["right_is_empty"] = right_is_empty
        return comparison

    @staticmethod
    def _is_too_broad_click_selector(selector: str) -> bool:
        return selector.strip().lower() in {"a", "button", "*", "[role='button']", '[role="button"]'}

    @staticmethod
    def _selector_looks_like_plain_text_click_target(selector: str) -> bool:
        candidate = str(selector or "").strip()
        if not candidate:
            return False
        if any(token in candidate for token in ("#", ".", "[", "]", ">", ":", "=", "/", "*")):
            return False
        if re.search(r"\s+", candidate):
            return True
        words = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", candidate)
        if len(words) >= 2 and all(word[:1].isupper() for word in words if word):
            return True
        return bool(re.fullmatch(r"[A-Za-zА-Яа-яЁё0-9 _-]{3,80}", candidate))

    @classmethod
    def _is_overly_broad_repeated_pattern(cls, pattern: str) -> bool:
        token = str(pattern or "").strip()
        if not token:
            return True
        compact = re.sub(r"\s+", "", token)
        if compact in {"(.+)", "^(.+)$", "(.*)", ".*", "^.*$"}:
            return True
        if re.fullmatch(r"^\^?\((?:\.\\s*)?[\*\+]\)\$?$", compact):
            return True
        try:
            compiled = re.compile(token)
        except re.error:
            return False
        if compiled.groups != 1:
            return False
        structural_markers = ("\\t", ",", ";", "\\|", "\\s{2,", "\\n", "\\r", "href", "<", ">")
        has_structure = any(marker in token for marker in structural_markers)
        if not has_structure and re.fullmatch(r"^\^?\(\.\*\+?\)\$?$", compact):
            return True
        return False

    @classmethod
    def _assert_section_heading_grounded(
        cls,
        *,
        heading_text: str,
        runtime_state,
        action_args: dict,
    ) -> None:
        if not isinstance(runtime_state, dict):
            return
        snapshot = runtime_state.get("last_page_snapshot")
        if not isinstance(snapshot, dict):
            return
        visible_headings = [cls._normalize_line(item) for item in (snapshot.get("visible_headings") or []) if cls._normalize_line(item)]
        page_text = str(snapshot.get("page_text") or snapshot.get("page_text_excerpt") or "")
        normalized_heading = cls._normalize_line(heading_text)
        heading_lc = normalized_heading.lower()
        headings_lc = [item.lower() for item in visible_headings]
        page_text_lc = page_text.lower()
        if heading_lc and (heading_lc in headings_lc or heading_lc in page_text_lc):
            return
        excerpt = re.sub(r"\s+", " ", page_text).strip()[:350]
        diagnostic = {
            "code": "section_heading_not_grounded_in_current_snapshot",
            "current_url": str(snapshot.get("url") or ""),
            "visible_headings": visible_headings[:12],
            "page_text_excerpt": excerpt,
            "heading_text": heading_text,
            "instruction": "choose heading only from current snapshot evidence",
        }
        action_args["_grounding_diagnostic"] = diagnostic
        raise StructuredExtractionError(
            code="section_heading_not_grounded_in_current_snapshot",
            message=(
                "section_heading_not_grounded_in_current_snapshot: heading_text is absent from current visible_headings/page_text."
            ),
            details=diagnostic,
        )

    @classmethod
    def _build_empty_section_diagnostics(cls, *, runtime_state, failed_heading: str) -> dict[str, list[dict[str, Any]]]:
        snapshot = runtime_state.get("last_page_snapshot") if isinstance(runtime_state, dict) else None
        headings_payload = snapshot.get("headings") if isinstance(snapshot, dict) else []
        available: list[dict[str, Any]] = []
        failed_norm = cls._normalize_line(failed_heading).lower()
        for item in headings_payload if isinstance(headings_payload, list) else []:
            if not isinstance(item, dict):
                continue
            text = cls._normalize_line(item.get("text", ""))
            if not text:
                continue
            line_count = int(item.get("line_count_after", 0) or 0)
            if line_count <= 0:
                continue
            available.append(
                {
                    "text": text,
                    "line_count_after": line_count,
                    "visible": bool(item.get("visible", True)),
                    "region": str(item.get("region", "unknown") or "unknown"),
                    "preview_after": [cls._normalize_line(v) for v in item.get("preview_after", []) if cls._normalize_line(v)],
                    "is_content_heading": bool(item.get("is_content_heading", False)),
                }
            )
        available = [
            item
            for item in available
            if item["visible"]
            and item["line_count_after"] > 0
            and item["preview_after"]
            and item["region"] not in {"nav", "header", "footer", "aside"}
        ]
        available = sorted(
            available,
            key=lambda x: (1 if x.get("is_content_heading") else 0, int(x.get("line_count_after", 0))),
            reverse=True,
        )
        suggested = [item for item in available if cls._normalize_line(item.get("text", "")).lower() != failed_norm][:5]
        return {
            "available_non_empty_headings": [
                {
                    "text": item["text"],
                    "line_count_after": item["line_count_after"],
                    "region": item["region"],
                    "preview_after": item["preview_after"][:2],
                }
                for item in available
            ],
            "suggested_next_headings": [
                {
                    "text": item["text"],
                    "line_count_after": item["line_count_after"],
                    "region": item["region"],
                    "preview_after": item["preview_after"][:2],
                }
                for item in suggested
            ],
        }

    @classmethod
    def _find_heading_indices(cls, lines: list[str], *, heading_text: str, ignore_case: bool) -> list[int]:
        target = cls._normalize_line(heading_text)
        if not target:
            return []
        matches: list[int] = []
        target_cmp = target.lower() if ignore_case else target
        for idx, line in enumerate(lines):
            candidate = cls._normalize_line(line)
            candidate_cmp = candidate.lower() if ignore_case else candidate
            if candidate_cmp == target_cmp or target_cmp in candidate_cmp:
                matches.append(idx)
        return matches

    @classmethod
    def _prioritize_heading_indices(
        cls,
        *,
        heading_indices: list[int],
        heading_text: str,
        runtime_state,
    ) -> list[int]:
        if len(heading_indices) <= 1 or not isinstance(runtime_state, dict):
            return heading_indices
        snapshot = runtime_state.get("last_page_snapshot")
        headings_payload = snapshot.get("headings") if isinstance(snapshot, dict) else []
        candidate_meta: list[dict[str, Any]] = []
        heading_norm = cls._normalize_line(heading_text).lower()
        for item in headings_payload if isinstance(headings_payload, list) else []:
            if not isinstance(item, dict):
                continue
            text = cls._normalize_line(item.get("text", "")).lower()
            if text != heading_norm:
                continue
            candidate_meta.append(
                {
                    "region": str(item.get("region", "unknown") or "unknown").lower(),
                    "line_count_after": int(item.get("line_count_after", 0) or 0),
                    "preview_after": [cls._normalize_line(v) for v in item.get("preview_after", []) if cls._normalize_line(v)],
                }
            )
        if not candidate_meta:
            return heading_indices
        scored: list[tuple[int, int]] = []
        for order, idx in enumerate(heading_indices):
            meta = candidate_meta[min(order, len(candidate_meta) - 1)]
            region = meta["region"]
            region_score = 2 if region in {"main", "article", "content"} else (1 if region == "unknown" else -2)
            content_score = 2 if meta["line_count_after"] > 0 and meta["preview_after"] else 0
            scored.append((idx, region_score + content_score))
        return [idx for idx, _score in sorted(scored, key=lambda item: item[1], reverse=True)]

    @staticmethod
    def _infer_href_slug_from_text(text: str) -> str:
        token = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")
        return token[:48]

    @classmethod
    def _is_meta_click_label(cls, value: str) -> bool:
        token = re.sub(r"\s+", " ", str(value or "").strip().lower()).strip(" :.-_")
        return bool(token) and token in cls._CLICK_META_LABELS

    async def _discover_href_from_visible_links(self, *, page, text: str) -> str | None:
        needle = re.sub(r"\s+", " ", str(text or "").strip().lower())
        if not needle:
            return None
        payload = await page.evaluate(
            """
            () => Array.from(document.querySelectorAll("a[href]")).slice(0, 250).map((a) => ({
              text: (a.innerText || a.textContent || "").replace(/\\s+/g, " ").trim(),
              href: a.getAttribute("href") || ""
            }))
            """
        )
        for item in payload or []:
            label = str(item.get("text", "")).strip().lower()
            href = str(item.get("href", "")).strip()
            if not label or not href:
                continue
            if needle in label:
                return href
        return None

    async def _extract_table_rows(self, *, page, limit: int) -> list[list[str]]:
        rows = await page.evaluate(
            """
            ({ limit }) => {
              const results = [];
              const trNodes = document.querySelectorAll("table tr");
              for (const tr of trNodes) {
                const cells = Array.from(tr.querySelectorAll("th,td"))
                  .map((cell) => (cell.innerText || cell.textContent || "").replace(/\\s+/g, " ").trim())
                  .filter(Boolean);
                if (cells.length < 2) continue;
                results.push(cells);
                if (results.length >= limit) break;
              }
              return results;
            }
            """,
            {"limit": max(limit, 1)},
        )
        return [list(row) for row in (rows or []) if isinstance(row, list) and len(row) >= 2]

    async def _extract_table_rows_as_dicts(self, *, page, limit: int) -> list[dict[str, Any]]:
        rows = await page.evaluate(
            """
            ({ limit }) => {
              const norm = (value) => String(value || "").replace(/\\s+/g, " ").trim();
              const cssEscape = (value) => {
                if (window.CSS && CSS.escape) return CSS.escape(value);
                return String(value).replace(/[^a-zA-Z0-9_-]/g, "\\\\$&");
              };
              const cssPath = (el) => {
                if (!el || !el.tagName) return "";
                if (el.id) return `#${cssEscape(el.id)}`;
                const parts = [];
                let cur = el;
                while (cur && cur.nodeType === Node.ELEMENT_NODE && cur !== document.body && parts.length < 6) {
                  const tag = cur.tagName.toLowerCase();
                  let part = tag;
                  const cls = Array.from(cur.classList || []).slice(0, 2).map(cssEscape);
                  if (cls.length) part += "." + cls.join(".");
                  const parent = cur.parentElement;
                  if (parent) {
                    const siblings = Array.from(parent.children).filter((sib) => sib.tagName === cur.tagName);
                    if (siblings.length > 1) part += `:nth-of-type(${siblings.indexOf(cur) + 1})`;
                  }
                  parts.unshift(part);
                  cur = parent;
                }
                return parts.join(" > ");
              };
              const tables = Array.from(document.querySelectorAll("table"));
              const results = [];
              for (let tableIndex = 0; tableIndex < tables.length; tableIndex++) {
                const table = tables[tableIndex];
                const allRows = Array.from(table.querySelectorAll("tr"));
                let headers = [];
                const headerRow = allRows.find((tr) => tr.querySelectorAll("th").length >= 2);
                if (headerRow) {
                  headers = Array.from(headerRow.querySelectorAll("th,td")).map((cell) => norm(cell.innerText || cell.textContent || "")).filter(Boolean);
                }
                for (const tr of allRows) {
                  const cells = Array.from(tr.querySelectorAll("td,th")).map((cell) => norm(cell.innerText || cell.textContent || "")).filter(Boolean);
                  if (cells.length < 2) continue;
                  if (headers.length && cells.join("|") === headers.join("|")) continue;
                  results.push({
                    headers,
                    cells,
                    text: cells.join(" "),
                    row_id: `table_${tableIndex + 1}_row_${allRows.indexOf(tr) + 1}`,
                    table_index: tableIndex,
                    row_index: allRows.indexOf(tr),
                    selector: cssPath(tr)
                  });
                  if (results.length >= limit) return results;
                }
              }
              return results;
            }
            """,
            {"limit": max(limit, 1)},
        )
        normalized: list[dict[str, Any]] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            normalized.append(self._build_row_payload(row))
        return normalized

    async def _extract_table_like_rows(self, *, page, limit: int) -> list[list[str]]:
        table_like = await page.evaluate(
            """
            ({ limit }) => {
              const normalizedText = (value) => (value || "").replace(/\\s+/g, " ").trim();
              const inContent = (el) => !!el.closest("main, article, [role='main'], #content, .content");
              const inIgnoredRegion = (el) => !!el.closest("nav, header, footer, aside, [role='navigation']");
              const rows = [];
              const candidates = document.querySelectorAll("main dl, article dl, #content dl, .content dl, main ul li, article ul li, #content ul li, .content ul li, main ol li, article ol li");
              for (const node of candidates) {
                if (!inContent(node) || inIgnoredRegion(node)) continue;
                const text = normalizedText(node.innerText || node.textContent || "");
                if (!text || text.length < 4) continue;
                const anchor = node.querySelector ? node.querySelector("a[href]") : null;
                const href = normalizedText(anchor ? anchor.getAttribute("href") : "");
                const title = normalizedText(anchor ? (anchor.innerText || anchor.textContent || "") : "");
                const row = [];
                if (title) row.push(title);
                if (href) row.push(href);
                if (!title || text !== title) row.push(text);
                if (row.length < 2) continue;
                rows.push(row);
                if (rows.length >= limit) break;
              }
              return rows;
            }
            """,
            {"limit": max(limit, 1)},
        )
        return [list(row) for row in (table_like or []) if isinstance(row, list) and len(row) >= 2]

    async def _extract_repeated_link_or_list_items(self, *, page, limit: int) -> list[list[str]]:
        rows = await page.evaluate(
            """
            ({ limit }) => {
              const normalizedText = (value) => (value || "").replace(/\\s+/g, " ").trim();
              const inIgnoredRegion = (el) => !!el.closest("nav, header, footer, aside, [role='navigation']");
              const inContent = (el) => !!el.closest("main, article, [role='main'], #content, .content");
              const results = [];
              const linkNodes = document.querySelectorAll("main a[href], article a[href], ul li, ol li");
              for (const node of linkNodes) {
                if (inIgnoredRegion(node) || !inContent(node)) continue;
                const text = normalizedText(node.innerText || node.textContent || "");
                if (!text || text.length < 2) continue;
                const href = node.getAttribute ? (node.getAttribute("href") || "") : "";
                const anchor = node.querySelector ? node.querySelector("a[href]") : null;
                const resolvedHref = normalizedText(href || (anchor ? anchor.getAttribute("href") : ""));
                const linkText = normalizedText(anchor ? (anchor.innerText || anchor.textContent || "") : "");
                if (text.length < 4 && !resolvedHref) continue;
                results.push(href ? [text, href] : [text]);
                if (linkText && resolvedHref) {
                  results[results.length - 1] = [linkText || text, resolvedHref, text];
                }
                if (results.length >= limit) break;
              }
              return results;
            }
            """,
            {"limit": max(limit, 1)},
        )
        return [list(row) for row in (rows or []) if isinstance(row, list) and row]

    async def _extract_repeated_entity_blocks(self, *, page, limit: int) -> list[dict[str, str]]:
        rows = await page.evaluate(
            """
            ({ limit }) => {
              const normalizedText = (value) => (value || "").replace(/\\s+/g, " ").trim();
              const candidates = [];
              const nodes = document.querySelectorAll("main li, article li, main article, article article, section li, section article, main div, article div");
              for (const node of nodes) {
                const text = normalizedText(node.innerText || node.textContent || "");
                if (!text || text.length < 8 || text.length > 260) continue;
                const directHref = node.getAttribute ? (node.getAttribute("href") || "") : "";
                const nestedAnchor = node.querySelector ? node.querySelector("a[href]") : null;
                const href = normalizedText(directHref || (nestedAnchor ? nestedAnchor.getAttribute("href") : ""));
                const linkText = normalizedText(nestedAnchor ? (nestedAnchor.innerText || nestedAnchor.textContent || "") : "");
                candidates.push({
                  text,
                  raw_text: text,
                  href,
                  link_text: linkText
                });
                if (candidates.length >= Math.max(limit * 10, 60)) break;
              }
              return candidates;
            }
            """,
            {"limit": max(limit, 1)},
        )
        cleaned: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            raw_text = self._normalize_line(str(row.get("raw_text", "") or row.get("text", "")))
            if not raw_text:
                continue
            fingerprint = raw_text.lower()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            href = self._normalize_line(str(row.get("href", "")))
            title = self._normalize_line(str(row.get("link_text", "")))
            item: dict[str, str] = {"raw_text": raw_text, "text": raw_text}
            if href:
                item["href"] = href
                item["link"] = href
            if title:
                item["title"] = title
                item["name"] = title
            if self._looks_like_navigation_item(raw_text):
                continue
            cleaned.append(item)
        ranked = sorted(
            cleaned,
            key=lambda item: (
                int(bool(item.get("title"))),
                int(bool(item.get("href"))),
                len(item.get("raw_text", "")),
            ),
            reverse=True,
        )
        return ranked[: max(limit, 1)]

    @classmethod
    def _normalize_table_rows_to_objects(cls, *, rows: list[list[str]], limit: int) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for row in rows[: max(limit, 1)]:
            if not row:
                continue
            normalized_cells = [cls._normalize_line(str(cell)) for cell in row if cls._normalize_line(str(cell))]
            if not normalized_cells:
                continue
            title = normalized_cells[0]
            href = next((cell for cell in normalized_cells if cls._looks_like_href_token(cell)), "")
            raw_text = " | ".join(normalized_cells)
            item: dict[str, Any] = {
                "title": title,
                "name": title,
                "raw_text": raw_text,
                "text": raw_text,
            }
            if href:
                item["href"] = href
                item["link"] = href
            item["columns"] = normalized_cells
            items.append(item)
        return items

    @classmethod
    def _project_structured_rows_to_fields(cls, *, rows: list[list[str]], fields: Any, limit: int) -> list[dict[str, Any]]:
        if not rows:
            return []
        if isinstance(fields, dict) and fields:
            names = [str(name).strip() for name in fields.keys() if str(name).strip()]
        elif isinstance(fields, list) and fields:
            names = [str(name).strip() for name in fields if str(name).strip()]
        else:
            names = ["name", "detail"]
        items: list[dict[str, Any]] = []
        for row in rows[: max(limit, 1)]:
            if not row:
                continue
            item: dict[str, Any] = {}
            for idx, name in enumerate(names):
                if idx < len(row):
                    item[name] = row[idx]
                elif row:
                    item[name] = row[-1]
            if item:
                items.append(item)
        return items

    @classmethod
    def _score_structured_fallback_quality(cls, *, items: list[dict[str, Any]], limit: int, fallback_kind: str) -> dict[str, Any]:
        count = len(items)
        if count == 0:
            return {"score": 0.0, "grade": "low", "is_acceptable": False}

        meaningful_keys = {"name", "title", "description", "snippet", "href", "url", "link", "detail", "text", "raw_text", "columns"}
        non_empty = sum(1 for item in items if any(str(v).strip() for v in item.values()))
        rich = 0
        raw_only = 0
        nav_like = 0
        short_item_count = 0
        duplicate_count = 0
        with_anchor_signals = 0
        signatures: list[str] = []
        fingerprints: set[str] = set()
        for item in items:
            present = {
                str(key).lower()
                for key, value in item.items()
                if value is not None and str(value).strip() and str(key).lower() in meaningful_keys
            }
            if len(present) >= 2:
                rich += 1
            text = str(item.get("raw_text") or item.get("text") or item.get("name") or "")
            normalized_text = cls._normalize_line(text).lower()
            if len(normalized_text) < 6:
                short_item_count += 1
            if normalized_text:
                if normalized_text in fingerprints:
                    duplicate_count += 1
                fingerprints.add(normalized_text)
            has_structural = any(
                bool(item.get(key))
                for key in ("href", "link", "title", "name", "columns")
            ) or (len(present) >= 2)
            if not has_structural:
                raw_only += 1
            if cls._looks_like_navigation_item(text):
                nav_like += 1
            if any(bool(item.get(key)) for key in ("href", "link", "title", "name", "columns")):
                with_anchor_signals += 1
            signature = "|".join(sorted(present))
            if signature:
                signatures.append(signature)

        dominant_ratio = 0.0
        if signatures:
            dominant_count = max(signatures.count(sig) for sig in set(signatures))
            dominant_ratio = dominant_count / max(len(signatures), 1)

        min_items = min(max(limit, 1), 3)
        count_score = min(count / max(min_items, 1), 1.0)
        non_empty_ratio = non_empty / max(count, 1)
        rich_ratio = rich / max(count, 1)
        raw_only_ratio = raw_only / max(count, 1)
        nav_ratio = nav_like / max(count, 1)
        short_ratio = short_item_count / max(count, 1)
        duplicate_ratio = duplicate_count / max(count, 1)
        anchor_signal_ratio = with_anchor_signals / max(count, 1)

        score = (
            0.24 * count_score
            + 0.15 * non_empty_ratio
            + 0.23 * rich_ratio
            + 0.14 * dominant_ratio
            + 0.16 * anchor_signal_ratio
            - 0.18 * raw_only_ratio
            - 0.16 * nav_ratio
            - 0.10 * short_ratio
            - 0.10 * duplicate_ratio
        )
        if fallback_kind == "table_rows":
            table_column_rich = sum(1 for item in items if len(item.get("columns", []) if isinstance(item.get("columns"), list) else []) >= 2)
            score += 0.08 + 0.12 * (table_column_rich / max(count, 1))
        if fallback_kind == "table_like_rows":
            score += 0.08
        if fallback_kind == "repeated_entity_blocks":
            entity_rich = sum(1 for item in items if item.get("href") or item.get("title"))
            score += 0.05 + 0.08 * (entity_rich / max(count, 1))
        if fallback_kind == "list_items" and (nav_ratio >= 0.35 or raw_only_ratio >= 0.5 or anchor_signal_ratio < 0.5):
            score -= 0.20
        score = max(0.0, min(score, 1.0))
        grade = "high" if score >= 0.70 else ("medium" if score >= 0.52 else "low")
        is_acceptable = grade == "high" if fallback_kind == "list_items" else grade in {"high", "medium"}
        return {"score": score, "grade": grade, "is_acceptable": is_acceptable}

    @staticmethod
    def _looks_like_href_token(value: str) -> bool:
        token = str(value or "").strip().lower()
        return bool(token) and (
            token.startswith("/")
            or token.startswith("./")
            or token.startswith("../")
            or token.startswith("http://")
            or token.startswith("https://")
        )

    @classmethod
    def _looks_like_navigation_item(cls, text: str) -> bool:
        token = cls._normalize_line(text).lower()
        if not token:
            return True
        nav_markers = {
            "home",
            "about",
            "privacy",
            "terms",
            "menu",
            "skip to content",
            "sign in",
            "login",
            "next",
            "previous",
            "copyright",
            "footer",
        }
        if token in nav_markers:
            return True
        if len(token) <= 2:
            return True
        if re.fullmatch(r"[›»«•|/\\-]+", token):
            return True
        return False

    @staticmethod
    def _extract_match_value(match: re.Match[str], group_index: int | None):
        try:
            if group_index is not None:
                return match.group(int(group_index))
            return match.group(1) if match.groups() else match.group(0)
        except IndexError as exc:
            requested = int(group_index) if group_index is not None else 1
            available_groups = len(match.groups())
            raise StructuredExtractionError(
                code="regex_group_mismatch",
                message=(
                    "Regex group reference is out of range during extraction: "
                    f"requested_group={requested}, available_groups={available_groups}. "
                    "Check pattern/group_index consistency."
                ),
                details={"requested_group": requested, "available_groups": available_groups},
            ) from exc

    @staticmethod
    def _normalize_line(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n-•*|")

    @classmethod
    def _split_visible_lines(cls, text: str) -> list[str]:
        raw_lines = re.split(r"[\r\n]+", str(text or ""))
        lines: list[str] = []
        for line in raw_lines:
            normalized = cls._normalize_line(line)
            if normalized:
                lines.append(normalized)
        return lines

    @classmethod
    def _looks_like_heading_line(cls, line: str) -> bool:
        token = cls._normalize_line(line)
        if not token:
            return False
        if len(token) <= 80 and re.fullmatch(r"[A-Za-zА-Яа-я0-9][A-Za-zА-Яа-я0-9\s/&:+()\-]{1,79}", token):
            if token.endswith(":"):
                return True
            words = token.split()
            if 1 <= len(words) <= 8 and sum(ch.isalpha() for ch in token) >= 3:
                return True
        return False

    async def _collect_anchor_candidates(
        self,
        *,
        page,
        anchor_text: str,
        search_direction: str,
        same_block_only: bool,
        max_distance_chars: int | None,
        matching_mode: str = "auto",
        runtime_state=None,
    ) -> list[dict[str, Any]]:
        window_chars = max_distance_chars if isinstance(max_distance_chars, int) and max_distance_chars > 0 else 600
        candidates = await page.evaluate(
            r"""
            ({ anchorText, direction, sameBlockOnly, windowChars, matchingMode }) => {
              const normalizeText = (text) => (text || "").replace(/\s+/g, " ").trim();
              const sectionSelector = "section, article, main, aside, footer, header, nav, form, dl, table";
              const blockSelector = "p, li, dt, dd, td, th, div, article, section";
              const reasonableSelector = "li, tr, td, th, p, dt, dd, article, section, div, span, a";
              const collect = [];
              const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
              const anchorLower = (anchorText || "").toLowerCase();
              let node;
              while ((node = walker.nextNode())) {
                const value = normalizeText(node.nodeValue || "");
                if (!value) continue;
                if (!value.toLowerCase().includes(anchorLower)) continue;
                const host = node.parentElement;
                if (!host) continue;

                const containerChain = [];
                if (sameBlockOnly) {
                  const localBlock = host.closest(blockSelector) || host.closest(reasonableSelector) || host;
                  const sectionBlock = host.closest(sectionSelector);
                  if (localBlock) containerChain.push({ node: localBlock, source: "dom_local_block", sourceRank: 0 });
                  if (sectionBlock && sectionBlock !== localBlock) {
                    containerChain.push({ node: sectionBlock, source: "dom_section_block", sourceRank: 1 });
                  }
                } else {
                  const broad = host.closest(sectionSelector) || document.body;
                  containerChain.push({ node: broad, source: "dom_broad_block", sourceRank: 1 });
                  containerChain.push({ node: document.body, source: "dom_page", sourceRank: 2 });
                }

                for (const entry of containerChain) {
                  const container = entry.node;
                  const text = normalizeText(container?.innerText || container?.textContent || "");
                  if (!text) continue;
                  const lower = text.toLowerCase();
                  let anchorIdx = lower.indexOf(anchorLower);
                  if (anchorIdx < 0 && matchingMode === "contains") {
                    const token = anchorLower.split(" ").filter(Boolean)[0] || "";
                    anchorIdx = token ? lower.indexOf(token) : -1;
                  }
                  if (anchorIdx < 0) continue;

                  let start = 0;
                  let end = text.length;
                  if (direction === "after") {
                    start = anchorIdx;
                    end = Math.min(text.length, anchorIdx + (anchorText || "").length + windowChars);
                  } else if (direction === "before") {
                    start = Math.max(0, anchorIdx - windowChars);
                    end = anchorIdx + (anchorText || "").length;
                  } else {
                    start = Math.max(0, anchorIdx - windowChars);
                    end = Math.min(text.length, anchorIdx + (anchorText || "").length + windowChars);
                  }

                  collect.push({
                    source: entry.source,
                    source_rank: entry.sourceRank,
                    block_selector: container.tagName ? container.tagName.toLowerCase() : "unknown",
                    scope_text: text,
                    anchor_idx_in_scope: anchorIdx,
                    window_text: text.slice(start, end),
                    anchor_idx_in_window: anchorIdx - start,
                  });
                }
              }
              return collect;
            }
            """,
            {
                "anchorText": anchor_text,
                "direction": search_direction,
                "sameBlockOnly": same_block_only,
                "windowChars": window_chars,
                "matchingMode": matching_mode,
            },
        )

        if candidates:
            return sorted(
                candidates,
                key=lambda item: (
                    int(item.get("source_rank", 3)),
                    0 if str(item.get("source", "")).startswith("dom_") else 1,
                ),
            )

        source_text = await self._load_source_text(page=page, runtime_state=runtime_state)
        flags = re.IGNORECASE
        anchor_match = re.search(re.escape(anchor_text), source_text, flags=flags)
        if not anchor_match:
            raise ValueError(f"Anchor text not found: {anchor_text}")
        start, end = 0, len(source_text)
        if search_direction == "after":
            start = anchor_match.start()
            end = min(len(source_text), anchor_match.end() + window_chars)
        elif search_direction == "before":
            start = max(0, anchor_match.start() - window_chars)
            end = anchor_match.end()
        else:
            start = max(0, anchor_match.start() - window_chars)
            end = min(len(source_text), anchor_match.end() + window_chars)

        return [
            {
                "source": "page_text_anchor_fallback",
                "source_rank": 3,
                "scope_text": source_text,
                "anchor_idx_in_scope": anchor_match.start(),
                "window_text": source_text[start:end],
                "anchor_idx_in_window": anchor_match.start() - start,
            }
        ]

    def _select_best_value_near_anchor(
        self,
        *,
        candidates: list[dict[str, Any]],
        value_pattern: str,
        search_direction: str,
        required_right_context: str | None,
        required_left_context: str | None,
        max_distance_chars: int | None,
        flags_value: int,
        group_index: int | None,
        prefer_local_match: bool = False,
    ) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None

        for candidate in candidates:
            window_text = str(candidate.get("window_text") or "")
            scope_text = str(candidate.get("scope_text") or window_text)
            anchor_idx = int(candidate.get("anchor_idx_in_scope", candidate.get("anchor_idx_in_window", 0)))
            for match in re.finditer(value_pattern, scope_text, flags=flags_value):
                if not self._contexts_match(
                    window_text=scope_text,
                    match=match,
                    required_left_context=required_left_context,
                    required_right_context=required_right_context,
                    flags_value=flags_value,
                ):
                    continue

                distance = self._distance_from_anchor(
                    anchor_idx=anchor_idx,
                    match=match,
                    direction=search_direction,
                )
                if distance is None:
                    continue
                if isinstance(max_distance_chars, int) and max_distance_chars > 0 and distance > max_distance_chars:
                    continue

                extracted_value = self._extract_match_value(match, group_index=group_index)
                match_scope = self._classify_anchor_source_scope(str(candidate.get("source", "")))
                confidence = self._classify_anchor_match_confidence(
                    distance=distance,
                    match_scope=match_scope,
                )
                if prefer_local_match and match_scope == "fallback" and confidence != "high":
                    continue
                score = (
                    0 if confidence == "high" else (1 if confidence == "medium" else 2),
                    0 if match_scope == "local_block" else (1 if match_scope == "section" else 2),
                    distance,
                    int(candidate.get("source_rank", 3)),
                    0 if str(candidate.get("source", "")).startswith("dom_") else 1,
                )
                if best is None or score < best["score"]:
                    best = {
                        "value": extracted_value,
                        "distance": distance,
                        "source": candidate.get("source", "unknown"),
                        "match_scope": match_scope,
                        "confidence": confidence,
                        "score": score,
                    }
        return best

    @staticmethod
    def _classify_anchor_source_scope(source: str) -> str:
        lowered = source.strip().lower()
        if lowered == "dom_local_block":
            return "local_block"
        if lowered in {"dom_section_block", "dom_broad_block"}:
            return "section"
        return "fallback"

    @staticmethod
    def _classify_anchor_match_confidence(*, distance: int, match_scope: str) -> str:
        if match_scope == "local_block":
            if distance <= 140:
                return "high"
            if distance <= 260:
                return "medium"
            return "low"
        if match_scope == "section":
            if distance <= 220:
                return "high"
            if distance <= 400:
                return "medium"
            return "low"
        if distance <= 120:
            return "high"
        if distance <= 220:
            return "medium"
        return "low"

    @staticmethod
    def _build_comparison_side_summary(*, value: Any, label: str) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "label": label,
            "type": type(value).__name__,
            "is_null": value is None,
        }
        if isinstance(value, dict):
            summary["keys"] = sorted(str(key) for key in value.keys())
            summary["size"] = len(value)
        elif isinstance(value, list):
            summary["size"] = len(value)
            summary["item_types"] = sorted({type(item).__name__ for item in value})
        elif isinstance(value, str):
            summary["length"] = len(value)
            summary["preview"] = value[:120]
        else:
            summary["repr"] = repr(value)[:120]
        return summary

    @staticmethod
    def _is_heading_selector(selector: str) -> bool:
        normalized = selector.strip().lower()
        return normalized in {"h1", "main h1", "article h1", "header h1"}

    async def _resolve_primary_heading_locator(self, *, page, fallback_locator):
        selected_index = await page.evaluate(
            r"""
            () => {
              const headings = Array.from(document.querySelectorAll("h1"));
              if (!headings.length) return -1;
              const visible = (el) => {
                if (!el || !el.isConnected) return false;
                if (el.closest("[hidden], [aria-hidden='true']")) return false;
                const style = window.getComputedStyle(el);
                if (!style || style.display === "none" || style.visibility === "hidden" || Number(style.opacity || "1") === 0) {
                  return false;
                }
                const rect = el.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0;
              };
              let best = null;
              headings.forEach((el, idx) => {
                const text = (el.innerText || el.textContent || "").replace(/\s+/g, " ").trim();
                if (!text || !visible(el)) return;
                const inMain = !!el.closest("main");
                const inArticle = !!el.closest("article");
                const inHeader = !!el.closest("header");
                const inNavLike = !!el.closest("nav, aside, footer");
                const rect = el.getBoundingClientRect();
                let bucket = 3;
                if (inMain) bucket = 0;
                else if (inArticle) bucket = 1;
                else if (inHeader) bucket = 2;
                let score = 1000 - bucket * 100;
                if (inNavLike) score -= 300;
                score -= Math.max(0, Math.floor(rect.top));
                const candidate = { idx, bucket, score, top: rect.top };
                if (!best || candidate.score > best.score || (candidate.score === best.score && candidate.top < best.top)) {
                  best = candidate;
                }
              });
              return best ? best.idx : -1;
            }
            """
        )
        if isinstance(selected_index, int) and selected_index >= 0:
            note = (
                'Selector "h1" used primary heading resolution '
                f"(priority: main>article>header>h1); selected index={selected_index}."
            )
            return page.locator("h1").nth(selected_index), note

        count = await fallback_locator.count()
        note = f'Selector "h1" primary heading resolution failed; fallback to first of {count}.'
        return fallback_locator.first, note

    async def _extract_text_with_heading_fallback(self, *, page, selector: str, fallback_locator):
        primary_count = await fallback_locator.count()
        if primary_count > 0:
            try:
                target_locator, note = await self._resolve_primary_heading_locator(page=page, fallback_locator=fallback_locator)
                await target_locator.wait_for(state="visible", timeout=1200)
                text = self._normalize_line(await target_locator.inner_text())
                if text:
                    return text, f"{note} extract_text fallback=visible_h1"
            except Exception:  # noqa: BLE001
                pass

        fallback_payload = await page.evaluate(
            r"""
            () => {
              const normalize = (value) => (value || "").replace(/\s+/g, " ").trim();
              const isVisible = (el) => {
                if (!el || !el.isConnected) return false;
                if (el.closest("[hidden], [aria-hidden='true']")) return false;
                const style = window.getComputedStyle(el);
                if (!style || style.display === "none" || style.visibility === "hidden" || Number(style.opacity || "1") === 0) {
                  return false;
                }
                const rect = el.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0;
              };
              const firstMeaningfulText = (root) => {
                if (!root) return "";
                const textCandidates = root.querySelectorAll("h1,h2,h3,p,li,dt,dd,div,span,a");
                for (const node of textCandidates) {
                  if (!isVisible(node)) continue;
                  const text = normalize(node.innerText || node.textContent || "");
                  if (text.length >= 12) return text;
                }
                const rootText = normalize(root.innerText || root.textContent || "");
                return rootText.length >= 12 ? rootText : "";
              };

              const headingSelectors = ["h1", "h2"];
              for (const sel of headingSelectors) {
                const nodes = Array.from(document.querySelectorAll(sel));
                for (const node of nodes) {
                  if (!isVisible(node)) continue;
                  const text = normalize(node.innerText || node.textContent || "");
                  if (text) return { value: text, fallback: `visible_${sel}` };
                }
              }

              const headingLikeSelectors = [
                "[class*='title']",
                "[class*='Title']",
                "[class*='heading']",
                "[class*='header']"
              ];
              for (const sel of headingLikeSelectors) {
                const nodes = Array.from(document.querySelectorAll(sel));
                for (const node of nodes) {
                  if (!isVisible(node)) continue;
                  const text = normalize(node.innerText || node.textContent || "");
                  if (text.length >= 4) return { value: text, fallback: "visible_heading_like" };
                }
              }

              const contentRoots = Array.from(document.querySelectorAll("main, section, article"));
              for (const root of contentRoots) {
                if (!isVisible(root)) continue;
                const text = firstMeaningfulText(root);
                if (text) return { value: text, fallback: "visible_main_section_article_text" };
              }

              const title = normalize(document.title || "");
              if (title) return { value: title, fallback: "document_title" };

              const ogTitle = normalize(document.querySelector("meta[property='og:title']")?.getAttribute("content") || "");
              if (ogTitle) return { value: ogTitle, fallback: "meta_og_title" };
              const metaTitle = normalize(document.querySelector("meta[name='title']")?.getAttribute("content") || "");
              if (metaTitle) return { value: metaTitle, fallback: "meta_title" };

              return { value: "", fallback: "none" };
            }
            """
        )
        value = self._normalize_line(str((fallback_payload or {}).get("value", "")))
        fallback_name = str((fallback_payload or {}).get("fallback", "none"))
        if value:
            note = (
                f'Selector "{selector}" heading fallback resolution used fallback={fallback_name}; '
                f"primary_match_count={primary_count}."
            )
            return value, note
        return (
            self._normalize_line(await page.title()),
            f'Selector "{selector}" fallback chain exhausted; fallback=document_title.',
        )

    @staticmethod
    def _is_empty_structured_value(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not bool(value.strip())
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0
        return False

    @staticmethod
    def _distance_from_anchor(anchor_idx: int, match: re.Match[str], direction: str) -> int | None:
        if direction == "after":
            if match.start() < anchor_idx:
                return None
            return match.start() - anchor_idx
        if direction == "before":
            if match.end() > anchor_idx:
                return None
            return anchor_idx - match.end()
        if match.start() >= anchor_idx:
            return match.start() - anchor_idx
        return anchor_idx - match.end()

    @staticmethod
    def _contexts_match(
        *,
        window_text: str,
        match: re.Match[str],
        required_left_context: str | None,
        required_right_context: str | None,
        flags_value: int,
    ) -> bool:
        left_text = window_text[: match.start()]
        right_text = window_text[match.end() :]
        if required_left_context:
            if not re.search(re.escape(required_left_context), left_text, flags=flags_value):
                return False
        if required_right_context:
            if not re.search(re.escape(required_right_context), right_text, flags=flags_value):
                return False
        return True

    @staticmethod
    def _normalize_number_token(value: str, *, number_type: str, strip_plus: bool):
        candidate = str(value).strip()
        if strip_plus:
            candidate = candidate.rstrip("+").strip()

        grouped_pattern = r"^\d{1,3}(?:[ \t,\.\u00A0\u202F]\d{3})+$"
        plain_integer = r"^\d+$"

        if number_type == "int":
            if re.fullmatch(grouped_pattern, candidate):
                compact = re.sub(r"[ \t,\.\u00A0\u202F]", "", candidate)
            elif re.fullmatch(plain_integer, candidate):
                compact = candidate
            else:
                raise ValueError(f"Value is not grouped/plain integer-like: {value!r}")
            if not re.fullmatch(plain_integer, compact):
                raise ValueError(f"Normalized value is not integer-like: {value!r} -> {compact!r}")
            return int(compact)

        if number_type == "float":
            compact = re.sub(r"[ \t\u00A0\u202F]", "", candidate)
            if not re.fullmatch(r"^\d+(?:[.,]\d+)?$", compact):
                raise ValueError(f"Value is not float-like: {value!r}")
            return float(compact.replace(",", "."))

        raise ValueError(f"Unsupported number_type: {number_type}")

    @staticmethod
    def _record_pattern_artifact(
        *,
        runtime_state,
        pattern: str,
        success: bool,
        reason: str | None = None,
        raw_match: str | None = None,
        normalized_value: int | None = None,
        group_index: int | None = None,
        occurrence: int | None = None,
    ) -> None:
        if runtime_state is None:
            return
        artifacts = runtime_state.setdefault("pattern_extractions", [])
        artifacts.append(
            {
                "pattern": pattern,
                "success": success,
                "reason": reason,
                "raw_match": raw_match,
                "normalized_value": normalized_value,
                "group_index": group_index,
                "occurrence": occurrence,
            }
        )

    async def _extract_field_value(self, container, field_name: str, rule: Any):
        selector = None
        attr = None
        pattern = None
        group_index = None
        normalize_number = False
        number_type = "int"
        strip_plus = True
        anchor_text = None
        value_pattern = None
        search_direction = "after"
        required_right_context = None
        required_left_context = None
        max_distance_chars = None
        ignore_case = True

        if isinstance(rule, str):
            if rule.endswith(".href"):
                selector = rule[: -len(".href")]
                attr = "href"
            else:
                selector = rule
                if rule == "a" and "link" in field_name.lower():
                    attr = "href"
        elif isinstance(rule, dict):
            selector = rule.get("selector")
            attr = rule.get("attr") or rule.get("attribute")
            if isinstance(attr, str) and attr.strip().lower() in {"text", "inner_text", "innertext"}:
                attr = None
            pattern = rule.get("pattern")
            group_index = rule.get("group_index")
            normalize_number = bool(rule.get("normalize_number", False))
            number_type = str(rule.get("number_type", "int")).lower()
            strip_plus = bool(rule.get("strip_plus", True))
            anchor_text = rule.get("anchor_text")
            value_pattern = rule.get("value_pattern")
            search_direction = str(rule.get("search_direction", "after")).lower()
            required_right_context = rule.get("required_right_context")
            required_left_context = rule.get("required_left_context")
            max_distance_chars = rule.get("max_distance_chars")
            ignore_case = bool(rule.get("ignore_case", True))

        target = container
        if selector:
            target = container.locator(selector).first

        if attr:
            value = await target.get_attribute(attr)
            return (value or "").strip()

        text = (await target.inner_text()).strip()
        if not text:
            return None

        flags_value = re.IGNORECASE if ignore_case else 0
        if anchor_text and value_pattern:
            match = self._match_value_near_anchor_in_text(
                text=text,
                anchor_text=str(anchor_text),
                value_pattern=str(value_pattern),
                search_direction=search_direction,
                required_right_context=None if required_right_context is None else str(required_right_context),
                required_left_context=None if required_left_context is None else str(required_left_context),
                max_distance_chars=(
                    int(max_distance_chars) if max_distance_chars is not None else None
                ),
                flags_value=flags_value,
            )
            if match is None:
                return None
            extracted = self._extract_match_value(match, group_index=group_index)
        elif pattern:
            match = re.search(str(pattern), text, flags=flags_value)
            if not match:
                return None
            extracted = self._extract_match_value(match, group_index=group_index)
        else:
            extracted = text

        if normalize_number:
            return self._normalize_number_token(
                extracted,
                number_type=number_type,
                strip_plus=strip_plus,
            )

        return extracted

    def _match_value_near_anchor_in_text(
        self,
        *,
        text: str,
        anchor_text: str,
        value_pattern: str,
        search_direction: str,
        required_right_context: str | None,
        required_left_context: str | None,
        max_distance_chars: int | None,
        flags_value: int,
    ):
        anchor_match = re.search(re.escape(anchor_text), text, flags=flags_value)
        if not anchor_match:
            return None

        anchor_idx = anchor_match.start()
        best = None
        for match in re.finditer(value_pattern, text, flags=flags_value):
            if not self._contexts_match(
                window_text=text,
                match=match,
                required_left_context=required_left_context,
                required_right_context=required_right_context,
                flags_value=flags_value,
            ):
                continue
            distance = self._distance_from_anchor(anchor_idx=anchor_idx, match=match, direction=search_direction)
            if distance is None:
                continue
            if isinstance(max_distance_chars, int) and max_distance_chars > 0 and distance > max_distance_chars:
                continue
            if best is None or distance < best[0]:
                best = (distance, match)
        return None if best is None else best[1]

    async def screenshot(self, page, args, runtime_state=None):
        path = args["path"]
        await page.screenshot(path=path)
        return path
