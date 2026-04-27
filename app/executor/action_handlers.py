import asyncio
import re
from datetime import datetime, timezone
from typing import Any

from app.observer.page_observer import PageObserver


class StructuredExtractionError(ValueError):
    def __init__(self, *, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class ActionHandlers:
    def __init__(self):
        self.page_observer = PageObserver()

    async def open_url(self, page, args, runtime_state=None):
        wait_until = str(args.get("wait_until", "domcontentloaded"))
        timeout_ms = int(args.get("timeout_ms", 20000))
        await page.goto(args["url"], wait_until=wait_until, timeout=timeout_ms)

    async def click(self, page, args, runtime_state=None):
        locator, candidate = await self._resolve_ranked_click_locator(page=page, args=args, runtime_state=runtime_state)
        await self._wait_for_actionable(locator)
        await locator.first.click()
        args["_executor_note"] = (
            f"click used ranked locator strategy={candidate.get('strategy')}; selector={candidate.get('selector')!r}; "
            f"candidates_checked={candidate.get('candidates_checked', 0)}"
        )

    async def type(self, page, args, runtime_state=None):
        await page.fill(args["selector"], args["text"])

    async def wait_for(self, page, args, runtime_state=None):
        timeout_ms = int(args.get("timeout_ms", 12000))
        if "selector" in args and str(args.get("selector", "")).strip():
            await page.wait_for_selector(args["selector"], state="visible", timeout=timeout_ms)
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
            target_locator, note = await self._resolve_primary_heading_locator(page=page, fallback_locator=locator)
            args["_executor_note"] = note
        else:
            target_locator = locator.first
            if match_count > 1:
                args["_executor_note"] = (
                    f'Selector "{selector}" matched {match_count} elements; used first element.'
                )

        return (await target_locator.inner_text()).strip()

    async def extract_html(self, page, args, runtime_state=None):
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
        container_selector = args["container_selector"]
        limit = int(args["limit"])
        fields = args["fields"]

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
        pattern = args["pattern"]
        limit = int(args["limit"])
        fields = args["fields"]
        flags = args.get("flags")
        benchmark_context = runtime_state.get("benchmark_context", {}) if isinstance(runtime_state, dict) else {}
        task_family = str(benchmark_context.get("task_family", "")).strip().lower()

        if task_family == "repeated_structured_items" and self._is_overly_broad_repeated_pattern(pattern):
            table_rows = await self._extract_table_rows(page=page, limit=limit)
            if table_rows:
                projected = self._project_structured_rows_to_fields(rows=table_rows, fields=fields, limit=limit)
                args["_executor_note"] = (
                    "extract_structured_items fallback=table_rows; "
                    f"broad_pattern_rejected={pattern!r}; returned {len(projected)} row(s)"
                )
                return projected

            list_rows = await self._extract_repeated_link_or_list_items(page=page, limit=limit)
            if list_rows:
                projected = self._project_structured_rows_to_fields(rows=list_rows, fields=fields, limit=limit)
                args["_executor_note"] = (
                    "extract_structured_items fallback=list_items; "
                    f"broad_pattern_rejected={pattern!r}; returned {len(projected)} row(s)"
                )
                return projected
            raise StructuredExtractionError(
                code="broad_regex_rejected_for_repeated_structured_items",
                message=(
                    "Regex pattern is too broad for repeated structured extraction and no DOM fallback was found."
                ),
                details={"pattern": pattern, "task_family": task_family},
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
        heading_index = self._find_heading_index(lines, heading_text=heading_text, ignore_case=bool(args.get("ignore_case", True)))
        if heading_index is None:
            raise ValueError(f"Section heading not found: {heading_text!r}")

        stop_at_heading = bool(args.get("stop_at_heading", True))
        min_line_length = int(args.get("min_line_length", 1))
        if min_line_length <= 0:
            min_line_length = 1

        collected: list[str] = []
        for line in lines[heading_index + 1 :]:
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

        args["_executor_note"] = (
            f"extract_section_lines heading={heading_text!r}; start_index={heading_index}; collected={len(collected)}"
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
        return snapshot.model_dump(mode="json")

    async def extract_pattern_from_page_text(self, page, args, runtime_state=None):
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
        if normalize_number or field.endswith("_count") or field in {"count", "total", "article_count"}:
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
        benchmark_anchor_candidates = list(anchor_candidates)
        should_use_fallback_candidates = not benchmark_anchor_candidates and not anchor_text
        default_anchor_candidates = (
            self._default_anchor_candidates(
                value_type=value_type,
                page_language=effective_page_language,
            )
            if should_use_fallback_candidates
            else []
        )
        benchmark_candidates_ranked = list(
            dict.fromkeys(
                [*([anchor_text] if anchor_text else []), *benchmark_anchor_candidates]
            )
        )
        fallback_candidates_ranked = list(dict.fromkeys(default_anchor_candidates))
        resolved_candidates = list(dict.fromkeys([*benchmark_candidates_ranked, *fallback_candidates_ranked]))
        if resolved_candidates:
            args["anchor_candidates"] = resolved_candidates
        if resolved_candidates:
            enforce_anchor_language_filter = bool(args.get("enforce_anchor_language_filter", True))
            resolved_anchor_text = ""
            if benchmark_candidates_ranked:
                # User/benchmark-provided anchors must be evaluated first and should
                # not be filtered out by page-language heuristics.
                try:
                    resolved_anchor_text = await self._resolve_anchor_text(
                        page=page,
                        preferred_anchor="",
                        anchor_candidates=benchmark_candidates_ranked,
                        anchor_matching_mode=anchor_matching_mode,
                        page_language=effective_page_language,
                        enforce_anchor_language_filter=False,
                        value_pattern=str(value_pattern) if value_pattern else None,
                        runtime_state=runtime_state,
                    )
                except ValueError:
                    resolved_anchor_text = ""

            if not resolved_anchor_text and fallback_candidates_ranked:
                resolved_anchor_text = await self._resolve_anchor_text(
                    page=page,
                    preferred_anchor="",
                    anchor_candidates=fallback_candidates_ranked,
                    anchor_matching_mode=anchor_matching_mode,
                    page_language=effective_page_language,
                    enforce_anchor_language_filter=enforce_anchor_language_filter,
                    value_pattern=str(value_pattern) if value_pattern else None,
                    runtime_state=runtime_state,
                )

            if not resolved_anchor_text:
                raise ValueError(f"Anchor text not found for candidates={resolved_candidates}")
            anchor_text = resolved_anchor_text
            args["anchor_text"] = anchor_text
        if not value_pattern:
            raise ValueError("extract_value_near_anchor requires value_pattern or supported value_type")
        value_pattern = str(value_pattern)
        prefer_local_for_contact = value_type in {"email", "phone"}
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
        elif prefer_local_for_contact:
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
            prefer_local_for_contact=prefer_local_for_contact,
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
                prefer_local_for_contact=prefer_local_for_contact,
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
                prefer_local_for_contact=prefer_local_for_contact,
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
                    prefer_local_for_contact=prefer_local_for_contact,
                )

        if best_match is None:
            raise ValueError(
                f"Value not found near anchor_text={anchor_text!r}; pattern={value_pattern!r}; "
                f"required_left_context={required_left_context!r}; required_right_context={required_right_context!r}"
            )
        allow_low_confidence_contact_match = bool(args.get("allow_low_confidence_contact_match", False))
        best_match_value = best_match.get("value")
        best_match_confidence = str(best_match.get("confidence", "")).strip().lower()
        best_match_scope = str(best_match.get("match_scope", "")).strip().lower()
        best_match_distance = int(best_match.get("distance", 10**9))
        allow_close_valid_high_confidence_fallback_contact = (
            prefer_local_for_contact
            and best_match_scope == "fallback"
            and best_match_confidence == "high"
            and best_match_distance <= 80
            and self._is_valid_typed_contact_value(value=best_match_value, value_type=value_type)
        )
        if (
            prefer_local_for_contact
            and not allow_low_confidence_contact_match
            and not strict_context_disabled
            and not allow_close_valid_high_confidence_fallback_contact
            and (
                best_match_confidence == "low"
                or (best_match_scope == "fallback" and best_match_confidence != "high")
            )
        ):
            raise ValueError(
                "Value found near anchor but rejected by confidence policy for contact extraction "
                f"(scope={best_match_scope}, confidence={best_match_confidence}, "
                f"distance={best_match_distance})"
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
        if value_type in {"article_count", "count", "number"}:
            return r"([0-9][0-9\s,\.\u00A0\u202F\+]*)"
        if value_type in {"float", "rating"}:
            return r"([0-9]+(?:[.,][0-9]+)?)"
        if value_type == "email":
            return r"([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,63})"
        if value_type == "phone":
            return r"(\+?\d[\d\-\(\)\s\.]{6,}\d)"
        if value_type == "email_or_phone":
            return r"([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,63}|\+?\d[\d\-\(\)\s\.]{6,}\d)"
        return None

    @staticmethod
    def _default_anchor_candidates(*, value_type: str, page_language: str) -> list[str]:
        if value_type not in {"email", "phone", "email_or_phone"}:
            return []
        if page_language in {"ru", "russian"}:
            return ["Контакты", "Поддержка", "Электронная почта", "Почта", "Телефон", "Помощь"]
        if page_language not in {"en", "english"}:
            return []
        return ["Contact", "Support", "Email", "Help", "Phone"]

    @staticmethod
    def _is_valid_typed_contact_value(*, value: Any, value_type: str) -> bool:
        token = str(value or "").strip()
        if not token:
            return False
        if value_type == "email":
            return bool(re.fullmatch(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,63}", token, flags=re.IGNORECASE))
        if value_type == "phone":
            digits_only = re.sub(r"\D+", "", token)
            if len(digits_only) < 8 or len(digits_only) > 15:
                return False
            return bool(re.fullmatch(r"\+?\d[\d\-\(\)\s\.]{6,}\d", token))
        if value_type == "email_or_phone":
            return ActionHandlers._is_valid_typed_contact_value(value=token, value_type="email") or ActionHandlers._is_valid_typed_contact_value(value=token, value_type="phone")
        return True

    @staticmethod
    def _looks_like_broad_prose_match(*, value: Any, pattern: str) -> bool:
        text = str(value or "").strip()
        if len(text) < 90:
            return False
        if re.search(r"@", text):
            return False
        if re.search(r"\b\d{1,4}([\-./]\d{1,2}){1,2}\b", text):
            return False
        if re.search(r"\$\s?\d|\d+\s?(usd|eur|руб|₽|€|\$)", text, flags=re.IGNORECASE):
            return False
        has_capture_group = False
        try:
            has_capture_group = re.compile(pattern).groups > 0
        except re.error:
            has_capture_group = False
        word_count = len(re.findall(r"\b\w+\b", text))
        sentence_like = bool(re.search(r"[.;:]\s", text))
        return word_count >= 14 and sentence_like and not has_capture_group

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
        role = str(args.get("role", "")).strip()
        name = str(args.get("name", "")).strip()
        href_contains = str(args.get("href_contains", "")).strip()
        scope_selector = str(args.get("scope_selector", "")).strip()
        label = str(args.get("label", "")).strip()
        placeholder = str(args.get("placeholder", "")).strip()
        exact = bool(args.get("exact", False))
        visible_only = bool(args.get("visible_only", True))

        if selector and self._selector_looks_like_plain_text_click_target(selector):
            if not text:
                text = selector
                args["text"] = text
                args.setdefault("exact", True)
            if not href_contains:
                inferred_href = self._infer_href_slug_from_text(text)
                if inferred_href:
                    args["href_contains"] = inferred_href
                    href_contains = inferred_href
            args["_selector_canonicalized_from_plain_text"] = selector
            args.pop("selector", None)
            selector = ""

        scope = page.locator(scope_selector) if scope_selector else page
        candidates: list[dict[str, Any]] = []
        if text:
            candidates.append({"strategy": "role_link_name", "locator": scope.get_by_role("link", name=text, exact=exact), "selector": f"role=link, name={text}"})
        if role and name:
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

        return {
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

    @staticmethod
    def _infer_href_slug_from_text(text: str) -> str:
        token = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")
        return token[:48]

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

    async def _extract_repeated_link_or_list_items(self, *, page, limit: int) -> list[list[str]]:
        rows = await page.evaluate(
            """
            ({ limit }) => {
              const results = [];
              const linkNodes = document.querySelectorAll("main a[href], article a[href], ul li, ol li");
              for (const node of linkNodes) {
                const text = (node.innerText || node.textContent || "").replace(/\\s+/g, " ").trim();
                if (!text || text.length < 2) continue;
                const href = node.getAttribute ? (node.getAttribute("href") || "") : "";
                results.push(href ? [text, href] : [text]);
                if (results.length >= limit) break;
              }
              return results;
            }
            """,
            {"limit": max(limit, 1)},
        )
        return [list(row) for row in (rows or []) if isinstance(row, list) and row]

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
    def _find_heading_index(cls, lines: list[str], *, heading_text: str, ignore_case: bool) -> int | None:
        target = cls._normalize_line(heading_text)
        if not target:
            return None
        if ignore_case:
            target_cmp = target.lower()
            for idx, line in enumerate(lines):
                candidate = cls._normalize_line(line).lower()
                if candidate == target_cmp or target_cmp in candidate:
                    return idx
            return None
        for idx, line in enumerate(lines):
            candidate = cls._normalize_line(line)
            if candidate == target or target in candidate:
                return idx
        return None

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
                "source": "text_fallback",
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
        prefer_local_for_contact: bool = False,
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
                if prefer_local_for_contact and match_scope == "fallback" and confidence != "high":
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
            attr = rule.get("attr")
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
