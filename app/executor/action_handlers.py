import re
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
        await page.goto(args["url"])

    async def click(self, page, args, runtime_state=None):
        selector = str(args.get("selector", "")).strip()
        text = str(args.get("text", "")).strip()
        role = str(args.get("role", "")).strip()
        name = str(args.get("name", "")).strip()
        href_contains = str(args.get("href_contains", "")).strip()
        scope_selector = str(args.get("scope_selector", "")).strip()
        exact = bool(args.get("exact", False))
        visible_only = bool(args.get("visible_only", True))

        if selector:
            if self._is_too_broad_click_selector(selector):
                raise ValueError(
                    f"Click selector is too broad: {selector!r}. Use specific selector or text/role/href filter."
                )
            resolved_selector = selector
            if visible_only:
                resolved_selector = f"{selector}:visible"
            locator = page.locator(resolved_selector)
            await locator.first.click()
            return

        scope = page.locator(scope_selector) if scope_selector else page

        if role and name:
            locator = scope.get_by_role(role, name=name, exact=exact)
            await locator.first.click()
            return

        if href_contains:
            href_selector = f'a[href*="{href_contains}"]'
            if visible_only:
                href_selector = f"{href_selector}:visible"
            locator = scope.locator(href_selector)
            if text:
                locator = locator.filter(has_text=text)
            await locator.first.click()
            return

        if text:
            if not exact and not scope_selector:
                raise ValueError(
                    "Ambiguous or weak click target: text-based click requires exact=true or scope_selector."
                )
            target_selector = "a, button, [role='button'], [role='link']"
            if visible_only:
                target_selector = "a:visible, button:visible, [role='button']:visible, [role='link']:visible"
            locator = scope.locator(target_selector).filter(has_text=text)
            await locator.first.click()
            return

        raise ValueError(
            "click requires selector or text/role+name/href_contains (optionally with scope_selector/exact)"
        )

    async def type(self, page, args, runtime_state=None):
        await page.fill(args["selector"], args["text"])

    async def wait_for(self, page, args, runtime_state=None):
        await page.wait_for_selector(args["selector"])

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

        delegated_args = {
            "pattern": pattern,
            "limit": limit,
            "fields": fields,
            "occurrence": 1,
        }
        if flags is not None:
            delegated_args["flags"] = flags

        items = await self.extract_pattern_from_page_text(page, delegated_args, runtime_state)
        args["_executor_note"] = (
            f"extract_structured_items matched pattern={pattern!r}; returned {len(items)} repeated item(s)"
        )
        return items

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

        source_text = await self._load_source_text(page=page, runtime_state=runtime_state)

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

        source_text = await self._load_source_text(page=page, runtime_state=runtime_state)
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
        if not value_pattern:
            value_pattern = self._resolve_value_pattern(value_type)
        if anchor_matching_mode not in {"auto", "exact", "contains"}:
            anchor_matching_mode = "auto"
        if anchor_candidates:
            anchor_text = await self._resolve_anchor_text(
                page=page,
                preferred_anchor=anchor_text,
                anchor_candidates=anchor_candidates,
                anchor_matching_mode=anchor_matching_mode,
                page_language=page_language,
                value_pattern=str(value_pattern) if value_pattern else None,
                runtime_state=runtime_state,
            )
            args["anchor_text"] = anchor_text
        if not value_pattern:
            raise ValueError("extract_value_near_anchor requires value_pattern or supported value_type")
        value_pattern = str(value_pattern)
        search_direction = str(args.get("search_direction", "after")).lower()
        same_block_only = bool(args.get("same_block_only", True))
        required_right_context = args.get("required_right_context")
        required_left_context = args.get("required_left_context")
        max_distance_chars = args.get("max_distance_chars")
        group_index = args.get("group_index", 1)
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
                )

        if best_match is None:
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
        value_pattern: str | None,
        runtime_state=None,
    ) -> str:
        source_text = await self._load_source_text(page=page, runtime_state=runtime_state)
        visible_anchors = await self._collect_visible_anchor_texts(page)
        ranked = [preferred_anchor] + anchor_candidates if preferred_anchor else list(anchor_candidates)
        score_best: tuple[int, str] | None = None

        for candidate in ranked:
            if not self._anchor_present(
                source_text=source_text,
                visible_anchors=visible_anchors,
                candidate=candidate,
                matching_mode=anchor_matching_mode,
                page_language=page_language,
            ):
                continue
            if not value_pattern:
                return candidate
            candidate_score = await self._score_anchor_candidate_block_match(
                page=page,
                anchor_text=candidate,
                value_pattern=value_pattern,
            )
            if score_best is None or candidate_score > score_best[0]:
                score_best = (candidate_score, candidate)

        if score_best is not None:
            return score_best[1]
        raise ValueError(f"Anchor text not found for candidates={ranked}")

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
        return [str(item).strip() for item in anchors if str(item).strip()]

    @classmethod
    def _anchor_present(
        cls,
        *,
        source_text: str,
        visible_anchors: list[str],
        candidate: str,
        matching_mode: str,
        page_language: str,
    ) -> bool:
        normalized_candidate = candidate.strip()
        if not normalized_candidate:
            return False
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
        return None

    async def _load_source_text(self, page, runtime_state=None) -> str:
        source_text = ""
        if runtime_state is not None:
            source_text = runtime_state.get("last_page_text") or ""
        if not source_text:
            source_text = (await page.locator("body").inner_text()).strip()
            if runtime_state is not None:
                runtime_state["last_page_text"] = source_text
        return source_text

    @staticmethod
    def _is_too_broad_click_selector(selector: str) -> bool:
        return selector.strip().lower() in {"a", "button", "*", "[role='button']", '[role="button"]'}

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
            """
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
                score = (
                    distance,
                    int(candidate.get("source_rank", 3)),
                    0 if str(candidate.get("source", "")).startswith("dom_") else 1,
                )
                if best is None or score < best["score"]:
                    best = {
                        "value": extracted_value,
                        "distance": distance,
                        "source": candidate.get("source", "unknown"),
                        "score": score,
                    }
        return best

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
