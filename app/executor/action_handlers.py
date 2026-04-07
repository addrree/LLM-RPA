import re
from typing import Any

from app.observer.page_observer import PageObserver


class ActionHandlers:
    def __init__(self):
        self.page_observer = PageObserver()

    async def open_url(self, page, args, runtime_state=None):
        await page.goto(args["url"])

    async def click(self, page, args, runtime_state=None):
        await page.click(args["selector"])

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
        anchor_text = str(args["anchor_text"])
        value_pattern = str(args["value_pattern"])
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

        if best_match is None and same_block_only:
            fallback_used = True
            relaxed_candidates = await self._collect_anchor_candidates(
                page=page,
                anchor_text=anchor_text,
                search_direction=search_direction,
                same_block_only=False,
                max_distance_chars=max_distance_chars,
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
            f"source={best_match['source']}; fallback_used={fallback_used}"
        )
        return result

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
    def _extract_match_value(match: re.Match[str], group_index: int | None):
        if group_index is not None:
            return match.group(int(group_index))
        return match.group(1) if match.groups() else match.group(0)

    async def _collect_anchor_candidates(
        self,
        *,
        page,
        anchor_text: str,
        search_direction: str,
        same_block_only: bool,
        max_distance_chars: int | None,
        runtime_state=None,
    ) -> list[dict[str, Any]]:
        window_chars = max_distance_chars if isinstance(max_distance_chars, int) and max_distance_chars > 0 else 400
        candidates = await page.evaluate(
            """
            ({ anchorText, direction, sameBlockOnly, windowChars }) => {
              const normalizeText = (text) => (text || "").replace(/\\s+/g, " ").trim();
              const reasonableSelector = "li, tr, td, th, p, dt, dd, article, section, div, span, a";
              const collect = [];
              const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
              let node;
              while ((node = walker.nextNode())) {
                const value = node.nodeValue || "";
                const idx = value.toLowerCase().indexOf((anchorText || "").toLowerCase());
                if (idx < 0) continue;
                const host = node.parentElement;
                if (!host) continue;
                let container = host;
                if (sameBlockOnly) {
                  container = host.closest(reasonableSelector) || host.parentElement || host;
                } else {
                  container = document.body;
                }
                const text = normalizeText(container.innerText || container.textContent || "");
                if (!text) continue;
                const anchorIdx = text.toLowerCase().indexOf((anchorText || "").toLowerCase());
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
                  source: sameBlockOnly ? "dom_same_block" : "dom_page",
                  full_text: text,
                  anchor_idx: anchorIdx,
                  window_start: start,
                  window_end: end,
                  window_text: text.slice(start, end),
                  anchor_idx_in_window: anchorIdx - start
                });
              }
              return collect;
            }
            """,
            {
                "anchorText": anchor_text,
                "direction": search_direction,
                "sameBlockOnly": same_block_only,
                "windowChars": window_chars,
            },
        )

        if candidates:
            return candidates

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
            anchor_idx = int(candidate.get("anchor_idx_in_window", 0))
            for match in re.finditer(value_pattern, window_text, flags=flags_value):
                if not self._contexts_match(
                    window_text=window_text,
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
                score = (distance, 0 if candidate.get("source") == "dom_same_block" else 1)
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
