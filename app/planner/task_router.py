from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field


TaskType = Literal[
    "single_entity_metadata",
    "search_results_extraction",
    "structured_table_extraction",
    "repeated_items_extraction",
    "catalog_or_card_extraction",
    "direct_value_extraction",
    "semantic_navigation",
    "row_or_item_action",
    "visual_or_spatial_task",
    "generic_web_task",
]


SUPPORTED_RUNTIME_INTENTS = {
    "current_url",
    "page_title",
    "value_near_anchor",
    "package_metadata",
    "search_results",
    "product_cards",
    "table_rows",
    "article_results",
    "repository_results",
    "paper_results",
    "news_items",
}

COMMON_RUNTIME_INTENTS = ["current_url", "page_title"]


class PlanningProfile(BaseModel):
    name: str
    allowed_actions: list[str] = Field(default_factory=list)
    preferred_intents: list[str] = Field(default_factory=list)
    conceptual_intents: list[str] = Field(default_factory=list)
    forbidden_actions: list[str] = Field(default_factory=list)
    expected_fields: list[str] = Field(default_factory=list)
    expected_output_type: str = "object"
    required_skill_groups: list[str] = Field(default_factory=list)
    prompt_profile: str = "restricted_profile"
    profile_prompt_length: int = 0
    full_vocabulary_was_used: bool = False

    @property
    def preferred_runtime_intents(self) -> list[str]:
        return [intent for intent in self.preferred_intents if intent in SUPPORTED_RUNTIME_INTENTS]


class TaskRoute(BaseModel):
    task_type: TaskType
    confidence: float
    reason: str
    signals: list[str] = Field(default_factory=list)
    required_skill_groups: list[str] = Field(default_factory=list)
    item_type: str | None = None
    expected_output_type: str = "object"
    expected_fields: list[str] = Field(default_factory=list)
    requires_navigation: bool = False
    requires_form_fill: bool = False
    requires_table_extraction: bool = False
    requires_visual: bool = False
    needs_observe_first: bool = True
    alternative_task_types: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    profile: PlanningProfile

    def diagnostics(self) -> dict[str, Any]:
        profile = self.profile.model_dump(mode="json")
        return {
            "task_type": self.task_type,
            "item_type": self.item_type or "",
            "router_confidence": self.confidence,
            "router_signals": list(self.signals),
            "alternative_task_types": list(self.alternative_task_types),
            "required_skill_groups": list(self.required_skill_groups),
            "router_warnings": list(self.warnings),
            "planning_profile": profile,
            "allowed_actions": list(self.profile.allowed_actions),
            "preferred_intents": list(self.profile.preferred_runtime_intents),
            "preferred_runtime_intents": list(self.profile.preferred_runtime_intents),
            "conceptual_profile_intents": list(self.profile.conceptual_intents),
            "forbidden_actions": list(self.profile.forbidden_actions),
            "prompt_profile": self.profile.prompt_profile,
            "profile_prompt_length": self.profile.profile_prompt_length,
            "full_vocabulary_was_used": self.profile.full_vocabulary_was_used,
        }


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _runtime_intents(*intents: str) -> list[str]:
    return _unique([*COMMON_RUNTIME_INTENTS, *(intent for intent in intents if intent in SUPPORTED_RUNTIME_INTENTS)])


def _profile(
    *,
    name: str,
    allowed_actions: list[str],
    preferred_intents: list[str],
    conceptual_intents: list[str],
    expected_output_type: str,
    required_skill_groups: list[str],
    expected_fields: list[str] | None = None,
    forbidden_actions: list[str] | None = None,
) -> PlanningProfile:
    return PlanningProfile(
        name=name,
        allowed_actions=_unique(allowed_actions),
        preferred_intents=_unique(intent for intent in preferred_intents if intent in SUPPORTED_RUNTIME_INTENTS),
        conceptual_intents=_unique(conceptual_intents),
        forbidden_actions=_unique(forbidden_actions or []),
        expected_fields=_unique(expected_fields or []),
        expected_output_type=expected_output_type,
        required_skill_groups=_unique(required_skill_groups),
        prompt_profile="restricted_profile",
        full_vocabulary_was_used=False,
    )


BASE_ACTIONS = ["open_url", "observe_page", "wait_for", "finish"]
FORM_ACTIONS = ["fill_by_semantic_target", "click_by_semantic_target", "press"]
SEMANTIC_NAV_ACTIONS = ["click_by_semantic_target", "navigate_to_relevant_section", "wait_for"]
GENERIC_EXTRACTION_ACTIONS = [
    "extract_by_intent",
    "extract_visible_links",
    "extract_value_near_anchor",
    "extract_structured_items",
    "extract_items",
]


PROFILES: dict[str, PlanningProfile] = {
    "single_entity_metadata": _profile(
        name="single_entity_metadata",
        allowed_actions=[
            *BASE_ACTIONS,
            *FORM_ACTIONS,
            "extract_by_intent",
            "extract_value_near_anchor",
            "extract_visible_links",
            "extract_structured_items",
            "extract_items",
        ],
        preferred_intents=_runtime_intents("package_metadata", "value_near_anchor"),
        conceptual_intents=["entity_metadata"],
        expected_output_type="object",
        required_skill_groups=["semantic_form_fill", "entity_metadata_extraction"],
        forbidden_actions=["extract_pattern_from_page_text"],
    ),
    "search_results_extraction": _profile(
        name="search_results_extraction",
        allowed_actions=[
            *BASE_ACTIONS,
            *FORM_ACTIONS,
            "extract_by_intent",
            "extract_visible_links",
            "extract_structured_items",
            "extract_items",
        ],
        preferred_intents=_runtime_intents(
            "search_results",
            "article_results",
            "repository_results",
            "paper_results",
            "news_items",
        ),
        conceptual_intents=["result_list"],
        expected_output_type="list",
        required_skill_groups=["semantic_form_fill", "result_list_extraction"],
        forbidden_actions=["extract_pattern_from_page_text"],
    ),
    "structured_table_extraction": _profile(
        name="structured_table_extraction",
        allowed_actions=[
            *BASE_ACTIONS,
            "extract_by_intent",
            "find_row_by_condition",
            "extract_structured_items",
            "extract_items",
            "extract_visible_links",
        ],
        preferred_intents=_runtime_intents("table_rows"),
        conceptual_intents=["table_rows"],
        expected_output_type="table",
        required_skill_groups=["table_extraction"],
        forbidden_actions=["extract_pattern_from_page_text"],
    ),
    "repeated_items_extraction": _profile(
        name="repeated_items_extraction",
        allowed_actions=[
            *BASE_ACTIONS,
            "extract_by_intent",
            "extract_visible_links",
            "extract_structured_items",
            "extract_items",
        ],
        preferred_intents=_runtime_intents(
            "search_results",
            "article_results",
            "repository_results",
            "paper_results",
            "news_items",
        ),
        conceptual_intents=["repeated_items"],
        expected_output_type="list",
        required_skill_groups=["repeated_item_extraction"],
        forbidden_actions=["extract_pattern_from_page_text"],
    ),
    "catalog_or_card_extraction": _profile(
        name="catalog_or_card_extraction",
        allowed_actions=[
            *BASE_ACTIONS,
            *FORM_ACTIONS,
            "extract_by_intent",
            "extract_structured_items",
            "extract_items",
            "extract_visible_links",
        ],
        preferred_intents=_runtime_intents("product_cards", "search_results"),
        conceptual_intents=["card_items"],
        expected_output_type="list",
        required_skill_groups=["card_or_catalog_extraction"],
        forbidden_actions=["extract_pattern_from_page_text"],
    ),
    "direct_value_extraction": _profile(
        name="direct_value_extraction",
        allowed_actions=[
            *BASE_ACTIONS,
            "extract_by_intent",
            "extract_value_near_anchor",
            "extract_text",
            "extract_html",
        ],
        preferred_intents=_runtime_intents("value_near_anchor"),
        conceptual_intents=["direct_value"],
        expected_output_type="value",
        required_skill_groups=["anchor_value_extraction"],
        forbidden_actions=["extract_pattern_from_page_text"],
    ),
    "semantic_navigation": _profile(
        name="semantic_navigation",
        allowed_actions=[
            *BASE_ACTIONS,
            *SEMANTIC_NAV_ACTIONS,
            "click",
            "extract_by_intent",
            "extract_visible_links",
        ],
        preferred_intents=_runtime_intents(),
        conceptual_intents=["semantic_navigation"],
        expected_output_type="navigation",
        required_skill_groups=["semantic_navigation"],
        forbidden_actions=["extract_pattern_from_page_text"],
    ),
    "row_or_item_action": _profile(
        name="row_or_item_action",
        allowed_actions=[
            *BASE_ACTIONS,
            "extract_by_intent",
            "find_row_by_condition",
            "click_row_action",
            "click_by_semantic_target",
            "wait_for",
        ],
        preferred_intents=_runtime_intents("table_rows", "search_results"),
        conceptual_intents=["row_action"],
        expected_output_type="action",
        required_skill_groups=["row_selection", "semantic_action"],
        forbidden_actions=["extract_pattern_from_page_text"],
    ),
    "visual_or_spatial_task": _profile(
        name="visual_or_spatial_task",
        allowed_actions=[
            *BASE_ACTIONS,
            "visual_observe",
            "visual_extract_object_count",
            "visual_click_by_geometry",
            "screenshot",
        ],
        preferred_intents=_runtime_intents(),
        conceptual_intents=["visual_spatial"],
        expected_output_type="visual",
        required_skill_groups=["visual_grounding"],
    ),
    "generic_web_task": _profile(
        name="generic_web_task",
        allowed_actions=[
            *BASE_ACTIONS,
            *FORM_ACTIONS,
            "click",
            "fill",
            "extract_text",
            "extract_html",
            "extract_by_intent",
            "extract_visible_links",
            "extract_value_near_anchor",
            "extract_structured_items",
            "extract_items",
            "find_row_by_condition",
            "click_row_action",
            "visual_observe",
            "visual_extract_object_count",
            "visual_click_by_geometry",
            "extract_section_lines",
            "compare_structured_values",
            "extract_pattern_from_page_text",
        ],
        preferred_intents=_runtime_intents(
            "value_near_anchor",
            "package_metadata",
            "search_results",
            "product_cards",
            "table_rows",
            "article_results",
            "repository_results",
            "paper_results",
            "news_items",
        ),
        conceptual_intents=["generic_web_task"],
        expected_output_type="unknown",
        required_skill_groups=["generic_web"],
    ),
}


class TaskRouter:
    confidence_threshold = 0.65

    def route(self, user_goal: str, benchmark_context: dict | None = None) -> TaskRoute:
        text = self._normalize_text(user_goal)
        signals: list[str] = []
        scores = {name: 0.0 for name in PROFILES}

        benchmark_family = str((benchmark_context or {}).get("task_family", "") or "").strip()
        if benchmark_family:
            family_route = {
                "single_value_extraction": "direct_value_extraction",
                "anchored_value_extraction": "direct_value_extraction",
                "repeated_structured_items": "repeated_items_extraction",
                "navigation_then_extraction": "semantic_navigation",
                "multi_step_information_retrieval": "generic_web_task",
            }.get(benchmark_family)
            if family_route:
                scores[family_route] += 0.35
                signals.append(f"benchmark_family:{benchmark_family}")

        has_search = self._has_any(text, ["search", "find", "query", "look up", "lookup", "найди", "поиск", "искать"])
        has_extract = self._has_any(text, ["extract", "export", "return", "collect", "get", "выгрузи", "извлеки", "верни", "получи", "собери"])
        has_list = self._has_any(text, ["list", "results", "top", "all", "links", "items", "rows", "список", "результат", "ссылк", "все", "топ"])
        has_table = self._has_any(text, ["table", "row", "column", "cell", "таблиц", "строк", "колон", "ячей"])
        has_card = self._has_any(text, ["card", "cards", "catalog", "product", "products", "price", "карточ", "каталог", "товар", "цена"])
        has_visual = self._has_any(text, ["visual", "screenshot", "image", "canvas", "coordinate", "x=", "y=", "визуал", "скрин", "изображ", "координат"])
        has_row_action = (
            has_table
            and not has_extract
            and self._has_any(text, ["click", "select", "delete", "star", "choose", "нажм", "выбери", "удали"])
        )
        has_navigation = self._has_any(text, ["click", "open link", "navigate", "go to", "follow", "press", "нажм", "перейд", "клик", "открой ссыл"])
        has_form = has_search or self._has_any(text, ["fill", "enter", "type", "input", "submit", "заполни", "введи", "форма"])
        has_anchor_value = self._has_any(text, ["near", "next to", "beside", "anchor", "label", "value", "рядом", "возле", "значени", "метк"])
        has_metadata_fields = self._metadata_field_count(text) >= 2
        has_single_object = self._has_any(text, ["package", "project", "library", "repo", "repository", "article", "profile", "пакет", "проект", "библиотек", "репозитор", "профил"])
        has_article = self._has_any(text, ["article", "articles", "news", "post", "стать", "новост", "публикац"])
        has_repository = self._has_any(text, ["repository", "repositories", "repo", "репозитор"])
        has_paper = self._has_any(text, ["paper", "papers", "preprint", "publication", "научн", "препринт"])

        # Keep a normalized Cyrillic pass because some legacy token literals above are mojibake.
        has_search = has_search or self._has_any(text, ["найди", "поиск", "искать"])
        has_extract = has_extract or self._has_any(text, ["выгрузи", "извлеки", "верни", "получи", "собери"])
        has_list = has_list or self._has_any(text, ["список", "результат", "ссылк", "все", "топ"])
        has_table = has_table or self._has_any(text, ["таблиц", "строк", "колон", "ячей"])
        has_card = has_card or self._has_any(text, ["карточ", "каталог", "товар", "цен", "витрин"])
        has_visual = has_visual or self._has_any(text, ["визуал", "скрин", "изображ", "координат"])
        has_row_action = has_row_action or (
            has_table
            and not has_extract
            and self._has_any(text, ["нажм", "выбери", "удали"])
        )
        has_navigation = has_navigation or self._has_any(text, ["нажм", "перейд", "клик", "открой ссыл"])
        has_form = has_form or has_search or self._has_any(text, ["заполни", "введи", "форма"])
        has_anchor_value = has_anchor_value or self._has_any(text, ["рядом", "возле", "значени", "метк"])
        has_single_object = has_single_object or self._has_any(
            text,
            ["пакет", "проект", "библиотек", "репозитор", "профил"],
        )
        has_article = has_article or self._has_any(text, ["стать", "новост", "публикац"])
        has_repository = has_repository or self._has_any(text, ["репозитор"])
        has_paper = has_paper or self._has_any(text, ["научн", "препринт"])

        if has_visual:
            scores["visual_or_spatial_task"] += 0.85
            signals.append("visual_or_spatial")
        if has_row_action:
            scores["row_or_item_action"] += 0.82
            signals.append("row_or_item_action")
        if has_table:
            scores["structured_table_extraction"] += 0.78
            signals.append("table_shape")
        if has_card:
            scores["catalog_or_card_extraction"] += 0.74
            signals.append("card_or_catalog_shape")
        if has_search and (has_list or self._has_any(text, ["result", "results", "результат"])):
            scores["search_results_extraction"] += 0.78
            signals.append("search_results_shape")
        if has_list and has_extract and not has_table and not has_card:
            scores["repeated_items_extraction"] += 0.72
            signals.append("repeated_items_shape")
        if has_metadata_fields and (has_single_object or has_search):
            scores["single_entity_metadata"] += 0.8
            signals.append("single_entity_metadata_fields")
        if has_anchor_value or (has_extract and not has_list and not has_table and not has_card and not has_metadata_fields):
            scores["direct_value_extraction"] += 0.68
            signals.append("direct_or_anchor_value")
        if has_navigation:
            scores["semantic_navigation"] += 0.76
            signals.append("semantic_navigation")
            if has_extract:
                signals.append("navigation_then_extraction")
                scores["direct_value_extraction"] -= 0.08

        if has_form:
            signals.append("requires_form_fill")
            for task_type in ("single_entity_metadata", "search_results_extraction", "catalog_or_card_extraction"):
                scores[task_type] += 0.08
        if has_article:
            signals.append("article_like_items")
            scores["search_results_extraction"] += 0.04
            scores["repeated_items_extraction"] += 0.04
        if has_repository:
            signals.append("repository_like_items")
            scores["search_results_extraction"] += 0.04
        if has_paper:
            signals.append("paper_like_items")
            scores["search_results_extraction"] += 0.04

        if has_card and has_extract:
            scores["catalog_or_card_extraction"] += 0.16
            if scores["search_results_extraction"] > 0:
                scores["search_results_extraction"] -= 0.1
            signals.append("card_catalog_priority")

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_type, confidence = ranked[0]
        warnings: list[str] = []
        if confidence < self.confidence_threshold:
            best_type = "generic_web_task"
            confidence = max(confidence, 0.5)
            warnings.append("router_confidence_below_threshold_using_restricted_generic_profile")
            signals.append("low_confidence_generic")

        profile = PROFILES[best_type].model_copy(deep=True)
        expected_fields = self._infer_expected_fields(text)
        profile.expected_fields = expected_fields
        route = TaskRoute(
            task_type=best_type,  # type: ignore[arg-type]
            confidence=round(confidence, 3),
            reason=self._reason_for(best_type, signals),
            signals=_unique(signals),
            required_skill_groups=list(profile.required_skill_groups),
            item_type=self._infer_item_type(text, best_type),
            expected_output_type=profile.expected_output_type,
            expected_fields=expected_fields,
            requires_navigation=has_navigation,
            requires_form_fill=has_form,
            requires_table_extraction=best_type in {"structured_table_extraction", "row_or_item_action"},
            requires_visual=best_type == "visual_or_spatial_task",
            needs_observe_first=True,
            alternative_task_types=[
                task_type
                for task_type, score in ranked[1:4]
                if score >= self.confidence_threshold and task_type != best_type
            ],
            warnings=warnings,
            profile=profile,
        )
        return route

    @staticmethod
    def _normalize_text(value: str) -> str:
        return str(value or "").casefold()

    @staticmethod
    def _has_any(text: str, tokens: list[str]) -> bool:
        return any(token.casefold() in text for token in tokens)

    @classmethod
    def _metadata_field_count(cls, text: str) -> int:
        groups = [
            ["name", "title", "назван", "имя"],
            ["version", "latest", "release", "верс", "релиз"],
            ["description", "summary", "описан", "кратк"],
            ["author", "owner", "maintainer", "автор"],
            ["date", "published", "updated", "дат", "обнов"],
        ]
        return sum(1 for group in groups if cls._has_any(text, group))

    @classmethod
    def _infer_expected_fields(cls, text: str) -> list[str]:
        fields: list[str] = []
        checks = [
            ("name", ["name", "title", "назван", "имя"]),
            ("latest_version", ["latest version", "version", "верс"]),
            ("description", ["description", "summary", "описан", "кратк"]),
            ("url", ["url", "link", "ссылк"]),
            ("price", ["price", "цена"]),
            ("rating", ["rating", "рейтинг"]),
            ("email", ["email", "mail", "почт"]),
            ("phone", ["phone", "телефон"]),
        ]
        for field, tokens in checks:
            if cls._has_any(text, tokens):
                fields.append(field)
        return _unique(fields)

    @classmethod
    def _infer_item_type(cls, text: str, task_type: str) -> str | None:
        checks = [
            ("product", ["product", "products", "товар", "карточ", "catalog", "каталог"]),
            ("article", ["article", "articles", "news", "post", "стать", "новост", "публикац"]),
            ("repository", ["repository", "repositories", "repo", "репозитор"]),
            ("paper", ["paper", "papers", "preprint", "publication", "научн", "препринт"]),
            ("package", ["package", "library", "module", "пакет", "библиотек"]),
            ("table_row", ["table", "row", "таблиц", "строк"]),
            ("link", ["link", "links", "ссылк"]),
        ]
        for item_type, tokens in checks:
            if cls._has_any(text, tokens):
                return item_type
        if task_type == "visual_or_spatial_task":
            return "visual"
        if task_type == "direct_value_extraction":
            return "value"
        return None

    @staticmethod
    def _reason_for(task_type: str, signals: list[str]) -> str:
        if task_type == "generic_web_task" and "low_confidence_generic" in signals:
            return "No structural task type reached the confidence threshold; using restricted generic profile."
        return f"Selected {task_type} from structural goal signals: {', '.join(_unique(signals)) or 'none'}."
