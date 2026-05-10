"""Task-type router that dispatches to the right execution backend.

The router no longer treats ``task.difficulty`` as the primary dispatch
signal. Difficulty is retained as a budget hint. The main route decision
is based on the compiled task type: table computation, document QA,
mixed context, image understanding, or pure-reasoning fallback.

Each route has its own endpoint and execution kind:

    - "operator_executor" -> tool-first code/RAG executor
    - "tablellm_direct" -> :class:`TableLLMDirectAgent`
    - "react"           -> :class:`ReActAgent`
    - "multi_agent"     -> :class:`MultiAgentOrchestrator`

The router records the decision and the per-call endpoint into the
``router_decision`` block of the result dict so it shows up in
``trace.json`` for post-mortems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.orchestrator import (
    MultiAgentOrchestrator,
    OrchestratorConfig,
)
from data_agent_baseline.agents.operator_executor import OperatorExecutor
from data_agent_baseline.agents.planner import PlannerConfig
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.agents.reasoner_repair import run_reasoner_repair
from data_agent_baseline.agents.specialist import SpecialistConfig
from data_agent_baseline.agents.synthesizer import SynthesizerConfig
from data_agent_baseline.agents.tablellm_direct import CodegenDirectAgent
from data_agent_baseline.agents.task_compiler import (
    CompiledTask,
    SourceCapability,
    compile_task,
)
from data_agent_baseline.benchmark.schema import PublicTask
from concurrent.futures import ThreadPoolExecutor

from data_agent_baseline.budget import (
    BudgetController,
    BudgetExceeded,
    get_budget_controller,
    set_budget_controller,
)
from data_agent_baseline.config import (
    AgentConfig,
    CrossModelVerifierSpec,
    CrossModelVerifyConfig,
    RouteConfig,
    RouterConfig,
    SelfConsistencyConfig,
)
from data_agent_baseline.eval.answer_validator import validate_answer_table
from data_agent_baseline.eval.column_match import column_signature
from data_agent_baseline.progress import get_progress_logger
from data_agent_baseline.tools.registry import create_default_tool_registry


# ---------------------------------------------------------------------------
# Answer-validation stamping (used by every route before cascade decision)
# ---------------------------------------------------------------------------


def _stamp_validation(payload: dict[str, Any]) -> None:
    """Run the answer validator and downgrade ``succeeded`` on invalid output.

    Mutates ``payload`` in place. After this call, a payload with an
    obviously-broken table (no rows, ragged rows, fully empty column,
    etc.) will have ``succeeded=False`` so the router cascade fires.
    """
    answer = payload.get("answer")
    result = validate_answer_table(answer)
    payload["answer_validation"] = result.to_dict()
    if not result.valid and payload.get("succeeded"):
        payload["succeeded"] = False
        existing = payload.get("failure_reason")
        msg = "; ".join(item.message for item in result.errors) or "answer_validation_failed"
        payload["failure_reason"] = (
            f"{existing} | invalid_answer: {msg}" if existing else f"invalid_answer: {msg}"
        )


# ---------------------------------------------------------------------------
# Difficulty heuristic (used when task.difficulty is missing/unknown)
# ---------------------------------------------------------------------------


_AGGREGATION_KEYWORDS = {
    "average", "avg", "mean", "median", "sum", "total", "count", "percentage",
    "percent", "ratio", "compare", "comparison", "difference", "trend",
    "max", "maximum", "min", "minimum", "highest", "lowest", "rank",
    "year-over-year", "growth", "between", "range",
}

_MULTI_HOP_KEYWORDS = {
    "join", "across", "both", "also", "then", "first", "second", "third",
    "after", "before", "while", "during", "whose", "with the same",
    "from each", "for each", "by each", "per", "every",
}

_DIFFICULTY_BUCKETS = ["Easy", "Medium", "Hard", "Extreme"]


def _classify_file_kind(path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return "csv"
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        return "db"
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix in {".md", ".txt"}:
        return "doc"
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "image"
    return "other"


def _estimate_difficulty(task: PublicTask) -> tuple[str, dict[str, Any]]:
    """Score a task's complexity from its files + question, return a difficulty bucket.

    The scoring is intentionally rough: it isn't trying to perfectly match
    the official label, just to land in the right ballpark when the
    label is unavailable. The features are explainable so we can audit
    misroutes by reading the trace.
    """
    files = [path for path in task.context_dir.rglob("*") if path.is_file()]
    kinds = [_classify_file_kind(path) for path in files]
    kind_counts: dict[str, int] = {}
    for kind in kinds:
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    distinct_kinds = sum(1 for kind in ("csv", "db", "json", "doc") if kind_counts.get(kind))
    total_bytes = sum(path.stat().st_size for path in files if path.is_file())

    question_lower = task.question.lower()
    question_word_count = len(question_lower.split())
    aggregation_hits = sum(1 for kw in _AGGREGATION_KEYWORDS if kw in question_lower)
    multihop_hits = sum(1 for kw in _MULTI_HOP_KEYWORDS if kw in question_lower)

    score = 0
    if len(files) >= 4:
        score += 1
    if distinct_kinds >= 2:
        score += 1
    if distinct_kinds >= 3:
        score += 1
    if kind_counts.get("doc", 0) >= 1:
        score += 1
    if kind_counts.get("db", 0) >= 1:
        score += 1
    if total_bytes >= 200_000:
        score += 1
    if total_bytes >= 1_500_000:
        score += 1
    if question_word_count >= 22:
        score += 1
    if aggregation_hits >= 2:
        score += 1
    if multihop_hits >= 1:
        score += 1
    if multihop_hits >= 2:
        score += 1

    if score <= 1:
        bucket = "Easy"
    elif score <= 3:
        bucket = "Medium"
    elif score <= 6:
        bucket = "Hard"
    else:
        bucket = "Extreme"

    features = {
        "score": score,
        "file_count": len(files),
        "distinct_kinds": distinct_kinds,
        "kind_counts": kind_counts,
        "total_bytes": total_bytes,
        "question_word_count": question_word_count,
        "aggregation_hits": aggregation_hits,
        "multihop_hits": multihop_hits,
        "bucket": bucket,
    }
    return bucket, features


# ---------------------------------------------------------------------------
# Public dispatch entry point
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RouterDecision:
    difficulty: str
    task_type: str
    budget_level: str
    needs_reasoner: bool
    route_name: str
    kind: str
    api_base: str
    model: str
    fallback_used: bool = False
    difficulty_source: str = "task_label"   # task_label | heuristic | default
    route_reason: str = "task_type"
    compiled_task: dict[str, Any] = field(default_factory=dict)
    estimate_features: dict[str, Any] = field(default_factory=dict)
    cascade_attempts: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "difficulty": self.difficulty,
            "task_type": self.task_type,
            "budget_level": self.budget_level,
            "needs_reasoner": self.needs_reasoner,
            "route_name": self.route_name,
            "kind": self.kind,
            "api_base": self.api_base,
            "model": self.model,
            "fallback_used": self.fallback_used,
            "difficulty_source": self.difficulty_source,
            "route_reason": self.route_reason,
            "compiled_task": dict(self.compiled_task),
            "estimate_features": dict(self.estimate_features),
            "cascade_attempts": list(self.cascade_attempts),
            "notes": list(self.notes),
        }


def _first_route_with_kind(
    router: RouterConfig,
    kinds: tuple[str, ...],
    *,
    require_rag: bool = False,
    visited: set[str] | None = None,
) -> str | None:
    visited = visited or set()
    for kind in kinds:
        for name, route in router.routes.items():
            if name in visited:
                continue
            if route.kind.lower() != kind:
                continue
            if require_rag and not route.rag.enabled:
                continue
            return name
    if require_rag:
        return _first_route_with_kind(router, kinds, require_rag=False, visited=visited)
    return None


def _first_route_named(
    router: RouterConfig,
    names: tuple[str, ...],
    *,
    visited: set[str] | None = None,
) -> str | None:
    visited = visited or set()
    for name in names:
        if name in router.routes and name not in visited:
            return name
    return None


def _is_light_table_task(compiled: CompiledTask, *, difficulty_key: str) -> bool:
    """True for small table tasks that should not start on a heavy mixed route."""
    flags = set(compiled.ambiguity_flags)
    if difficulty_key != "easy":
        return False
    if compiled.task_type not in {"table_computation", "table_with_semantic_rule"}:
        return False
    if "semantic_rule_context" in flags:
        return False
    if flags & {
        "record_text_context",
        "record_extraction_required",
        "large_document_context",
        "large_table_context",
        "needs_vision",
        "unsupported_file_type",
    }:
        return False
    return compiled.file_count <= 4


def _route_for_compiled_task(
    router: RouterConfig,
    compiled: CompiledTask,
    *,
    difficulty_key: str = "",
    visited: set[str] | None = None,
) -> tuple[str, str, bool]:
    profile_route = _route_for_execution_profile(router, compiled, visited=visited)
    if profile_route is not None:
        return profile_route

    task_type = compiled.task_type.lower()
    if _is_light_table_task(compiled, difficulty_key=difficulty_key):
        candidate = _first_route_named(router, ("easy",), visited=visited)
        if candidate is not None:
            return candidate, "light_table_easy", False

    mapped = router.task_type_routing.get(task_type)
    if mapped and mapped in router.routes and mapped not in (visited or set()):
        return mapped, "task_type_routing", False

    if task_type == "table_computation":
        candidate = _first_route_with_kind(
            router, ("operator_executor", "tablellm_direct", "react", "multi_agent"), visited=visited
        )
    elif task_type in {"document_qa", "mixed_context"}:
        candidate = _first_route_with_kind(
            router, ("operator_executor", "tablellm_direct", "react", "multi_agent"),
            require_rag=True,
            visited=visited,
        )
    elif task_type == "image_understanding":
        # We don't ship a dedicated vision executor right now; ReAct/multi_agent
        # gets the question + the image-file paths as plain context and lets
        # the LLM call into multimodal endpoints if the route's model
        # supports them. Otherwise the run will surface as a normal failure
        # for cascade fallback to handle.
        candidate = _first_route_with_kind(
            router, ("multi_agent", "react"), visited=visited,
        )
    else:
        candidate = _first_route_with_kind(router, ("react", "operator_executor", "multi_agent"), visited=visited)

    if candidate is not None:
        return candidate, "task_type_auto", False

    if router.default_route in router.routes and router.default_route not in (visited or set()):
        return router.default_route, "default_route", True
    for name in router.routes:
        if name not in (visited or set()):
            return name, "first_available", True
    raise ValueError("No unvisited route is available.")


def _route_for_execution_profile(
    router: RouterConfig,
    compiled: CompiledTask,
    *,
    visited: set[str] | None = None,
) -> tuple[str, str, bool] | None:
    profile = compiled.execution_profile or {}
    if not profile:
        return None
    visited = visited or set()

    context_size = str(profile.get("context_size") or "")
    source_shape = str(profile.get("source_shape") or "")
    verifiability = str(profile.get("verifiability") or "")
    op_complexity = str(profile.get("operation_complexity") or "")
    strategy = str(profile.get("recommended_strategy") or "")
    semantic_rule_required = bool(profile.get("semantic_rule_required"))
    low_complexity = op_complexity in {"direct_lookup", "single_filter"}

    if verifiability == "weak" or source_shape == "image":
        candidate = _first_route_with_kind(router, ("multi_agent", "react"), visited=visited)
        if candidate is not None:
            return candidate, "profile_weak_or_image", False

    if strategy == "rag_extract" or source_shape == "long_docs":
        candidate = _first_route_with_kind(
            router,
            ("operator_executor", "tablellm_direct", "react", "multi_agent"),
            require_rag=True,
            visited=visited,
        )
        if candidate is not None:
            return candidate, "profile_rag_extract", False

    if semantic_rule_required:
        # Semantic-rule tasks need rule resolution before codegen/judging.
        # Prefer the mixed/tool-first route if configured; avoid sending a
        # small rule task through the weakest easy path first.
        candidate = _first_route_named(
            router,
            ("tool_first_mixed", "hard", "easy"),
            visited=visited,
        )
        if candidate is not None:
            return candidate, "profile_operator_with_rule_resolution", False
        candidate = _first_route_with_kind(
            router,
            ("operator_executor", "tablellm_direct", "react"),
            visited=visited,
        )
        if candidate is not None:
            return candidate, "profile_operator_with_rule_resolution", False

    if (
        context_size == "small"
        and verifiability == "programmatic"
        and low_complexity
        and compiled.file_count <= 2 
        and op_complexity not in {"filter_join", "aggregation", "multi_hop"} 
    ):
        candidate = _first_route_named(router, ("easy",), visited=visited)
        if candidate is not None:
            return candidate, "profile_direct_candidate_verify_as_easy_operator", False
        candidate = _first_route_with_kind(
            router, ("operator_executor", "tablellm_direct"), visited=visited
        )
        if candidate is not None:
            return candidate, "profile_direct_candidate_verify_as_operator", False

    if context_size == "small" and verifiability == "evidence_based" and low_complexity:
        candidate = _first_route_with_kind(
            router, ("react", "operator_executor", "multi_agent"), visited=visited
        )
        if candidate is not None:
            return candidate, "profile_direct_candidate_evidence_stub", False

    if op_complexity in {"filter_join", "aggregation", "multi_hop"} or source_shape == "multi_table":
        candidate = _first_route_with_kind(
            router, ("operator_executor", "tablellm_direct", "react", "multi_agent"),
            visited=visited,
        )
        if candidate is not None:
            return candidate, "profile_operator_complex", False

    return None


def _pick_route(
    router: RouterConfig, task: PublicTask
) -> tuple[RouteConfig, RouterDecision]:
    notes: list[str] = []
    if not router.routes:
        raise ValueError("Router mode is selected but no routes are configured.")

    compiled = compile_task(task)
    raw_difficulty = (task.difficulty or "").strip()
    difficulty_key = raw_difficulty.lower()
    difficulty_source = "task_label"
    estimate_features: dict[str, Any] = {}

    # Difficulty is now a budget hint, not the primary route key. We only
    # estimate it when the label is absent so trace/budget information
    # remains useful on hidden tasks.
    if not raw_difficulty:
        if router.estimate_when_missing:
            estimated, features = _estimate_difficulty(task)
            estimate_features = features
            notes.append(
                f"difficulty_estimated_as_{estimated}"
                + ("" if not raw_difficulty else f"_(label_was_{raw_difficulty})")
            )
            raw_difficulty = estimated
            difficulty_key = raw_difficulty.lower()
            difficulty_source = "heuristic"
        else:
            difficulty_source = "default"

    route_name, route_reason, fallback_used = _route_for_compiled_task(
        router,
        compiled,
        difficulty_key=difficulty_key,
    )

    route = router.routes[route_name]
    decision = RouterDecision(
        difficulty=raw_difficulty,
        task_type=compiled.task_type,
        budget_level=compiled.budget_level,
        needs_reasoner=compiled.needs_reasoner,
        route_name=route_name,
        kind=route.kind,
        api_base=route.api_base,
        model=route.model,
        fallback_used=fallback_used,
        difficulty_source=difficulty_source,
        route_reason=route_reason,
        compiled_task=compiled.to_dict(),
        estimate_features=estimate_features,
        notes=notes,
    )
    return route, decision


def _next_repair_route(
    *,
    router: RouterConfig,
    current_route_name: str,
    visited: set[str],
    compiled: CompiledTask,
    failure_reason: str | None,
) -> str | None:
    """Return a failure-type fallback route, not a difficulty upgrade."""
    if not router.cascade_on_failure:
        return None
    failure_text = str(failure_reason or "")
    failure_type = _failure_type(failure_text, compiled)
    profile = compiled.execution_profile or {}
    small_programmatic = (
        profile.get("context_size") == "small"
        and profile.get("verifiability") == "programmatic"
    )

    if failure_type == "zero_rows":
        return None

    current = router.routes.get(current_route_name)
    current_kind = current.kind.lower() if current is not None else ""

    if failure_type in {"unsupported_file_type", "budget_exhausted"}:
        candidate = _first_route_named(router, ("fallback_multi_agent",), visited=visited)
        if candidate is not None:
            return candidate
        return _first_route_with_kind(router, ("multi_agent", "react"), visited=visited)

    if failure_type in {"retrieval_empty", "doc_context_miss"}:
        return _first_route_with_kind(
            router,
            ("operator_executor", "tablellm_direct", "react", "multi_agent"),
            require_rag=True,
            visited=visited,
        )

    if failure_type == "semantic_consistency_failed":
        candidate = _first_route_named(router, ("tool_first_mixed", "hard", "easy"), visited=visited)
        if candidate is not None:
            return candidate
        return _first_route_with_kind(router, ("operator_executor", "tablellm_direct", "react"), visited=visited)

    if failure_type in {"missing_answer", "syntax_error", "static_error", "exec_error"}:
        if small_programmatic:
            # Local/schema repair already ran inside the route. Do not jump
            # straight to extreme for a small programmatic task; try another
            # tool-first route only if it is not the extreme route.
            for candidate in ("tool_first_mixed", "hard", "easy", "medium"):
                if candidate == "extreme":
                    continue
                if candidate in router.routes and candidate not in visited:
                    return candidate
            return None
        candidate = _first_route_named(
            router,
            ("tool_first_mixed", "hard", "medium", "easy"),
            visited=visited,
        )
        if candidate is not None:
            return candidate
        return _first_route_with_kind(
            router,
            ("operator_executor", "tablellm_direct", "react"),
            visited=visited,
        )

    if current_kind == "react":
        kind_order = ("operator_executor", "tablellm_direct", "multi_agent")
    elif current_kind in {"operator_executor", "tablellm_direct"}:
        kind_order = ("operator_executor", "tablellm_direct", "react")
    else:
        kind_order = ()

    # MultiAgent is a last-resort fallback, not tied to Hard/Extreme.
    if (
        compiled.needs_reasoner
        and "multi_agent" not in kind_order
        and "unsupported_file_type" in set(compiled.ambiguity_flags)
    ):
        kind_order = (*kind_order, "multi_agent")

    candidate = _first_route_with_kind(router, kind_order, visited=visited)
    if candidate is not None:
        return candidate

    # Back-compat: if no typed repair route exists, use the user's old
    # cascade_order, but skip the current route and already-visited ones.
    for candidate in router.cascade_order or ():
        if small_programmatic and candidate == "extreme":
            continue
        if candidate in router.routes and candidate not in visited:
            return candidate
    return None


def _failure_type(failure_text: str, compiled: CompiledTask) -> str:
    text = failure_text.lower()
    if "unsupported_file_type" in set(compiled.ambiguity_flags):
        return "unsupported_file_type"
    if "budget_exhausted" in text:
        return "budget_exhausted"
    if "semantic_consistency_failed" in text or "filter_semantics" in text:
        return "semantic_consistency_failed"
    if "zero rows" in text or "no_rows" in text or "zero_row" in text:
        return "zero_rows"
    if "retrieval_empty" in text or "structured_doc_synth_empty" in text:
        return "retrieval_empty"
    if "doc_context_miss" in text or "not found in" in text:
        return "doc_context_miss"
    if "missing_answer" in text or "did not define an `answer`" in text:
        return "missing_answer"
    if "syntax_error" in text or "python_syntax" in text:
        return "syntax_error"
    if "static_error" in text or "static_check" in text or "no_such_column" in text:
        return "static_error"
    if "exec_error" in text or "keyerror" in text or "indexerror" in text:
        return "exec_error"
    return "unknown"


def _payload_indicates_failure(payload: dict[str, Any]) -> bool:
    if not payload.get("succeeded"):
        return True
    answer = payload.get("answer")
    if not isinstance(answer, dict):
        return True
    cols = answer.get("columns") or []
    if not cols:
        return True
    return False


def _build_route_adapter(
    route: RouteConfig, agent_defaults: AgentConfig | None = None
) -> OpenAIModelAdapter:
    """Construct the per-route adapter, with retry policy inheriting from agent-level."""
    max_retries = route.max_retries
    backoff = route.retry_backoff_seconds
    if max_retries < 0:
        max_retries = agent_defaults.max_retries if agent_defaults else 3
    if backoff < 0:
        backoff = agent_defaults.retry_backoff_seconds if agent_defaults else 1.0
    backoff_max = agent_defaults.retry_backoff_max_seconds if agent_defaults else 16.0
    return OpenAIModelAdapter(
        model=route.model,
        api_base=route.api_base,
        api_key=route.api_key,
        temperature=route.temperature,
        max_tokens=route.max_tokens,
        request_timeout=route.request_timeout,
        max_retries=max_retries,
        retry_backoff_seconds=backoff,
        retry_backoff_max_seconds=backoff_max,
    )


def _build_reasoner_repair_adapter(
    route: RouteConfig,
    agent_config: AgentConfig,
) -> OpenAIModelAdapter:
    repair = agent_config.reasoner_repair
    return OpenAIModelAdapter(
        model=repair.model or route.model,
        api_base=repair.api_base or route.api_base,
        api_key=repair.api_key or route.api_key,
        temperature=repair.temperature,
        max_tokens=repair.max_tokens,
        request_timeout=repair.request_timeout or route.request_timeout,
        max_retries=repair.max_retries,
        retry_backoff_seconds=agent_config.retry_backoff_seconds,
        retry_backoff_max_seconds=agent_config.retry_backoff_max_seconds,
    )


def _budget_limit(configured: int, compiled_value: int) -> int:
    if configured is None or configured < 0:
        return compiled_value
    return min(configured, compiled_value)


def _install_budget_controller(
    *,
    task: PublicTask,
    agent_config: AgentConfig,
    compiled_task: CompiledTask,
) -> BudgetController | None:
    budget_config = agent_config.budget
    if not budget_config.enabled:
        set_budget_controller(None)
        return None
    controller = BudgetController(
        task_id=task.task_id,
        max_llm_calls=_budget_limit(budget_config.max_llm_calls, compiled_task.max_llm_calls),
        max_tool_calls=_budget_limit(budget_config.max_tool_calls, compiled_task.max_tool_calls),
        max_seconds=budget_config.max_seconds,
        max_local_repairs=budget_config.max_local_repairs,
        max_reasoner_repairs=budget_config.max_reasoner_repairs,
        max_multiagent_fallbacks=budget_config.max_multiagent_fallbacks,
    )
    set_budget_controller(controller)
    logger = get_progress_logger()
    if logger is not None:
        logger.budget_started(
            max_llm_calls=controller.max_llm_calls,
            max_tool_calls=controller.max_tool_calls,
            max_seconds=controller.max_seconds,
        )
    return controller


def _attach_budget(payload: dict[str, Any]) -> None:
    budget = get_budget_controller()
    if budget is not None:
        payload["budget"] = budget.to_dict()


# ---------------------------------------------------------------------------
# Per-kind one-pass executors. Each returns a result dict shaped like the
# rest of runner.py expects.
# ---------------------------------------------------------------------------


def _build_rag_kwargs(route: RouteConfig) -> dict[str, Any] | None:
    """Translate the route's :class:`RagConfig` into kwargs for
    :func:`document_retriever.retrieve_top_k`.

    Each stage degrades gracefully:
    - missing embedding_* → BM25-only sparse retrieval.
    - query_expansion_enabled=False → identity transformer.
    - missing reranker_* → no second stage.
    """
    rag = route.rag
    if not rag.enabled:
        return None

    embedding_config = None
    if rag.embedding_model and rag.embedding_api_base:
        from data_agent_baseline.agents.document_retriever import EmbeddingClientConfig
        embedding_config = EmbeddingClientConfig(
            api_base=rag.embedding_api_base,
            api_key=rag.embedding_api_key,
            model=rag.embedding_model,
            request_timeout=rag.embedding_request_timeout,
            batch_size=rag.embedding_batch_size,
        )

    cache_dir: Path | None = None
    if rag.cache_dir:
        candidate = Path(rag.cache_dir)
        if not candidate.is_absolute():
            from data_agent_baseline.config import PROJECT_ROOT
            candidate = PROJECT_ROOT / candidate
        cache_dir = candidate

    # --- Query expansion: build a small, lightly-instrumented LLM client.
    query_expansion_model = None
    if rag.query_expansion_enabled:
        # Reuse the route's main endpoint unless the user gave overrides.
        qx_model = rag.query_expansion_model or route.model
        qx_api_base = rag.query_expansion_api_base or route.api_base
        qx_api_key = rag.query_expansion_api_key or route.api_key
        query_expansion_model = OpenAIModelAdapter(
            model=qx_model,
            api_base=qx_api_base,
            api_key=qx_api_key,
            temperature=0.4,
            max_tokens=512,
            request_timeout=route.request_timeout,
            max_retries=2,
            retry_backoff_seconds=1.0,
            retry_backoff_max_seconds=6.0,
        )

    # --- Reranker (optional cross-encoder).
    reranker_config = None
    if rag.reranker_model and rag.reranker_api_base:
        from data_agent_baseline.agents.document_retriever import RerankerConfig
        reranker_config = RerankerConfig(
            api_base=rag.reranker_api_base,
            api_key=rag.reranker_api_key,
            model=rag.reranker_model,
            request_timeout=rag.reranker_request_timeout,
            endpoint_path=rag.reranker_endpoint_path,
        )

    return {
        "rag_top_k": rag.top_k,
        "rag_max_chunk_chars": rag.max_chunk_chars,
        "rag_chunk_overlap_chars": rag.chunk_overlap_chars,
        "rag_doc_char_budget": rag.doc_char_budget,
        "embedding_config": embedding_config,
        "embedding_cache_dir": cache_dir,
        "use_hybrid": rag.use_hybrid,
        "query_expansion_model": query_expansion_model,
        "query_expansion_paraphrases": rag.query_expansion_paraphrases,
        "query_expansion_include_hyde": rag.query_expansion_include_hyde,
        "reranker_config": reranker_config,
        "first_stage_top_n": rag.first_stage_top_n,
        "rrf_k": rag.rrf_k,
    }


def _run_tablellm_pass(
    *,
    task: PublicTask,
    adapter: OpenAIModelAdapter,
    route: RouteConfig,
    compiled_task: CompiledTask,
    sample_temperature: float | None,
    sample_seed: int | None,
) -> dict[str, Any]:
    agent = CodegenDirectAgent(
        model=adapter,
        max_table_rows=route.tablellm_max_table_rows,
        max_input_chars=route.tablellm_max_input_chars,
        python_timeout=route.tablellm_python_timeout,
        sample_temperature=sample_temperature,
        sample_seed=sample_seed,
        rag_kwargs=_build_rag_kwargs(route),
        compiled=compiled_task,
    )
    result = agent.run(task)
    return {
        "task_id": task.task_id,
        "answer": result.answer.to_dict() if result.answer is not None else None,
        "steps": [],
        "succeeded": result.succeeded,
        "failure_reason": result.failure_reason,
        "agent_mode": "router",
        "tablellm_direct": result.to_dict(),
    }


def _run_operator_executor_pass(
    *,
    task: PublicTask,
    adapter: OpenAIModelAdapter,
    route: RouteConfig,
    compiled_task: CompiledTask,
    sample_temperature: float | None,
    sample_seed: int | None,
) -> dict[str, Any]:
    executor = OperatorExecutor(
        model=adapter,
        compiled_task=compiled_task,
        max_table_rows=route.tablellm_max_table_rows,
        max_input_chars=route.tablellm_max_input_chars,
        python_timeout=route.tablellm_python_timeout,
        sample_temperature=sample_temperature,
        sample_seed=sample_seed,
        rag_kwargs=_build_rag_kwargs(route),
        semantic_consistency_enabled=route.semantic_consistency_enabled,
        semantic_consistency_max_repairs=route.semantic_consistency_max_repairs,
    )
    result = executor.run(task)
    return {
        "task_id": task.task_id,
        "answer": result.answer.to_dict() if result.answer is not None else None,
        "steps": [],
        "succeeded": result.succeeded,
        "failure_reason": result.failure_reason,
        "agent_mode": "router",
        "operator_executor": {
            **result.to_dict(),
            "compiled_task": compiled_task.to_dict(),
        },
        "semantic_consistency_audit": _build_semantic_consistency_audit(result.manifest),
    }


def _build_semantic_consistency_audit(manifest: list[Any] | None) -> dict[str, Any]:
    """Summarize how the semantic-consistency machinery behaved on this task.

    Reads the operator-executor manifest (already a canonical audit log
    of each stage) and distills it into a flat, one-glance-readable
    block that lives at the top of trace.json. This answers the
    question "did the consistency judge actually run, and if not why
    not?" without making the reader dig through ``context_manifest``.

    The audit is non-authoritative: it reports, it does not decide. All
    behavioral choices still live inside the pipeline stages themselves.

    ``final_gate`` is the single most useful field. It takes one of:
      - ``"plan+judge"``            : analyst produced a plan and judge
                                       ran. The strong path.
      - ``"escalated+judge"``       : analyst did NOT produce a plan on
                                       the first try (exception or
                                       gated/skipped), but an escalation
                                       produced one — either via
                                       analyst-exception retry / fallback
                                       or via cheap-guard lazy escalation
                                       — and judge ran against it. The
                                       recovered path.
      - ``"cheap_guard_only"``      : cheap semantic guard flagged risk
                                       but escalation did not build a
                                       plan and judge was not engaged.
                                       Should be rare; investigate.
      - ``"bypassed"``              : no plan, no judge. The task was
                                       deemed low-risk and shipped
                                       without semantic verification.
    """
    plan_ran = False
    plan_failure: dict[str, Any] | None = None
    cheap_guard_triggered_escalation = False
    analyst_exception_retry_succeeded = False
    analyst_exception_fallback_used = False
    lazy_escalation_plan_built = False
    fallback_plan_from_cheap_guard = False
    plan_source: str | None = None
    judge_attempts = 0
    judge_repaired_code = False
    judge_final_verdict: str | None = None

    for entry in (manifest or []):
        if not isinstance(entry, dict):
            continue

        # Capture the plan-cache accelerator status. Written by every
        # branch of OperatorExecutor that attaches a semantic_plan; the
        # last one wins, which correctly reflects the plan that judge
        # actually ran against.
        if "plan_source" in entry:
            ps = entry.get("plan_source")
            if isinstance(ps, str) and ps:
                plan_source = ps

        sc_entry = entry.get("semantic_consistency")
        if isinstance(sc_entry, str):
            # Tag-only entries used by OperatorExecutor to mark the
            # Phase 3d escalation branches.
            if sc_entry == "enabled":
                plan_ran = True
            elif sc_entry == "analyst_exception_retry_succeeded":
                analyst_exception_retry_succeeded = True
                plan_failure = dict(entry.get("original_plan_failure") or {})
            elif sc_entry == "analyst_exception_fallback_plan":
                analyst_exception_fallback_used = True
                plan_failure = dict(entry.get("original_plan_failure") or {})
            elif sc_entry == "lazy_escalation":
                lazy_escalation_plan_built = True
            elif sc_entry == "fallback_plan_from_cheap_guard":
                fallback_plan_from_cheap_guard = True
            # Any string tag starting with a sibling key implies the
            # pipeline was engaged at some layer; treat as "plan path
            # active" for audit readability.
        elif isinstance(sc_entry, dict):
            # SemanticConsistencyPipeline.judge_and_repair ends with
            # result.manifest.append({"semantic_consistency": trace.to_dict()}),
            # which is the only place a dict shows up here. Its shape
            # comes from SemanticConsistencyResult.to_dict().
            judge_history = sc_entry.get("judge_history") or []
            repair_history = sc_entry.get("repair_history") or []
            judge_attempts = len([h for h in judge_history if isinstance(h, dict)])
            judge_repaired_code = any(
                isinstance(h, dict) and h.get("succeeded") for h in repair_history
            )
            judge_final_verdict = (
                str(sc_entry.get("final_verdict"))
                if sc_entry.get("final_verdict") is not None
                else None
            )

        if isinstance(entry.get("cheap_semantic_assessment"), dict):
            cheap_guard_triggered_escalation = True

    # Work out final_gate. Order matters: the strongest claim that
    # applies wins.
    judge_ran = judge_attempts > 0 or judge_final_verdict is not None
    if judge_ran and (
        analyst_exception_retry_succeeded
        or analyst_exception_fallback_used
        or lazy_escalation_plan_built
        or fallback_plan_from_cheap_guard
    ):
        final_gate = "escalated+judge"
    elif judge_ran:
        final_gate = "plan+judge"
    elif cheap_guard_triggered_escalation:
        final_gate = "cheap_guard_only"
    else:
        final_gate = "bypassed"

    return {
        "plan_ran": plan_ran,
        "plan_failure": plan_failure,
        "cheap_guard_triggered_escalation": cheap_guard_triggered_escalation,
        "analyst_exception_retry_succeeded": analyst_exception_retry_succeeded,
        "analyst_exception_fallback_used": analyst_exception_fallback_used,
        "lazy_escalation_plan_built": lazy_escalation_plan_built,
        "fallback_plan_from_cheap_guard": fallback_plan_from_cheap_guard,
        "plan_source": plan_source,
        "plan_cache_hit": plan_source == "cache",
        "judge_ran": judge_ran,
        "judge_attempts": judge_attempts,
        "judge_final_verdict": judge_final_verdict,
        "judge_repaired_code": judge_repaired_code,
        "final_gate": final_gate,
    }


def _run_react_pass_for_route(
    *,
    task: PublicTask,
    adapter: OpenAIModelAdapter,
    route: RouteConfig,
    cache_tool_results: bool,
    sample_temperature: float | None,
    sample_seed: int | None,
) -> dict[str, Any]:
    tools = create_default_tool_registry()
    agent = ReActAgent(
        model=adapter,
        tools=tools,
        config=ReActAgentConfig(
            max_steps=route.max_steps,
            sample_temperature=sample_temperature,
            sample_seed=sample_seed,
            cache_tool_results=cache_tool_results,
        ),
    )
    payload = agent.run(task).to_dict()
    payload["agent_mode"] = "router"
    return payload


def _run_multi_agent_pass_for_route(
    *,
    task: PublicTask,
    adapter: OpenAIModelAdapter,
    route: RouteConfig,
    sample_temperature: float | None,
    sample_seed: int | None,
) -> dict[str, Any]:
    ma = route.multi_agent
    orch_config = OrchestratorConfig(
        planner=PlannerConfig(
            sample_temperature=ma.planner_temperature
            if sample_temperature is None
            else sample_temperature
        ),
        specialist=SpecialistConfig(
            max_steps=ma.specialist_max_steps,
            sample_temperature=ma.specialist_temperature
            if sample_temperature is None
            else sample_temperature,
        ),
        synthesizer=SynthesizerConfig(
            max_steps=ma.synthesizer_max_steps,
            sample_temperature=ma.synthesizer_temperature
            if sample_temperature is None
            else sample_temperature,
        ),
        max_specialist_workers=ma.max_specialist_workers,
        enable_iterative_refinement=ma.enable_iterative_refinement,
    )
    if sample_seed is not None:
        orch_config.planner.sample_seed = sample_seed
        orch_config.specialist.sample_seed = sample_seed
        orch_config.synthesizer.sample_seed = sample_seed
    orchestrator = MultiAgentOrchestrator(model=adapter, config=orch_config)
    result = orchestrator.run(task)
    return {
        "task_id": task.task_id,
        "answer": result.answer.to_dict() if result.answer is not None else None,
        "steps": list(result.synthesizer_steps),
        "succeeded": result.succeeded,
        "failure_reason": result.failure_reason,
        "agent_mode": "router",
        "multi_agent": result.to_dict(),
    }


def _run_route_pass(
    *,
    task: PublicTask,
    adapter: OpenAIModelAdapter,
    route: RouteConfig,
    compiled_task: CompiledTask,
    cache_tool_results: bool,
    sample_temperature: float | None,
    sample_seed: int | None,
) -> dict[str, Any]:
    kind = route.kind.lower()
    if kind == "operator_executor":
        return _run_operator_executor_pass(
            task=task,
            adapter=adapter,
            route=route,
            compiled_task=compiled_task,
            sample_temperature=sample_temperature,
            sample_seed=sample_seed,
        )
    if kind == "tablellm_direct":
        return _run_tablellm_pass(
            task=task,
            adapter=adapter,
            route=route,
            compiled_task=compiled_task,
            sample_temperature=sample_temperature,
            sample_seed=sample_seed,
        )
    if kind == "react":
        return _run_react_pass_for_route(
            task=task,
            adapter=adapter,
            route=route,
            cache_tool_results=cache_tool_results,
            sample_temperature=sample_temperature,
            sample_seed=sample_seed,
        )
    if kind == "multi_agent":
        return _run_multi_agent_pass_for_route(
            task=task,
            adapter=adapter,
            route=route,
            sample_temperature=sample_temperature,
            sample_seed=sample_seed,
        )
    raise ValueError(f"Unknown route kind: {route.kind!r}")


# ---------------------------------------------------------------------------
# Self-consistency wrapper around one route
# ---------------------------------------------------------------------------


def _vote_self_consistency(
    *,
    samples: list[dict[str, Any]],
    sc_config: SelfConsistencyConfig,
    numeric_tolerance: float,
    case_insensitive: bool,
    strip_whitespace: bool,
) -> dict[str, Any]:
    """Apply column-vote self-consistency over a list of route-pass results."""
    successful = [
        sample for sample in samples
        if sample.get("succeeded") and isinstance(sample.get("answer"), dict)
    ]
    if not successful:
        last_failure = next(
            (sample.get("failure_reason") for sample in reversed(samples) if sample.get("failure_reason")),
            "All route samples failed.",
        )
        return {
            "answer": None,
            "steps": [],
            "succeeded": False,
            "failure_reason": last_failure,
            "self_consistency": {
                "num_samples": len(samples),
                "samples": samples,
                "decision": {"aggregator": sc_config.aggregator, "notes": ["no_successful_samples"]},
            },
        }
    if sc_config.aggregator.lower().strip() == "first_success":
        chosen = successful[0]
        return {
            "answer": chosen["answer"],
            "steps": chosen.get("steps") or [],
            "succeeded": True,
            "failure_reason": None,
            "self_consistency": {
                "num_samples": len(samples),
                "aggregator": "first_success",
                "chosen_sample_index": chosen.get("sample_index", 0),
                "samples": samples,
            },
        }

    sample_columns: list[list[list[Any]]] = []
    for sample in successful:
        answer = sample["answer"]
        cols = answer.get("columns") or []
        rows = answer.get("rows") or []
        per_column: list[list[Any]] = [[] for _ in cols]
        for row in rows:
            for idx in range(len(cols)):
                per_column[idx].append(row[idx] if idx < len(row) else None)
        sample_columns.append(per_column)

    sample_signatures = [
        [
            column_signature(
                col,
                numeric_tolerance=numeric_tolerance,
                case_insensitive=case_insensitive,
                strip_whitespace=strip_whitespace,
            )
            for col in cols
        ]
        for cols in sample_columns
    ]

    votes: dict[tuple[str, ...], int] = {}
    for sigs in sample_signatures:
        for sig in set(sigs):
            votes[sig] = votes.get(sig, 0) + 1

    counts = [len(sigs) for sigs in sample_signatures]
    target_count = max(set(counts), key=counts.count) if counts else 0
    eligible = [
        (sig, count) for sig, count in votes.items() if count >= max(sc_config.min_votes, 1)
    ]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    winning = [sig for sig, _ in eligible[:target_count]] or list(sample_signatures[0])
    winning_set = set(winning)

    best_idx = 0
    best_score: tuple[int, int] | None = None
    for idx, sigs in enumerate(sample_signatures):
        sig_set = set(sigs)
        coverage = len(sig_set & winning_set)
        extras = max(len(sig_set) - coverage, 0)
        score = (coverage, -extras)
        if best_score is None or score > best_score:
            best_score = score
            best_idx = idx

    chosen = successful[best_idx]
    chosen_answer = chosen["answer"]
    chosen_cols = list(chosen_answer.get("columns") or [])
    chosen_rows = [list(row) for row in chosen_answer.get("rows") or []]
    sigs_in_chosen = sample_signatures[best_idx]
    sig_to_idx: dict[tuple[str, ...], int] = {}
    for idx, sig in enumerate(sigs_in_chosen):
        sig_to_idx.setdefault(sig, idx)
    if all(sig in sig_to_idx for sig in winning):
        keep = [sig_to_idx[sig] for sig in winning]
        final_answer = {
            "columns": [chosen_cols[i] for i in keep],
            "rows": [[row[i] for i in keep] for row in chosen_rows],
        }
    else:
        final_answer = {"columns": chosen_cols, "rows": chosen_rows}

    return {
        "answer": final_answer,
        "steps": chosen.get("steps") or [],
        "succeeded": True,
        "failure_reason": None,
        "self_consistency": {
            "num_samples": len(samples),
            "aggregator": sc_config.aggregator,
            "voted_signatures": [
                {"signature_preview": list(sig)[:6], "votes": count}
                for sig, count in eligible
            ],
            "chosen_sample_index": best_idx,
            "samples": samples,
        },
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _run_one_route(
    *,
    task: PublicTask,
    route: RouteConfig,
    agent_config: AgentConfig,
    compiled_task: CompiledTask,
    numeric_tolerance: float,
    case_insensitive: bool,
    strip_whitespace: bool,
) -> tuple[dict[str, Any], str]:
    """Run a single route (with its own self-consistency if configured).

    Returns ``(result_payload, sc_origin)``. ``result_payload`` does NOT yet
    have ``router_decision`` attached — the caller does that after picking
    the final route to expose.
    """
    adapter = _build_route_adapter(route, agent_config)

    # Decide which self-consistency block applies for this route.
    if route.self_consistency.num_samples > 0:
        sc_config = route.self_consistency
        sc_origin = "route"
    else:
        sc_config = agent_config.self_consistency
        sc_origin = "agent"

    if sc_config.num_samples <= 1:
        result = _run_route_pass(
            task=task,
            adapter=adapter,
            route=route,
            compiled_task=compiled_task,
            cache_tool_results=agent_config.cache_tool_results,
            sample_temperature=route.temperature if route.temperature > 0 else None,
            sample_seed=None,
        )
        _stamp_validation(result)
        return result, sc_origin

    samples: list[dict[str, Any]] = []
    for sample_index in range(sc_config.num_samples):
        try:
            sample = _run_route_pass(
                task=task,
                adapter=adapter,
                route=route,
                compiled_task=compiled_task,
                cache_tool_results=agent_config.cache_tool_results,
                sample_temperature=sc_config.sample_temperature,
                sample_seed=sample_index + 1,
            )
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            sample = {
                "task_id": task.task_id,
                "answer": None,
                "succeeded": False,
                "failure_reason": f"sample_runtime_error: {exc}",
                "agent_mode": "router",
            }
        sample["sample_index"] = sample_index
        samples.append(sample)

    voted = _vote_self_consistency(
        samples=samples,
        sc_config=sc_config,
        numeric_tolerance=numeric_tolerance,
        case_insensitive=case_insensitive,
        strip_whitespace=strip_whitespace,
    )
    _stamp_validation(voted)
    return voted, sc_origin


def _try_reasoner_repair(
    *,
    task: PublicTask,
    route: RouteConfig,
    agent_config: AgentConfig,
    compiled_task: CompiledTask,
    payload: dict[str, Any],
    numeric_tolerance: float,
    case_insensitive: bool,
    strip_whitespace: bool,
) -> dict[str, Any]:
    if not agent_config.reasoner_repair.enabled:
        return payload
    if not _payload_indicates_failure(payload):
        return payload
    if route.kind.lower() not in {"operator_executor", "tablellm_direct"}:
        return payload
    flags = set(compiled_task.ambiguity_flags)
    reasoner_allowed = (
        compiled_task.needs_reasoner
        or (
            compiled_task.task_type == "document_qa"
            and "large_document_context" in flags
        )
    )
    if not reasoner_allowed:
        return payload
    failed_block = payload.get("operator_executor") or payload.get("tablellm_direct") or {}
    if isinstance(failed_block, dict) and not failed_block.get("program"):
        return payload

    budget = get_budget_controller()
    if budget is not None and not budget.can_reasoner_repair():
        payload.setdefault("reasoner_repair", {
            "attempted": False,
            "succeeded": False,
            "failure_reason": "budget_exhausted:max_reasoner_repairs",
        })
        return payload

    logger = get_progress_logger()
    failure_reason = str(payload.get("failure_reason") or "no answer")
    if logger is not None:
        logger.reasoner_repair_start(failure_reason=failure_reason)

    if budget is not None:
        budget.consume_reasoner_repair("router.reasoner_repair")
    try:
        repair_payload = run_reasoner_repair(
            task=task,
            model=_build_reasoner_repair_adapter(route, agent_config),
            compiled_task=compiled_task,
            failed_payload=payload,
            python_timeout=route.tablellm_python_timeout,
        )
    except BudgetExceeded:
        raise
    except Exception as exc:  # noqa: BLE001
        repair_payload = {
            "attempted": True,
            "succeeded": False,
            "failure_reason": f"repair_runtime_error:{exc}",
        }

    if logger is not None:
        logger.reasoner_repair_done(
            succeeded=bool(repair_payload.get("succeeded")),
            failure_reason=repair_payload.get("failure_reason"),
        )

    if repair_payload.get("succeeded") and isinstance(repair_payload.get("answer"), dict):
        repaired = {
            "task_id": task.task_id,
            "answer": repair_payload["answer"],
            "steps": [],
            "succeeded": True,
            "failure_reason": None,
            "agent_mode": "router",
            "reasoner_repair": {
                **repair_payload,
                "repaired_failure_reason": payload.get("failure_reason"),
            },
            "primary_failed_payload": payload,
        }
        _stamp_validation(repaired)
        return repaired

    payload["reasoner_repair"] = repair_payload
    return payload


# ---------------------------------------------------------------------------
# Cross-model verification (column-signature intersection)
# ---------------------------------------------------------------------------


def _route_with_verifier(
    primary: RouteConfig,
    verifier: CrossModelVerifierSpec,
) -> RouteConfig:
    """Clone the primary route, replacing model + endpoint with the verifier's.

    All other knobs (kind, multi_agent topology, RAG config, max_steps,
    etc.) are inherited verbatim. The whole point is to run the SAME
    pipeline on a SECOND model.
    """
    from dataclasses import replace as _dc_replace

    return _dc_replace(
        primary,
        name=f"{primary.name}+{verifier.name}",
        model=verifier.model or primary.model,
        api_base=verifier.api_base or primary.api_base,
        api_key=verifier.api_key or primary.api_key,
        temperature=(
            primary.temperature if verifier.temperature is None else verifier.temperature
        ),
        max_tokens=(
            primary.max_tokens if verifier.max_tokens is None else verifier.max_tokens
        ),
        request_timeout=(
            primary.request_timeout if verifier.request_timeout < 0 else verifier.request_timeout
        ),
        max_retries=(
            primary.max_retries if verifier.max_retries < 0 else verifier.max_retries
        ),
    )


def _columns_from_payload(payload: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    answer = payload.get("answer") or {}
    cols = list(answer.get("columns") or [])
    rows = [list(row) for row in (answer.get("rows") or [])]
    return cols, rows


def _per_column_signatures(
    cols: list[str],
    rows: list[list[Any]],
    *,
    numeric_tolerance: float,
    case_insensitive: bool,
    strip_whitespace: bool,
) -> list[tuple[str, ...]]:
    transposed: list[list[Any]] = [[] for _ in cols]
    for row in rows:
        for idx in range(len(cols)):
            transposed[idx].append(row[idx] if idx < len(row) else None)
    return [
        column_signature(
            col_values,
            numeric_tolerance=numeric_tolerance,
            case_insensitive=case_insensitive,
            strip_whitespace=strip_whitespace,
        )
        for col_values in transposed
    ]


def _intersect_payloads(
    *,
    primary_payload: dict[str, Any],
    verifier_payloads: list[tuple[str, dict[str, Any]]],
    min_agreement: int,
    numeric_tolerance: float,
    case_insensitive: bool,
    strip_whitespace: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return (intersected_answer_dict_or_None, debug_block).

    A column from the primary survives iff its content signature also
    appears in at least ``min_agreement - 1`` of the verifier payloads
    (the primary itself counts as 1, hence -1).

    ``intersected_answer_dict_or_None`` is None when the intersection is
    empty — the caller is expected to fall back to the primary in that
    case.
    """
    primary_cols, primary_rows = _columns_from_payload(primary_payload)
    primary_sigs = _per_column_signatures(
        primary_cols,
        primary_rows,
        numeric_tolerance=numeric_tolerance,
        case_insensitive=case_insensitive,
        strip_whitespace=strip_whitespace,
    )

    verifier_sig_sets: list[tuple[str, set[tuple[str, ...]], int]] = []
    for verifier_name, payload in verifier_payloads:
        cols, rows = _columns_from_payload(payload)
        sigs = _per_column_signatures(
            cols, rows,
            numeric_tolerance=numeric_tolerance,
            case_insensitive=case_insensitive,
            strip_whitespace=strip_whitespace,
        )
        verifier_sig_sets.append((verifier_name, set(sigs), len(cols)))

    keep_indices: list[int] = []
    column_decisions: list[dict[str, Any]] = []
    needed_agreements = max(min_agreement - 1, 0)
    for idx, sig in enumerate(primary_sigs):
        agree = [
            verifier_name
            for verifier_name, sig_set, _ in verifier_sig_sets
            if sig in sig_set
        ]
        decision_block = {
            "column_index": idx,
            "column_name": primary_cols[idx],
            "agreed_by": agree,
            "kept": False,
        }
        if len(agree) >= needed_agreements:
            keep_indices.append(idx)
            decision_block["kept"] = True
        column_decisions.append(decision_block)

    debug = {
        "min_agreement": min_agreement,
        "primary_column_count": len(primary_cols),
        "verifier_column_counts": {n: count for n, _, count in verifier_sig_sets},
        "column_decisions": column_decisions,
        "intersection_kept_indices": keep_indices,
    }

    if not keep_indices:
        return None, debug
    if keep_indices == list(range(len(primary_cols))):
        # Full agreement; nothing to project.
        return {
            "columns": primary_cols,
            "rows": primary_rows,
        }, debug

    projected_cols = [primary_cols[i] for i in keep_indices]
    projected_rows = [[row[i] if i < len(row) else None for i in keep_indices] for row in primary_rows]
    return {
        "columns": projected_cols,
        "rows": projected_rows,
    }, debug


def _run_cross_model_verify(
    *,
    task: PublicTask,
    primary_route: RouteConfig,
    primary_payload: dict[str, Any],
    config: CrossModelVerifyConfig,
    agent_config: AgentConfig,
    numeric_tolerance: float,
    case_insensitive: bool,
    strip_whitespace: bool,
) -> dict[str, Any]:
    """Run all verifiers, intersect, and (if useful) replace the primary
    answer with the intersection. Mutates and returns ``primary_payload``.
    """
    if not config.enabled or not config.verifiers:
        return primary_payload
    if _payload_indicates_failure(primary_payload):
        # Don't waste verifier calls when the primary already came back
        # empty — cascade has already taken its shot.
        primary_payload["cross_model_verify"] = {
            "skipped_reason": "primary_failed",
        }
        return primary_payload

    # Build (verifier_name, route_clone) pairs.
    verifier_routes = [
        (verifier.name, _route_with_verifier(primary_route, verifier))
        for verifier in config.verifiers
    ]
    logger = get_progress_logger()
    if logger is not None:
        logger.cross_verify_start(verifier_names=[name for name, _ in verifier_routes])

    def _runner(item: tuple[str, RouteConfig]) -> tuple[str, dict[str, Any]]:
        name, vroute = item
        try:
            payload, _ = _run_one_route(
                task=task,
                route=vroute,
                agent_config=agent_config,
                compiled_task=compile_task(task),
                numeric_tolerance=numeric_tolerance,
                case_insensitive=case_insensitive,
                strip_whitespace=strip_whitespace,
            )
            return name, payload
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            return name, {
                "answer": None,
                "succeeded": False,
                "failure_reason": f"verifier_runtime_error: {exc}",
            }

    if config.parallel and len(verifier_routes) > 1:
        with ThreadPoolExecutor(max_workers=len(verifier_routes)) as pool:
            verifier_results = list(pool.map(_runner, verifier_routes))
    else:
        verifier_results = [_runner(item) for item in verifier_routes]

    intersected, debug = _intersect_payloads(
        primary_payload=primary_payload,
        verifier_payloads=verifier_results,
        min_agreement=max(config.min_agreement, 1),
        numeric_tolerance=numeric_tolerance,
        case_insensitive=case_insensitive,
        strip_whitespace=strip_whitespace,
    )

    verify_block: dict[str, Any] = {
        "applied": True,
        "verifiers": [
            {
                "name": name,
                "model": route.model,
                "api_base": route.api_base,
                "succeeded": not _payload_indicates_failure(payload),
                "failure_reason": payload.get("failure_reason"),
                "answer": payload.get("answer"),
            }
            for (name, route), (_, payload) in zip(verifier_routes, verifier_results)
        ],
        "intersection": debug,
    }

    if intersected is None:
        verify_block["outcome"] = "empty_intersection_keep_primary"
    elif intersected["columns"] == primary_payload.get("answer", {}).get("columns") and \
            intersected["rows"] == primary_payload.get("answer", {}).get("rows"):
        verify_block["outcome"] = "full_agreement_keep_primary"
    else:
        verify_block["outcome"] = "intersection_replaces_primary"
        # Stash the original primary so we can audit how the intersection
        # changed the answer if a regression happens.
        verify_block["primary_answer_before_intersection"] = primary_payload.get("answer")
        primary_payload["answer"] = intersected
        # Re-validate after projection (should still pass; mostly a sanity check).
        _stamp_validation(primary_payload)

    primary_payload["cross_model_verify"] = verify_block
    if logger is not None:
        logger.cross_verify_done(
            outcome=verify_block["outcome"],
            column_decisions=debug.get("column_decisions") or [],
        )
    return primary_payload


def run_router(
    *,
    task: PublicTask,
    agent_config: AgentConfig,
    numeric_tolerance: float,
    case_insensitive: bool,
    strip_whitespace: bool,
) -> dict[str, Any]:
    router = agent_config.router
    if not router.enabled or not router.routes:
        raise ValueError(
            "Router mode requires agent.router.enabled=true with at least one route."
        )

    route, decision = _pick_route(router, task)
    visited: set[str] = {decision.route_name}
    source_capabilities = [
        SourceCapability(
            path=str(item.get("path", "")),
            kind=str(item.get("kind", "unknown")),
            bytes=int(item.get("bytes", 0) or 0),
            role=str(item.get("role", "data")),
            tool=str(item.get("tool", "")),
            tables=list(item.get("tables") or []),
            columns=list(item.get("columns") or []),
            row_count=int(item.get("row_count", 0) or 0),
            json_top_keys=list(item.get("json_top_keys") or []),
            json_record_fields=list(item.get("json_record_fields") or []),
            json_record_count=int(item.get("json_record_count", 0) or 0),
            sample=list(item.get("sample") or []),
            structured_records=list(item.get("structured_records") or []),
            structured_record_fields=list(item.get("structured_record_fields") or []),
            structured_record_count=int(item.get("structured_record_count", 0) or 0),
            structured_record_splitter=str(item.get("structured_record_splitter", "")),
            structured_id_pattern=str(item.get("structured_id_pattern", "")),
            column_value_samples={
                str(k): list(v)
                for k, v in (item.get("column_value_samples") or {}).items()
            },
            column_dtypes={
                str(k): str(v)
                for k, v in (item.get("column_dtypes") or {}).items()
            },
            column_cardinalities={
                str(k): int(v)
                for k, v in (item.get("column_cardinalities") or {}).items()
            },
            scan_error=str(item.get("scan_error", "")),
        )
        for item in (decision.compiled_task.get("source_capabilities") or [])
        if isinstance(item, dict)
    ]
    compiled_task = CompiledTask(
        task_type=str(decision.compiled_task.get("task_type", "pure_reasoning")),
        answer_type=str(decision.compiled_task.get("answer_type", "table")),
        data_sources=list(decision.compiled_task.get("data_sources") or []),
        modalities=list(decision.compiled_task.get("modalities") or []),
        operations=list(decision.compiled_task.get("operations") or []),
        primary_tool=str(decision.compiled_task.get("primary_tool", "react")),
        auxiliary_tools=list(decision.compiled_task.get("auxiliary_tools") or []),
        preferred_tools=list(decision.compiled_task.get("preferred_tools") or []),
        source_capabilities=source_capabilities,
        foreign_key_candidates=list(decision.compiled_task.get("foreign_key_candidates") or []),
        needs_reasoner=bool(decision.compiled_task.get("needs_reasoner", False)),
        needs_vision=bool(decision.compiled_task.get("needs_vision", False)),
        task_type_confidence=float(decision.compiled_task.get("task_type_confidence", 1.0)),
        operation_confidence=float(decision.compiled_task.get("operation_confidence", 1.0)),
        source_confidence=float(decision.compiled_task.get("source_confidence", 1.0)),
        ambiguity_flags=list(decision.compiled_task.get("ambiguity_flags") or []),
        budget_level=str(decision.compiled_task.get("budget_level", "medium")),
        max_llm_calls=int(decision.compiled_task.get("max_llm_calls", 4)),
        max_tool_calls=int(decision.compiled_task.get("max_tool_calls", 12)),
        context_bytes=int(decision.compiled_task.get("context_bytes", 0)),
        file_count=int(decision.compiled_task.get("file_count", 0)),
        execution_profile=dict(decision.compiled_task.get("execution_profile") or {}),
        notes=list(decision.compiled_task.get("notes") or []),
    )

    logger = get_progress_logger()
    _install_budget_controller(
        task=task,
        agent_config=agent_config,
        compiled_task=compiled_task,
    )
    if logger is not None:
        logger.task_compiled(compiled=decision.compiled_task)
        logger.router_decision(
            difficulty=decision.difficulty,
            route_name=decision.route_name,
            kind=decision.kind,
            model=decision.model,
            difficulty_source=decision.difficulty_source,
            task_type=decision.task_type,
            budget_level=decision.budget_level,
            needs_reasoner=decision.needs_reasoner,
        )

    payload: dict[str, Any] | None = None
    sc_origin = "none"
    try:
        # First attempt.
        payload, sc_origin = _run_one_route(
            task=task,
            route=route,
            agent_config=agent_config,
            compiled_task=compiled_task,
            numeric_tolerance=numeric_tolerance,
            case_insensitive=case_insensitive,
            strip_whitespace=strip_whitespace,
        )
        decision.cascade_attempts.append({
            "route_name": decision.route_name,
            "kind": route.kind,
            "model": route.model,
            "succeeded": not _payload_indicates_failure(payload),
            "failure_reason": payload.get("failure_reason"),
            "self_consistency_origin": sc_origin,
        })

        repaired_payload = _try_reasoner_repair(
            task=task,
            route=route,
            agent_config=agent_config,
            compiled_task=compiled_task,
            payload=payload,
            numeric_tolerance=numeric_tolerance,
            case_insensitive=case_insensitive,
            strip_whitespace=strip_whitespace,
        )
        if repaired_payload is not payload:
            decision.notes.append("reasoner_repair_succeeded")
            decision.cascade_attempts[-1]["reasoner_repair_succeeded"] = True
            payload = repaired_payload

        # Cascade if needed.
        extra_attempts = 0
        while (
            _payload_indicates_failure(payload)
            and router.cascade_on_failure
            and extra_attempts < max(router.cascade_max_extra_attempts, 0)
        ):
            next_name = _next_repair_route(
                router=router,
                current_route_name=decision.route_name,
                visited=visited,
                compiled=compiled_task,
                failure_reason=str(payload.get("failure_reason") or ""),
            )
            if next_name is None:
                decision.notes.append("repair_cascade_exhausted")
                break

            decision.notes.append(
                f"repair_from_{decision.route_name}_to_{next_name}_after_failure"
            )
            if logger is not None:
                logger.router_cascade(
                    from_route=decision.route_name,
                    to_route=next_name,
                    reason=str(payload.get("failure_reason") or "no answer"),
                )
            decision.route_name = next_name
            next_route = router.routes[next_name]
            decision.kind = next_route.kind
            decision.api_base = next_route.api_base
            decision.model = next_route.model
            visited.add(next_name)

            payload, sc_origin = _run_one_route(
                task=task,
                route=next_route,
                agent_config=agent_config,
                compiled_task=compiled_task,
                numeric_tolerance=numeric_tolerance,
                case_insensitive=case_insensitive,
                strip_whitespace=strip_whitespace,
            )
            decision.cascade_attempts.append({
                "route_name": next_name,
                "kind": next_route.kind,
                "model": next_route.model,
                "succeeded": not _payload_indicates_failure(payload),
                "failure_reason": payload.get("failure_reason"),
                "self_consistency_origin": sc_origin,
            })
            repaired_payload = _try_reasoner_repair(
                task=task,
                route=next_route,
                agent_config=agent_config,
                compiled_task=compiled_task,
                payload=payload,
                numeric_tolerance=numeric_tolerance,
                case_insensitive=case_insensitive,
                strip_whitespace=strip_whitespace,
            )
            if repaired_payload is not payload:
                decision.notes.append("reasoner_repair_succeeded")
                decision.cascade_attempts[-1]["reasoner_repair_succeeded"] = True
                payload = repaired_payload
            extra_attempts += 1

        # Cross-model verification stage. Only applies on routes the user
        # opted in for and skipped if primary came back empty.
        cmv = router.cross_model_verify
        final_route_name = decision.route_name
        final_route = router.routes.get(final_route_name, route)
        # Cross-model verification is opt-in via config. When enabled for a
        # route, apply it to the actual final executor kind instead of
        # silently skipping tool-first routes; otherwise the config can say
        # "verify hard/extreme" while the code never does.
        verify_kind_allowlist = {"operator_executor", "tablellm_direct", "multi_agent"}
        if (
            cmv.enabled
            and cmv.verifiers
            and final_route.kind.lower() in verify_kind_allowlist
            and (not cmv.apply_to_routes or final_route_name in cmv.apply_to_routes)
        ):
            primary_route_for_verify = final_route
            payload = _run_cross_model_verify(
                task=task,
                primary_route=primary_route_for_verify,
                primary_payload=payload,
                config=cmv,
                agent_config=agent_config,
                numeric_tolerance=numeric_tolerance,
                case_insensitive=case_insensitive,
                strip_whitespace=strip_whitespace,
            )
    except BudgetExceeded as exc:
        decision.notes.append("budget_exhausted")
        if payload is not None and isinstance(payload.get("answer"), dict):
            payload["budget_exhausted"] = True
            payload["failure_reason"] = str(exc)
            payload["succeeded"] = True
        else:
            payload = {
                "task_id": task.task_id,
                "answer": None,
                "steps": [],
                "succeeded": False,
                "failure_reason": str(exc),
                "agent_mode": "router",
                "budget_exhausted": True,
            }

    payload.update({
        "task_id": task.task_id,
        "agent_mode": "router",
        "router_decision": decision.to_dict(),
        "self_consistency_origin": sc_origin,
    })
    profile = compiled_task.execution_profile or {}
    if profile.get("direct_llm_candidate_allowed"):
        payload.setdefault("direct_candidate", {
            "enabled": False,
            "role": profile.get("direct_candidate_role", "proposal_only"),
            "reason": (
                "stubbed: direct LLM candidate is only allowed after verifier/evidence judge; "
                "current patch routes to the verifying executor instead of submitting it"
            ),
        })
    _attach_budget(payload)
    set_budget_controller(None)
    return payload
