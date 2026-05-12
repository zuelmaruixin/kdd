from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "runs"


def _default_gold_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "output"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)
    gold_root: Path = field(default_factory=_default_gold_root)


@dataclass(frozen=True, slots=True)
class SelfConsistencyConfig:
    """Cross-way / multi-sample voting controls (TableLLM-style)."""

    num_samples: int = 1
    sample_temperature: float = 0.7
    aggregator: str = "column_vote"  # column_vote | first_success
    min_votes: int = 1
    keep_failed_samples: bool = False


@dataclass(frozen=True, slots=True)
class MultiAgentConfig:
    """Knobs for the planner / specialists / synthesizer pipeline."""

    enabled: bool = False
    planner_temperature: float | None = None
    specialist_max_steps: int = 12
    specialist_temperature: float | None = None
    synthesizer_max_steps: int = 6
    synthesizer_temperature: float | None = None
    max_specialist_workers: int = 4
    enable_iterative_refinement: bool = True


@dataclass(frozen=True, slots=True)
class RagConfig:
    """RAG knobs for long-document tasks.

    Production-grade pipeline:
        chunk → query_expansion (LLM, opt-in)
              → BM25 + embedding hybrid retrieval (RRF fused)
              → cross-encoder rerank (opt-in)
              → top-K

    Setting fewer fields just disables the corresponding stages:
    leave ``embedding_*`` empty for sparse-only; leave
    ``reranker_*`` empty to skip second-stage rerank;
    leave ``query_expansion_enabled`` false to skip multi-query.
    """

    enabled: bool = False
    top_k: int = 6
    first_stage_top_n: int = 30
    rrf_k: int = 60
    max_chunk_chars: int = 1200
    chunk_overlap_chars: int = 150
    doc_char_budget: int = 6000

    # Hybrid retrieval (BM25 + embeddings). When embedding_* is empty we
    # fall back to BM25-only regardless of this flag.
    use_hybrid: bool = True

    # Embedding endpoint (optional, OpenAI-compatible /v1/embeddings).
    embedding_model: str = ""
    embedding_api_base: str = ""
    embedding_api_key: str = ""
    embedding_request_timeout: float = 60.0
    embedding_batch_size: int = 32

    # Persistent embedding cache directory (relative paths resolved against PROJECT_ROOT).
    cache_dir: str = "artifacts/cache/embeddings"

    # LLM-based query expansion (paraphrases + HyDE). Reuses the route's
    # main agent endpoint by default; provide overrides if you want a
    # cheaper / faster model just for expansion.
    query_expansion_enabled: bool = False
    query_expansion_paraphrases: int = 2
    query_expansion_include_hyde: bool = True
    query_expansion_model: str = ""
    query_expansion_api_base: str = ""
    query_expansion_api_key: str = ""

    # Cross-encoder reranker (e.g. DashScope gte-rerank, Cohere rerank).
    reranker_model: str = ""
    reranker_api_base: str = ""
    reranker_api_key: str = ""
    reranker_endpoint_path: str = "/rerank"
    reranker_request_timeout: float = 60.0


@dataclass(frozen=True, slots=True)
class RouteConfig:
    """One path inside the task-type router.

    `kind` selects the agent flavor that handles tasks routed here:
        - "agentic_operator": Planner -> tool action -> reflection -> optional retry
        - "operator_executor": tool-first pandas/SQL/RAG code execution
        - "codegen_direct" : legacy alias for one-shot executable codegen
        - "tablellm_direct": legacy alias kept for old configs
        - "react"          : the existing single-agent ReAct loop
        - "multi_agent"    : Planner -> Specialists -> Synthesizer
    Each route may point at its own OpenAI-compatible endpoint, so you
    can mix providers (e.g. TableLLM via DeepInfra + Qwen via DashScope).
    """

    name: str = "default"
    kind: str = "react"
    model: str = ""
    api_base: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int | None = None
    request_timeout: float = 120.0
    # Per-route retry policy override (-1 means: inherit from agent-level).
    max_retries: int = -1
    retry_backoff_seconds: float = -1.0
    # ReAct-only knob.
    max_steps: int = 16
    # ReAct-only: how many extra `answer` rounds to require for
    # self-verification. 0 = legacy behavior (commit on first answer).
    verification_rounds: int = 1
    # codegen_direct / operator_executor knobs. The tablellm_* field
    # names are kept for config compatibility; YAML may use codegen_*.
    tablellm_max_table_rows: int = 50
    tablellm_max_input_chars: int = 12000
    tablellm_python_timeout: int = 30
    # multi_agent-only knobs.
    multi_agent: MultiAgentConfig = field(default_factory=MultiAgentConfig)
    # Per-route self-consistency override; if num_samples == 0 the router
    # falls back to the agent-level self_consistency block.
    self_consistency: SelfConsistencyConfig = field(default_factory=SelfConsistencyConfig)
    # Long-doc RAG knobs (off by default).
    rag: RagConfig = field(default_factory=RagConfig)
    # Dual-path semantic analyst / consistency judge. Applied only by
    # operator_executor and only on table-centric tasks.
    semantic_consistency_enabled: bool = True
    semantic_consistency_max_repairs: int = 3


@dataclass(frozen=True, slots=True)
class CrossModelVerifierSpec:
    """One additional model that gets to re-solve the same task in parallel
    with the primary route, used by ``CrossModelVerifyConfig``.

    Inherits the primary route's agent kind (agentic_operator /
    multi_agent / react / tablellm_direct) and per-stage knobs; only the
    model + endpoint are swapped out. This lets you run e.g. a Qwen
    agent and a DeepSeek agent over the same topology without duplicating
    the rest of the route config.
    """

    name: str = ""
    model: str = ""
    api_base: str = ""
    api_key: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    request_timeout: float = -1.0
    max_retries: int = -1


@dataclass(frozen=True, slots=True)
class CrossModelVerifyConfig:
    """Cross-model column-signature verification.

    For routes listed in ``apply_to_routes``, the router runs the primary
    plus every entry in ``verifiers`` in parallel and intersects their
    answer column-signatures: a column survives only if at least
    ``min_agreement`` models produced an identical signature for it.
    Empty intersection -> fall back to the primary's untouched answer.

    The math: for the official rubric
    ``score = max(0, recall - lambda * extra/pred)``, taking the
    intersection strictly dominates "submit primary alone" whenever
    the disagreement columns are wrong, and only loses in the rare case
    where all disagreement columns happen to be in gold.
    """

    enabled: bool = False
    apply_to_routes: tuple[str, ...] = ()
    verifiers: tuple[CrossModelVerifierSpec, ...] = ()
    min_agreement: int = 2
    parallel: bool = True


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    enabled: bool = True
    max_llm_calls: int = -1
    max_tool_calls: int = -1
    max_seconds: float = 0.0
    # Repair-loop ceilings — how many deterministic local patches and
    # reasoner-driven rewrites a single task may consume before the
    # router gives up. Multi-agent fallback is the last-resort tier.
    max_local_repairs: int = 3
    max_reasoner_repairs: int = 1
    max_multiagent_fallbacks: int = 1


@dataclass(frozen=True, slots=True)
class ReasonerRepairConfig:
    enabled: bool = True
    model: str = ""
    api_base: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int | None = 1024
    request_timeout: float = 120.0
    max_retries: int = 1


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Top-level router config (only consumed when agent.mode == 'router')."""

    enabled: bool = False
    routes: dict[str, RouteConfig] = field(default_factory=dict)
    task_type_routing: dict[str, str] = field(default_factory=dict)
    difficulty_routing: dict[str, str] = field(default_factory=dict)
    default_route: str = "medium"
    # When task.difficulty is missing/unknown, classify it from the question
    # + context shape using a deterministic heuristic.
    estimate_when_missing: bool = True
    # If a route fails to produce an AnswerTable, automatically retry on
    # the next-heavier route. The order is the keys of `routes` as listed
    # in `cascade_order`; an empty list disables cascading.
    cascade_on_failure: bool = True
    cascade_order: tuple[str, ...] = ("easy", "medium", "hard", "extreme")
    cascade_max_extra_attempts: int = 1
    cross_model_verify: CrossModelVerifyConfig = field(default_factory=CrossModelVerifyConfig)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    provider: str = "openai_compatible"  # openai_compatible | scripted
    mode: str = "react"  # react | multi_agent | router
    model: str = "qwen-plus"
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    max_steps: int = 16
    temperature: float = 0.0
    max_tokens: int | None = None
    request_timeout: float = 120.0
    # Transient-error retry policy (network / 429 / 5xx).
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    retry_backoff_max_seconds: float = 16.0
    # Cache observations of read-only tool calls within a single ReAct loop.
    cache_tool_results: bool = True
    self_consistency: SelfConsistencyConfig = field(default_factory=SelfConsistencyConfig)
    multi_agent: MultiAgentConfig = field(default_factory=MultiAgentConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    reasoner_repair: ReasonerRepairConfig = field(default_factory=ReasonerRepairConfig)
    router: RouterConfig = field(default_factory=RouterConfig)


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    redundancy_lambda: float = 0.5
    numeric_tolerance: float = 1e-2
    case_insensitive: bool = True
    strip_whitespace: bool = True


@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    run_id: str | None = None
    max_workers: int = 4
    task_timeout_seconds: int = 600


@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _load_self_consistency(payload: dict, defaults: SelfConsistencyConfig) -> SelfConsistencyConfig:
    if not payload:
        return defaults
    return SelfConsistencyConfig(
        num_samples=int(payload.get("num_samples", defaults.num_samples)),
        sample_temperature=float(payload.get("sample_temperature", defaults.sample_temperature)),
        aggregator=str(payload.get("aggregator", defaults.aggregator)).strip().lower(),
        min_votes=int(payload.get("min_votes", defaults.min_votes)),
        keep_failed_samples=_bool_value(
            payload.get("keep_failed_samples"), defaults.keep_failed_samples
        ),
    )


def _opt_float(value: Any, default: float | None) -> float | None:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    return float(value)


def _bool_value(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _normalized_agent_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    valid_modes = {"react", "multi_agent", "router"}
    if mode not in valid_modes:
        raise ValueError(
            f"agent.mode must be one of {sorted(valid_modes)}, got {value!r}."
        )
    return mode


def _normalized_route_kind(value: Any) -> str:
    kind = str(value).strip().lower() or "react"
    if kind == "codegen_direct":
        return "tablellm_direct"
    valid_kinds = {
        "agentic_operator",
        "operator_executor",
        "tablellm_direct",
        "react",
        "multi_agent",
    }
    if kind not in valid_kinds:
        raise ValueError(
            f"agent.router.routes.<name>.kind must be one of {sorted(valid_kinds)}, got {value!r}."
        )
    return kind


def _load_rag(payload: dict, defaults: RagConfig) -> RagConfig:
    if not payload:
        return defaults
    return RagConfig(
        enabled=_bool_value(payload.get("enabled"), defaults.enabled),
        top_k=int(payload.get("top_k", defaults.top_k)),
        first_stage_top_n=int(payload.get("first_stage_top_n", defaults.first_stage_top_n)),
        rrf_k=int(payload.get("rrf_k", defaults.rrf_k)),
        max_chunk_chars=int(payload.get("max_chunk_chars", defaults.max_chunk_chars)),
        chunk_overlap_chars=int(payload.get("chunk_overlap_chars", defaults.chunk_overlap_chars)),
        doc_char_budget=int(payload.get("doc_char_budget", defaults.doc_char_budget)),
        use_hybrid=_bool_value(payload.get("use_hybrid"), defaults.use_hybrid),
        embedding_model=str(payload.get("embedding_model", defaults.embedding_model)),
        embedding_api_base=str(payload.get("embedding_api_base", defaults.embedding_api_base)),
        embedding_api_key=str(payload.get("embedding_api_key", defaults.embedding_api_key)),
        embedding_request_timeout=float(
            payload.get("embedding_request_timeout", defaults.embedding_request_timeout)
        ),
        embedding_batch_size=int(payload.get("embedding_batch_size", defaults.embedding_batch_size)),
        cache_dir=str(payload.get("cache_dir", defaults.cache_dir)),
        query_expansion_enabled=_bool_value(
            payload.get("query_expansion_enabled"), defaults.query_expansion_enabled
        ),
        query_expansion_paraphrases=int(
            payload.get("query_expansion_paraphrases", defaults.query_expansion_paraphrases)
        ),
        query_expansion_include_hyde=_bool_value(
            payload.get("query_expansion_include_hyde"), defaults.query_expansion_include_hyde
        ),
        query_expansion_model=str(
            payload.get("query_expansion_model", defaults.query_expansion_model)
        ),
        query_expansion_api_base=str(
            payload.get("query_expansion_api_base", defaults.query_expansion_api_base)
        ),
        query_expansion_api_key=str(
            payload.get("query_expansion_api_key", defaults.query_expansion_api_key)
        ),
        reranker_model=str(payload.get("reranker_model", defaults.reranker_model)),
        reranker_api_base=str(payload.get("reranker_api_base", defaults.reranker_api_base)),
        reranker_api_key=str(payload.get("reranker_api_key", defaults.reranker_api_key)),
        reranker_endpoint_path=str(
            payload.get("reranker_endpoint_path", defaults.reranker_endpoint_path)
        ),
        reranker_request_timeout=float(
            payload.get("reranker_request_timeout", defaults.reranker_request_timeout)
        ),
    )


def _load_route(name: str, payload: dict, agent_defaults: "AgentConfig") -> RouteConfig:
    sc_default = SelfConsistencyConfig(num_samples=0)  # 0 means "fall back to agent-level"
    sc = _load_self_consistency(payload.get("self_consistency", {}) or {}, sc_default)
    ma = _load_multi_agent(payload.get("multi_agent", {}) or {}, agent_defaults.multi_agent)
    rag = _load_rag(payload.get("rag", {}) or {}, RagConfig())
    semantic_payload = payload.get("semantic_consistency", {}) or {}
    raw_max_tokens = payload.get("max_tokens")
    if raw_max_tokens is None or (isinstance(raw_max_tokens, str) and not raw_max_tokens.strip()):
        max_tokens = agent_defaults.max_tokens
    else:
        max_tokens = int(raw_max_tokens)
    return RouteConfig(
        name=name,
        kind=_normalized_route_kind(payload.get("kind", "react")),
        model=str(payload.get("model", agent_defaults.model)),
        api_base=str(payload.get("api_base", agent_defaults.api_base)),
        api_key=str(payload.get("api_key", agent_defaults.api_key)),
        temperature=float(payload.get("temperature", agent_defaults.temperature)),
        max_tokens=max_tokens,
        request_timeout=float(payload.get("request_timeout", agent_defaults.request_timeout)),
        max_retries=int(payload.get("max_retries", -1)),
        retry_backoff_seconds=float(payload.get("retry_backoff_seconds", -1.0)),
        max_steps=int(payload.get("max_steps", agent_defaults.max_steps)),
        verification_rounds=int(payload.get("verification_rounds", 1)),
        tablellm_max_table_rows=int(payload.get("codegen_max_table_rows", payload.get("tablellm_max_table_rows", 50))),
        tablellm_max_input_chars=int(payload.get("codegen_max_input_chars", payload.get("tablellm_max_input_chars", 12000))),
        tablellm_python_timeout=int(payload.get("codegen_python_timeout", payload.get("tablellm_python_timeout", 30))),
        multi_agent=ma,
        self_consistency=sc,
        rag=rag,
        semantic_consistency_enabled=_bool_value(
            semantic_payload.get(
                "enabled",
                payload.get("semantic_consistency_enabled"),
            ),
            True,
        ),
        semantic_consistency_max_repairs=int(
            semantic_payload.get(
                "max_repairs",
                payload.get("semantic_consistency_max_repairs", 3),
            )
        ),
    )


def _load_cross_model_verifier(payload: dict, *, fallback_name: str) -> CrossModelVerifierSpec:
    return CrossModelVerifierSpec(
        name=str(payload.get("name", fallback_name)).strip() or fallback_name,
        model=str(payload.get("model", "")).strip(),
        api_base=str(payload.get("api_base", "")).strip(),
        api_key=str(payload.get("api_key", "")).strip(),
        temperature=(
            float(payload["temperature"]) if payload.get("temperature") is not None else None
        ),
        max_tokens=(
            int(payload["max_tokens"]) if payload.get("max_tokens") is not None else None
        ),
        request_timeout=float(payload.get("request_timeout", -1.0)),
        max_retries=int(payload.get("max_retries", -1)),
    )


def _load_cross_model_verify(payload: dict) -> CrossModelVerifyConfig:
    if not payload:
        return CrossModelVerifyConfig()
    raw_apply = payload.get("apply_to_routes") or []
    if not isinstance(raw_apply, list):
        raise ValueError("agent.router.cross_model_verify.apply_to_routes must be a list.")
    raw_verifiers = payload.get("verifiers") or []
    if not isinstance(raw_verifiers, list):
        raise ValueError("agent.router.cross_model_verify.verifiers must be a list.")
    verifiers = tuple(
        _load_cross_model_verifier(item or {}, fallback_name=f"verifier_{idx}")
        for idx, item in enumerate(raw_verifiers)
    )
    return CrossModelVerifyConfig(
        enabled=_bool_value(payload.get("enabled"), bool(verifiers)),
        apply_to_routes=tuple(str(name).strip() for name in raw_apply if str(name).strip()),
        verifiers=verifiers,
        min_agreement=int(payload.get("min_agreement", 2)),
        parallel=_bool_value(payload.get("parallel"), True),
    )


def _load_budget(payload: dict, defaults: BudgetConfig) -> BudgetConfig:
    if not payload:
        return defaults
    return BudgetConfig(
        enabled=_bool_value(payload.get("enabled"), defaults.enabled),
        max_llm_calls=int(payload.get("max_llm_calls", defaults.max_llm_calls)),
        max_tool_calls=int(payload.get("max_tool_calls", defaults.max_tool_calls)),
        max_seconds=float(payload.get("max_seconds", defaults.max_seconds)),
        max_local_repairs=int(payload.get("max_local_repairs", defaults.max_local_repairs)),
        max_reasoner_repairs=int(
            payload.get("max_reasoner_repairs", defaults.max_reasoner_repairs)
        ),
        max_multiagent_fallbacks=int(
            payload.get("max_multiagent_fallbacks", defaults.max_multiagent_fallbacks)
        ),
    )


def _load_reasoner_repair(
    payload: dict,
    defaults: ReasonerRepairConfig,
) -> ReasonerRepairConfig:
    if not payload:
        return defaults
    raw_max_tokens = payload.get("max_tokens", defaults.max_tokens)
    max_tokens = None if raw_max_tokens is None else int(raw_max_tokens)
    return ReasonerRepairConfig(
        enabled=_bool_value(payload.get("enabled"), defaults.enabled),
        model=str(payload.get("model", defaults.model)).strip(),
        api_base=str(payload.get("api_base", defaults.api_base)).strip(),
        api_key=str(payload.get("api_key", defaults.api_key)).strip(),
        temperature=float(payload.get("temperature", defaults.temperature)),
        max_tokens=max_tokens,
        request_timeout=float(payload.get("request_timeout", defaults.request_timeout)),
        max_retries=int(payload.get("max_retries", defaults.max_retries)),
    )


def _load_router(payload: dict, agent_defaults: "AgentConfig") -> RouterConfig:
    if not payload:
        return RouterConfig()
    routes_payload = payload.get("routes") or {}
    if not isinstance(routes_payload, dict):
        raise ValueError("agent.router.routes must be a mapping of route names to configs.")
    routes = {
        name: _load_route(name, route_payload or {}, agent_defaults)
        for name, route_payload in routes_payload.items()
    }
    diff_routing_payload = payload.get("difficulty_routing") or {}
    if not isinstance(diff_routing_payload, dict):
        raise ValueError("agent.router.difficulty_routing must be a mapping.")
    difficulty_routing = {
        str(key).strip().lower(): str(value).strip()
        for key, value in diff_routing_payload.items()
    }
    task_type_routing_payload = payload.get("task_type_routing") or {}
    if not isinstance(task_type_routing_payload, dict):
        raise ValueError("agent.router.task_type_routing must be a mapping.")
    task_type_routing = {
        str(key).strip().lower(): str(value).strip()
        for key, value in task_type_routing_payload.items()
    }
    default_route = str(payload.get("default_route", "medium")).strip()
    enabled = _bool_value(payload.get("enabled"), bool(routes))

    raw_cascade_order = payload.get("cascade_order")
    if raw_cascade_order is None:
        cascade_order: tuple[str, ...] = ("easy", "medium", "hard", "extreme")
    elif isinstance(raw_cascade_order, list):
        cascade_order = tuple(str(item).strip() for item in raw_cascade_order)
    else:
        raise ValueError("agent.router.cascade_order must be a list of route names.")

    return RouterConfig(
        enabled=enabled,
        routes=routes,
        task_type_routing=task_type_routing,
        difficulty_routing=difficulty_routing,
        default_route=default_route,
        estimate_when_missing=_bool_value(payload.get("estimate_when_missing"), True),
        cascade_on_failure=_bool_value(payload.get("cascade_on_failure"), True),
        cascade_order=cascade_order,
        cascade_max_extra_attempts=int(payload.get("cascade_max_extra_attempts", 1)),
        cross_model_verify=_load_cross_model_verify(payload.get("cross_model_verify") or {}),
    )


def _load_multi_agent(payload: dict, defaults: MultiAgentConfig) -> MultiAgentConfig:
    if not payload:
        return defaults
    return MultiAgentConfig(
        enabled=_bool_value(payload.get("enabled"), defaults.enabled),
        planner_temperature=_opt_float(payload.get("planner_temperature"), defaults.planner_temperature),
        specialist_max_steps=int(payload.get("specialist_max_steps", defaults.specialist_max_steps)),
        specialist_temperature=_opt_float(
            payload.get("specialist_temperature"), defaults.specialist_temperature
        ),
        synthesizer_max_steps=int(payload.get("synthesizer_max_steps", defaults.synthesizer_max_steps)),
        synthesizer_temperature=_opt_float(
            payload.get("synthesizer_temperature"), defaults.synthesizer_temperature
        ),
        max_specialist_workers=int(payload.get("max_specialist_workers", defaults.max_specialist_workers)),
        enable_iterative_refinement=_bool_value(
            payload.get("enable_iterative_refinement"), defaults.enable_iterative_refinement
        ),
    )


def load_app_config(config_path: Path) -> AppConfig:
    payload = yaml.safe_load(config_path.read_text()) or {}
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()
    scoring_defaults = ScoringConfig()

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    run_payload = payload.get("run", {})
    scoring_payload = payload.get("scoring", {})

    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
        gold_root=_path_value(dataset_payload.get("gold_root"), dataset_defaults.gold_root),
    )

    self_consistency = _load_self_consistency(
        agent_payload.get("self_consistency", {}) or {},
        agent_defaults.self_consistency,
    )
    multi_agent_config = _load_multi_agent(
        agent_payload.get("multi_agent", {}) or {},
        agent_defaults.multi_agent,
    )
    budget_config = _load_budget(
        agent_payload.get("budget", {}) or {},
        agent_defaults.budget,
    )
    reasoner_repair_config = _load_reasoner_repair(
        agent_payload.get("reasoner_repair", {}) or {},
        agent_defaults.reasoner_repair,
    )

    raw_max_tokens = agent_payload.get("max_tokens")
    if raw_max_tokens is None or (isinstance(raw_max_tokens, str) and not raw_max_tokens.strip()):
        max_tokens = agent_defaults.max_tokens
    else:
        max_tokens = int(raw_max_tokens)

    agent_without_router = AgentConfig(
        provider=str(agent_payload.get("provider", agent_defaults.provider)),
        mode=_normalized_agent_mode(agent_payload.get("mode", agent_defaults.mode)),
        model=str(agent_payload.get("model", agent_defaults.model)),
        api_base=str(agent_payload.get("api_base", agent_defaults.api_base)),
        api_key=str(agent_payload.get("api_key", agent_defaults.api_key)),
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
        max_tokens=max_tokens,
        request_timeout=float(agent_payload.get("request_timeout", agent_defaults.request_timeout)),
        max_retries=int(agent_payload.get("max_retries", agent_defaults.max_retries)),
        retry_backoff_seconds=float(
            agent_payload.get("retry_backoff_seconds", agent_defaults.retry_backoff_seconds)
        ),
        retry_backoff_max_seconds=float(
            agent_payload.get("retry_backoff_max_seconds", agent_defaults.retry_backoff_max_seconds)
        ),
        cache_tool_results=_bool_value(
            agent_payload.get("cache_tool_results"), agent_defaults.cache_tool_results
        ),
        self_consistency=self_consistency,
        multi_agent=multi_agent_config,
        budget=budget_config,
        reasoner_repair=reasoner_repair_config,
    )
    router_config = _load_router(agent_payload.get("router", {}) or {}, agent_without_router)
    agent_config = AgentConfig(
        provider=agent_without_router.provider,
        mode=agent_without_router.mode,
        model=agent_without_router.model,
        api_base=agent_without_router.api_base,
        api_key=agent_without_router.api_key,
        max_steps=agent_without_router.max_steps,
        temperature=agent_without_router.temperature,
        max_tokens=agent_without_router.max_tokens,
        request_timeout=agent_without_router.request_timeout,
        max_retries=agent_without_router.max_retries,
        retry_backoff_seconds=agent_without_router.retry_backoff_seconds,
        retry_backoff_max_seconds=agent_without_router.retry_backoff_max_seconds,
        cache_tool_results=agent_without_router.cache_tool_results,
        self_consistency=agent_without_router.self_consistency,
        multi_agent=agent_without_router.multi_agent,
        budget=agent_without_router.budget,
        reasoner_repair=agent_without_router.reasoner_repair,
        router=router_config,
    )

    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)),
    )

    scoring_config = ScoringConfig(
        redundancy_lambda=float(scoring_payload.get("redundancy_lambda", scoring_defaults.redundancy_lambda)),
        numeric_tolerance=float(scoring_payload.get("numeric_tolerance", scoring_defaults.numeric_tolerance)),
        case_insensitive=_bool_value(
            scoring_payload.get("case_insensitive"), scoring_defaults.case_insensitive
        ),
        strip_whitespace=_bool_value(
            scoring_payload.get("strip_whitespace"), scoring_defaults.strip_whitespace
        ),
    )

    return AppConfig(
        dataset=dataset_config,
        agent=agent_config,
        run=run_config,
        scoring=scoring_config,
    )
