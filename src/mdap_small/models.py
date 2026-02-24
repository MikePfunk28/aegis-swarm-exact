from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    name: str
    provider: str = "ollama"
    model_id: str
    role: str = "worker"
    enabled: bool = True
    temperature_first: float = 0.0
    temperature_followup: float = 0.1
    max_tokens: int = 750
    expert: str = "general"
    routing_weight: float = 1.0


class RuntimeSpec(BaseModel):
    models: list[ModelSpec] = Field(default_factory=list)
    ahead_k: int = 3
    parallel_votes: int = 3
    workflow_mode: str = "swarm"
    max_agents: int = 15
    min_agents: int = 3
    auto_scale_agents: bool = True
    moe_enabled: bool = True
    expert_rotation: list[str] = Field(
        default_factory=lambda: ["planner", "executor", "verifier", "general"]
    )
    tools_enabled: bool = False
    tool_budget_per_vote: int = 1
    allowed_shell_commands: list[str] = Field(default_factory=list)
    parser_mode: str = "red_flagging"
    red_flag_token_cutoff: int = 750
    graph_memory_enabled: bool = True

    def active_models(self) -> list[ModelSpec]:
        return [m for m in self.models if m.enabled]

    def leader_model(self) -> ModelSpec | None:
        for model in self.active_models():
            if model.role == "leader":
                return model
        active = self.active_models()
        return active[0] if active else None

    def worker_models(self) -> list[ModelSpec]:
        workers = [m for m in self.active_models() if m.role == "worker"]
        return workers if workers else self.active_models()


DEFAULT_MODELS = RuntimeSpec(
    models=[
        ModelSpec(
            name="lfm2_1p2b",
            model_id="LFM2.5-1.2B",
            role="leader",
            expert="planner",
            routing_weight=1.2,
        ),
        ModelSpec(
            name="gemma3_270m_a",
            model_id="gemma3:270m-it",
            role="worker",
            expert="executor",
        ),
        ModelSpec(
            name="gemma3_270m_b",
            model_id="gemma3:270m-it",
            role="worker",
            expert="executor",
        ),
        ModelSpec(
            name="gemma3_270m_c",
            model_id="gemma3:270m-it",
            role="worker",
            expert="executor",
        ),
        ModelSpec(
            name="qwen3_1p7b",
            model_id="qwen3:1.7b",
            role="worker",
            expert="verifier",
        ),
        ModelSpec(
            name="gemma3_1b",
            model_id="gemma3:1b",
            role="worker",
            expert="general",
        ),
        ModelSpec(
            name="nanbeige4p1_3b",
            model_id="nanbeige4.1:3b",
            role="worker",
            expert="verifier",
        ),
        ModelSpec(
            name="ministral3_3b",
            model_id="ministral3:3b",
            role="worker",
            expert="general",
            enabled=False,
        ),
        ModelSpec(
            name="ministral3_7b",
            model_id="ministral3:7b",
            role="worker",
            expert="verifier",
            enabled=False,
        ),
    ]
)
