from dataclasses import dataclass

from mdap_small.models import ModelSpec, RuntimeSpec


@dataclass
class StepContext:
    step_idx: int
    workflow_mode: str


class ExpertRouter:
    """Simple MoE-style router for selecting a mixed swarm pool."""

    def build_pool(
        self, runtime: RuntimeSpec, base_pool: list[ModelSpec], step_idx: int
    ) -> list[ModelSpec]:
        if not base_pool:
            return []
        if not runtime.moe_enabled:
            return base_pool

        target = len(base_pool)
        by_expert: dict[str, list[ModelSpec]] = {}
        for model in base_pool:
            by_expert.setdefault(model.expert, []).append(model)

        ordered_experts = runtime.expert_rotation or ["general"]
        routed: list[ModelSpec] = []

        leader = next((m for m in base_pool if m.role == "leader"), None)
        if leader:
            routed.append(leader)

        offset = step_idx % max(1, len(ordered_experts))
        rotated = ordered_experts[offset:] + ordered_experts[:offset]

        picks = 0
        while len(routed) < target and picks < target * 4:
            for expert in rotated:
                bucket = by_expert.get(expert, [])
                if not bucket:
                    continue
                pick = bucket[picks % len(bucket)]
                if pick not in routed:
                    routed.append(pick)
                    if len(routed) >= target:
                        break
            picks += 1

        if len(routed) < target:
            for model in base_pool:
                if model not in routed:
                    routed.append(model)
                if len(routed) >= target:
                    break

        return routed
