from mdap_small.hanoi import HanoiState, render_state
from mdap_small.models import ModelSpec, RuntimeSpec


def build_step_prompt(
    runtime: RuntimeSpec,
    disks: int,
    step_idx: int,
    state: HanoiState,
    models: list[ModelSpec],
    prev_move: tuple[int, int, int] | None,
) -> str:
    model_names = ", ".join(m.name for m in models)
    leader = runtime.leader_model()
    leader_name = leader.name if leader else "none"
    return (
        "You are a micro-agent in a MAKER/MDAP workflow.\n"
        "Task: output exactly one legal Tower of Hanoi move and next state.\n"
        "No explanation. Use exact two-line format:\n"
        "move = [disk, source, target]\n"
        "next_state = [[...],[...],[...]]\n"
        f"disks={disks}; step={step_idx}; current_state={render_state(state)}\n"
        f"previous_move={prev_move}.\n"
        f"step_parity={'odd' if step_idx % 2 else 'even'}.\n"
        "Strategy (optimal 3-peg policy):\n"
        "1) The smallest disk moves every other step.\n"
        "2) If disk count is odd, smallest disk cycles 0->2->1->0.\n"
        "3) If disk count is even, smallest disk cycles 0->1->2->0.\n"
        "4) On non-smallest-disk turns, perform the only legal move not involving disk 1.\n"
        f"workflow_mode={runtime.workflow_mode}; leader={leader_name}.\n"
        "Each micro-agent handles exactly one simple step.\n"
        "Use the optimal policy for 3 pegs and move one disk.\n"
        f"Interleaved swarm context active with model pool: {model_names}.\n"
        f"moe_enabled={runtime.moe_enabled}; experts={runtime.expert_rotation}.\n"
        f"tools_enabled={runtime.tools_enabled}; "
        "if needed request tool as `tool = powershell: <command>` or "
        "`tool = bash: <command>`.\n"
        "Example output:\n"
        "move = [1, 0, 2]\n"
        "next_state = [[4,3,2],[1],[]]"
    )
