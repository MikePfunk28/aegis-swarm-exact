from mdap_small.hanoi import HanoiState
from mdap_small.models import ModelSpec, RuntimeSpec


def build_step_prompt(
    runtime: RuntimeSpec,
    disks: int,
    step_idx: int,
    state: HanoiState,
    models: list[ModelSpec],
    prev_move: tuple[int, int, int] | None,
) -> str:
    state_json = [list(state.pegs[0]), list(state.pegs[1]), list(state.pegs[2])]
    return (
        "Return exactly two lines and nothing else.\n"
        "Task: one legal optimal Tower of Hanoi move for 3 pegs.\n"
        "move = [disk, source, target]\n"
        "next_state = [[...],[...],[...]]\n"
        f"disks={disks}; step={step_idx}; current_state={state_json}\n"
        f"previous_move={prev_move}.\n"
        "Rules: source and target in {0,1,2}; source!=target; exactly one disk moved.\n"
        "next_state must contain all disks 1..disks exactly once across three pegs.\n"
        "Each peg must be ordered largest->smallest."
    )
