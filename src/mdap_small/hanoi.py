from dataclasses import dataclass


Move = tuple[int, int, int]


@dataclass
class HanoiState:
    pegs: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]

    @staticmethod
    def initial(disks: int) -> "HanoiState":
        return HanoiState((tuple(range(disks, 0, -1)), tuple(), tuple()))

    def apply(self, move: Move) -> "HanoiState":
        disk, src, dst = move
        pegs = [list(p) for p in self.pegs]
        if not pegs[src] or pegs[src][-1] != disk:
            raise ValueError("illegal source move")
        if pegs[dst] and pegs[dst][-1] < disk:
            raise ValueError("illegal target move")
        pegs[src].pop()
        pegs[dst].append(disk)
        return HanoiState(tuple(tuple(p) for p in pegs))


def generate_moves(disks: int):
    # Iterative legal sequence generator for 3-peg Tower of Hanoi.
    pegs = [list(range(disks, 0, -1)), [], []]
    total = (1 << disks) - 1
    if disks % 2 == 0:
        pairs = [(0, 1), (0, 2), (1, 2)]
    else:
        pairs = [(0, 2), (0, 1), (1, 2)]

    for step in range(1, total + 1):
        a, b = pairs[(step - 1) % 3]
        move = _legal_between(pegs, a, b)
        yield move


def _legal_between(pegs: list[list[int]], a: int, b: int) -> Move:
    if not pegs[a]:
        disk = pegs[b].pop()
        pegs[a].append(disk)
        return (disk, b, a)
    if not pegs[b]:
        disk = pegs[a].pop()
        pegs[b].append(disk)
        return (disk, a, b)
    if pegs[a][-1] < pegs[b][-1]:
        disk = pegs[a].pop()
        pegs[b].append(disk)
        return (disk, a, b)
    disk = pegs[b].pop()
    pegs[a].append(disk)
    return (disk, b, a)


def render_state(state: HanoiState) -> str:
    return (
        f"peg0={list(state.pegs[0])}; "
        f"peg1={list(state.pegs[1])}; "
        f"peg2={list(state.pegs[2])}"
    )
