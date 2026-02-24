import ast
import json
import re

from pydantic import BaseModel


class ParsedMove(BaseModel):
    disk: int
    source: int
    target: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.disk, self.source, self.target)


class ParsedStep(BaseModel):
    move: ParsedMove
    next_state: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]


def parse_move(text: str, parser_mode: str = "red_flagging") -> ParsedMove | None:
    step = parse_step(text, parser_mode=parser_mode)
    if step:
        return step.move

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    raw = text[start : end + 1].replace("'", '"')
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return _parse_move_obj(obj)


def parse_step(text: str, parser_mode: str = "red_flagging") -> ParsedStep | None:
    if parser_mode == "red_flagging":
        return _parse_step_strict(text)
    return _parse_step_repairing(text)


def _parse_step_strict(text: str) -> ParsedStep | None:
    move_line = _extract_key_value(text, "move")
    state_line = _extract_key_value(text, "next_state")
    if move_line is None or state_line is None:
        return None

    move_obj = _parse_literal(move_line)
    state_obj = _parse_literal(state_line)

    move = _parse_move_obj(move_obj)
    next_state = _parse_state_obj(state_obj)
    if not move or not next_state:
        return None
    return ParsedStep(move=move, next_state=next_state)


def _parse_step_repairing(text: str) -> ParsedStep | None:
    move_line = _extract_key_value(text, "move")
    state_line = _extract_key_value(text, "next_state")

    move_obj = _parse_literal(move_line) if move_line else None
    state_obj = _parse_literal(state_line) if state_line else None

    if move_obj is None or state_obj is None:
        obj = _extract_json_obj(text)
        if isinstance(obj, dict):
            move_obj = obj.get("move", obj)
            state_obj = obj.get("next_state")

    move = _parse_move_obj(move_obj)
    next_state = _parse_state_obj(state_obj)
    if not move or not next_state:
        return None
    return ParsedStep(move=move, next_state=next_state)


def has_red_flags(
    text: str,
    max_tokens_approx: int = 750,
    parser_mode: str = "red_flagging",
) -> bool:
    token_est = max(1, len(text) // 4)
    if token_est > max_tokens_approx:
        return True
    if parse_step(text, parser_mode=parser_mode) is None:
        return True
    return False


def _extract_key_value(text: str, key: str) -> str | None:
    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*=\s*(.+)$", text)
    if not match:
        return None
    return match.group(1).strip()


def _parse_literal(raw: str | None):
    if raw is None:
        return None
    try:
        return ast.literal_eval(raw)
    except Exception:
        try:
            return json.loads(raw)
        except Exception:
            return None


def _extract_json_obj(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    raw = text[start : end + 1].replace("'", '"')
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _parse_move_obj(obj) -> ParsedMove | None:
    if isinstance(obj, dict):
        disk = obj.get("disk")
        source = obj.get("source")
        target = obj.get("target")
    elif isinstance(obj, (list, tuple)) and len(obj) == 3:
        disk, source, target = obj
    else:
        return None

    try:
        move = ParsedMove(disk=int(disk), source=int(source), target=int(target))
    except Exception:
        return None

    if move.source == move.target:
        return None
    if move.source not in (0, 1, 2) or move.target not in (0, 1, 2):
        return None
    if move.disk < 1:
        return None
    return move


def _parse_state_obj(
    obj,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
    if not isinstance(obj, (list, tuple)) or len(obj) != 3:
        return None

    pegs: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for peg in obj:
        if not isinstance(peg, (list, tuple)):
            return None
        try:
            disk_peg = tuple(int(x) for x in peg)
        except Exception:
            return None
        if any(x < 1 for x in disk_peg):
            return None
        for i in range(1, len(disk_peg)):
            if disk_peg[i - 1] <= disk_peg[i]:
                return None
        for disk in disk_peg:
            if disk in seen:
                return None
            seen.add(disk)
        pegs.append(disk_peg)

    return (pegs[0], pegs[1], pegs[2])
