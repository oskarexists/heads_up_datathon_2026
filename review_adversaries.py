"""Simple legal adversaries for the duplicate-match review.

Action names are the engine's: FOLD, CHECK_CALL, RAISE_HALF, RAISE_POT.
RAISE_HALF is the smallest raise the engine offers (half pot, raised to the
minimum when that is larger, or all-in when the stack is shorter).
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

from evaluator import category_of, score_five
from game import Action, AgentConfig, HandResult, Observation

THREE_OF_A_KIND = 4
TWO_PAIR = 2
ONE_PAIR = 1


def _must(obs: Observation, action: Action) -> Action:
    if action in obs.legal_actions:
        return action
    if Action.CHECK_CALL in obs.legal_actions:
        return Action.CHECK_CALL
    return obs.legal_actions[0]


class Maniac:
    """Always pot-raise. Check/call only when a raise is illegal. Never folds."""

    name = "maniac"

    def new_half(self, config: AgentConfig, memory: dict) -> None:
        return None

    def choose_action(self, obs: Observation, memory: dict) -> Action:
        return _must(obs, Action.RAISE_POT)

    def on_hand_end(self, result: HandResult, memory: dict) -> None:
        return None


class CallingStation:
    """Never folds and never raises."""

    name = "calling_station"

    def new_half(self, config: AgentConfig, memory: dict) -> None:
        return None

    def choose_action(self, obs: Observation, memory: dict) -> Action:
        return _must(obs, Action.CHECK_CALL)

    def on_hand_end(self, result: HandResult, memory: dict) -> None:
        return None


class MinRaiseProber:
    """Bets or raises the legal half-pot size whenever that action exists."""

    name = "min_raise"

    def new_half(self, config: AgentConfig, memory: dict) -> None:
        return None

    def choose_action(self, obs: Observation, memory: dict) -> Action:
        return _must(obs, Action.RAISE_HALF)

    def on_hand_end(self, result: HandResult, memory: dict) -> None:
        return None


def _made_category(hole: tuple[int, ...], board: tuple[int, ...]) -> int:
    """Best Omaha category on this board, or -1 preflop. Uses score_five, not the bot."""
    if len(board) < 3:
        return -1
    best = 0
    for hp in combinations(hole, 2):
        for bp in combinations(board, 3):
            best = max(best, score_five(hp + bp))
    return category_of(best)


def _premium_preflop(hole: tuple[int, ...]) -> bool:
    """Queens-or-better pair, two pair, or trips in the five hole cards."""
    counts = Counter(card >> 2 for card in hole)
    pairs = [rank for rank, n in counts.items() if n >= 2]
    return any(n >= 3 for n in counts.values()) or len(pairs) >= 2 or (pairs and max(pairs) >= 6)


class Exploiter:
    """Probe with the half-pot raise, then overbet if they fold and value-bet if they call.

    Fold rate is counted once per hand in which we raised, from the hand result:
    a fold is a win without showdown; anything else continued. That is the same
    information on_hand_end actually receives.
    """

    name = "exploiter"

    def new_half(self, config: AgentConfig, memory: dict) -> None:
        memory["bets"] = 0
        memory["folds"] = 0
        memory["pending"] = False

    def choose_action(self, obs: Observation, memory: dict) -> Action:
        bets = int(memory.get("bets", 0))
        folds = int(memory.get("folds", 0))
        rate = folds / bets if bets else 0.0
        post = _made_category(obs.hole_cards, obs.board) if obs.board else -1
        strong = _premium_preflop(obs.hole_cards) if obs.street == 0 else post >= THREE_OF_A_KIND
        playable = obs.street == 0 or post >= ONE_PAIR

        if bets >= 30 and rate >= 0.45:
            action = _must(obs, Action.RAISE_POT)
        elif bets >= 30 and rate <= 0.22:
            if obs.to_call == 0:
                action = _must(obs, Action.RAISE_POT if strong else Action.CHECK_CALL)
            elif strong or (playable and obs.to_call * 4 <= obs.pot):
                action = Action.CHECK_CALL
            else:
                action = _must(obs, Action.FOLD)
        else:
            if obs.to_call == 0:
                action = _must(obs, Action.RAISE_HALF)
            elif obs.to_call < obs.pot or playable:
                action = Action.CHECK_CALL
            else:
                action = _must(obs, Action.FOLD)

        if action not in obs.legal_actions:
            action = _must(obs, Action.CHECK_CALL)
        memory["pending"] = action in (Action.RAISE_HALF, Action.RAISE_POT)
        return action

    def on_hand_end(self, result: HandResult, memory: dict) -> None:
        if not memory.get("pending"):
            return
        memory["bets"] = int(memory.get("bets", 0)) + 1
        if result.winner == "me" and not result.went_to_showdown:
            memory["folds"] = int(memory.get("folds", 0)) + 1
        memory["pending"] = False


ADVERSARIES = {
    Maniac.name: Maniac,
    CallingStation.name: CallingStation,
    MinRaiseProber.name: MinRaiseProber,
    Exploiter.name: Exploiter,
}
