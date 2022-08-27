from pddlgym.core import get_successor_states, InvalidAction
from pddlgym.structs import Literal, Predicate, State

quit_state = State(frozenset({Literal(Predicate("has-quit", 0, []), [])}),
                   None, None)
quit_action = Literal(Predicate("quit", 0, []), [])


def get_all_reachable(s, A, env, reach=None):
    reach = {} if not reach else reach

    reach[s] = {}
    for a in A:
        try:
            succ = get_successor_states(s,
                                        a,
                                        env.domain,
                                        raise_error_on_invalid_action=True,
                                        return_probs=True)
        except InvalidAction:
            succ = {s: 1.0}
        reach[s][a] = {s_: prob for s_, prob in succ.items()}
        for s_ in succ:
            if s_ not in reach:
                reach.update(get_all_reachable(s_, A, env, reach))
    return reach


def traverse_all_reachable(s, A, env, fn):
    return _traverse_all_reachable(s, A, env, fn)


def _traverse_all_reachable(s, A, env, fn, visited=None):
    visited = set() if not visited else visited

    visited.add(s)
    reach_s = {}
    for a in A:
        try:
            succ = get_successor_states(s,
                                        a,
                                        env.domain,
                                        raise_error_on_invalid_action=True,
                                        return_probs=True)
        except InvalidAction:
            succ = {s: 1.0}
        reach_s[a] = {s_: prob for s_, prob in succ.items()}

        for s_ in succ:
            if s_ not in visited:
                visited.update(_traverse_all_reachable(s_, A, env, fn,
                                                       visited))
    fn(s, reach_s)
    return visited


def from_literals(literals):
    empty_set = frozenset()
    return State(literals, empty_set, empty_set)


def get_values_of_literal_by_name(obs, name):
    return [lit.variables for lit in get_literals_by_name(obs, name)]


def get_literals_by_name(s, name):
    return frozenset((lit for lit in s if lit.predicate.name == name))
