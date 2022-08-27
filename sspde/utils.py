import json
import os
import re

import numpy as np
from pddlgym.core import PDDLEnv
from pddlgym.structs import State, TypedEntity


def flatten(lst):
    return [x for l_i in lst for x in l_i]


def get_values(obs, name):
    values = []
    for lit in obs:
        if lit.predicate.name == name:
            values.append(lit.variables)
    return values


def get_objects_by_name(objs, name):
    return [obj for obj in objs if str(obj).split(':')[1] == name]


def get_literals_by_name(s, name):
    return frozenset((lit for lit in s if lit.predicate.name == name))


def get_literals_that_start_with(s, pattern):
    return frozenset(
        (lit for lit in s if lit.predicate.name.startswith(pattern)))


def get_coord_from_location_obj(obj):
    return tuple(map(int, obj.split(':')[0][1:-1].split('-')))


def get_coord_from_state(s):
    robot_at_lits = get_values(s.literals, 'robot-at')
    loc_objs = [x[1] for x in robot_at_lits]
    str_coords = loc_objs[0].split(':')[0][1:-1].split('-')

    return tuple(map(int, str_coords))


def create_problem_instance_from_file(dir_path,
                                      domain_path_name,
                                      problem_index=0):
    domain_file = os.path.join(dir_path, 'pddl', f'{domain_path_name}.pddl')
    problem_dir = os.path.join(dir_path, 'pddl', f'{domain_path_name}')

    env = PDDLEnv(domain_file,
                  problem_dir,
                  raise_error_on_invalid_action=True,
                  dynamic_action_space=False)

    env.fix_problem_index(problem_index)
    return env, env.problems[problem_index]


def create_states_from_base_literals(base_state_literals,
                                     state_literals,
                                     problem,
                                     only_literals=False):
    return [
        State(frozenset({*base_state_literals, *literals}),
              frozenset(problem.objects), problem.goal)
        if not only_literals else frozenset({*base_state_literals, *literals})
        for literals in state_literals
    ]


# Text rendering
# ===========================================================================


def tireworld_get_location(obs):
    obs = obs if type(obs) == State else from_literals(obs)
    for lit in obs.literals:
        if lit.predicate.name == 'vehicle-at':
            return lit.variables[0]
    return []


def tireworld_text_render(obs):
    obs = obs if type(obs) == State else from_literals(obs)
    vehicle_location = tireworld_get_location(obs)
    flattire = True
    spare_in_locs = []
    for lit in obs.literals:
        if lit.predicate.name == 'not-flattire':
            flattire = False
        elif lit.predicate.name == 'spare-in':
            spare_in_locs.append(lit.variables[0])
    return f"""
        Vehicle at {vehicle_location}
        Spare tires at {spare_in_locs}
        {"Flat tire" if flattire else ""}
    """


def navigation_get_locations(obs):
    return get_values(obs.literals, 'robot-at')


def river_alt_get_location(obs):
    location = None

    for lit in obs.literals:
        if lit.predicate.name == 'robot-at':
            location = lit.variables[1]
            break

    return location


def river_alt_text_render(obs):
    obs = obs if type(obs) == State else from_literals(obs)
    qualifiers = []
    location = river_alt_get_location(obs)
    for lit in obs.literals:
        if (lit.predicate.name != 'robot-at' and lit.predicate.name != 'conn'
                and lit.variables[0] == location):
            qualifiers.append(lit.predicate.name)
    return f"""
        Robot at {location}
        {f"Qualifiers: {qualifiers}" if len(qualifiers) > 0 else ""}
    """


def get_location_string(loc):
    if loc == []:
        return "deadend"
    elif type(loc) == TypedEntity:
        return loc
    elif type(loc[0]) == TypedEntity:
        return loc[0]
    else:
        return loc[0][0]


def parse_location_string(loc_str):
    if res := re.search(r"\d+-\d+", loc_str):
        return res.group()
    return loc_str


def parse_location(loc):
    return parse_location_string(get_location_string(loc))


def navigation_text_render(obs):
    obs = obs if type(obs) == State else from_literals(obs)
    qualifiers = []
    locations = navigation_get_locations(obs)

    location = "deadend" if locations == [] else locations[0][0]
    for lit in obs.literals:
        if (lit.predicate.name != 'robot-at' and lit.predicate.name != 'conn'
                and lit.variables[0] == location):
            qualifiers.append(lit.predicate.name)
    return f"""
        Robot at {location}
        {f"Qualifiers: {qualifiers}" if len(qualifiers) > 0 else ""}
    """


def expblocks_text_render(obs):
    clear = []
    ontable = []
    on = []
    holding = None
    destroyed_blocks = []
    table_destroyed = None
    for lit in obs.literals:
        if lit.predicate.name == 'clear':
            clear.append(lit.variables[0])
        elif lit.predicate.name == 'on':
            on.append(lit.variables[:2])
        elif lit.predicate.name == 'ontable':
            ontable.append(lit.variables[0])
        elif lit.predicate.name == 'destroyed':
            destroyed_blocks.append(lit.variables[0])
        elif lit.predicate.name == 'holding':
            holding = lit.variables[0]
        elif lit.predicate.name == 'table-destroyed':
            table_destroyed = True
    return f"""
        {"Table destroyed" if table_destroyed else ""}
        {f"Holding {holding}" if holding else "Hand empty"}
        Clear blocks at {clear}
        Blocks on table: {ontable}
        Destroyed blocks: {destroyed_blocks}
        {on}
    """


navigation_is = range(1, 11)
navigation_keys = [f"PDDLEnvNavigation{i}-v0" for i in navigation_is]
text_render_env_functions = {
    "PDDLEnvTireworld-v0": tireworld_text_render,
    "PDDLEnvExplodingblocks-v0": expblocks_text_render,
    "PDDLEnvExplodingblocksTest-v0": expblocks_text_render,
    "PDDLEnvRiver-alt-v0": river_alt_text_render,
    **{k: navigation_text_render
       for k in navigation_keys}
}

get_location_functions = {
    "PDDLEnvTireworld-v0": tireworld_get_location,
    "PDDLEnvRiver-alt-v0": river_alt_get_location,
    **{k: navigation_get_locations
       for k in navigation_keys}
}


def text_render(env, obs):
    if env.spec.id not in text_render_env_functions:
        return ""
    return text_render_env_functions[env.spec.id](obs)


# ===========================================================================
def output(output_filename, data, output_dir=None):
    output_dir = output_dir or "./output"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_file_path = os.path.join(output_dir, output_filename)

    with open(output_file_path, 'w') as fp:
        json.dump(data, fp, indent=2)

    return output_file_path


def from_literals(literals):
    empty_set = frozenset()
    return State(literals, empty_set, empty_set)


def get_succ_states_actions(s, A, succ_states):
    succs = set()
    for a in A:
        for s_ in succ_states[s, a]:
            succs.add(s_)
    return succs


def nparr_to_list(arr: np.ndarray) -> list:
    return [x.item() for x in arr]
