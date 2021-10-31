import time
from functools import partial

from gym import spaces

from constants import *
from luxai2021.env.agent import AgentWithModel
from luxai2021.game.actions import *
from luxai2021.game.city import CityTile
from luxai2021.game.constants import Constants
from luxai2021.game.game import Game
from luxai2021.game.unit import Worker


def get_pos(game, unit):
    x = unit.pos.x / game.map.width
    y = unit.pos.y / game.map.width

    return x, y


def calc_distance(x1, y1, x2, y2):
    a = abs(x2 - x1)
    b = abs(y2 - y1)
    return a + b


def get_direction_vec(x1, y1, x2, y2):
    direction_vec = np.zeros(5)
    x_diff = x2 - x1
    y_diff = y2 - y1

    if x_diff == 0 and y_diff == 0:
        direction_vec[0] = 1
        return direction_vec

    if x_diff > 0:
        direction_vec[1] = 1
    else:
        direction_vec[2] = 1

    if y_diff > 0:
        direction_vec[3] = 1
    else:
        direction_vec[4] = 1

    return direction_vec


def format_array(array, desired_size):
    if len(array) < NUM_OBSERVATIONS:
        array = np.concatenate((array, np.zeros((NUM_OBSERVATIONS - len(array), desired_size))))

    return array.flatten()


def get_cargo(game, unit, unit_type):
    if unit_type == "city":
        c = game.cities[unit.city_id]
        return min(c.fuel / (c.get_light_upkeep() * 200.0), 1.0)
    elif unit_type in RESOURCE_LIST:
        return min(unit.resource.amount / 500, 1.0)
    else:
        return min(unit.get_cargo_space_left() / 100, 1.0)


def get_game_state_vec(game, team):
    game_state_vec = np.zeros(len(GAME_STATE_CATEGORIES))

    # Is Night
    game_state_vec[GAME_STATE_IDX_DICT['is_night']] = game.is_night()

    # Percent of day/night cycle complete
    steps = NUM_STEPS_IN_NIGHT if game.is_night() else NUM_STEPS_IN_DAY
    game_state_vec[GAME_STATE_IDX_DICT['%_of_cycle_passed']] = (game.state['turn'] % steps) / steps

    # Percent of game done
    game_state_vec[GAME_STATE_IDX_DICT['%_of_cycle_passed']] = game.state["turn"] / MAX_DAYS

    # worker cap reached
    game_state_vec[GAME_STATE_IDX_DICT["worker_cap_reached"]] = game.worker_unit_cap_reached(team)

    game_state_vec[GAME_STATE_IDX_DICT["workers_self"]] = len(game.state["teamStates"][team]["units"]) / MAX_WORKERS
    game_state_vec[GAME_STATE_IDX_DICT["workers_opponent"]] = len(game.state["teamStates"][(team + 1) % 2]) / MAX_WORKERS

    city_count = 0
    city_tile_count = 0
    city_count_opponent = 0
    city_tile_count_opponent = 0
    for city in game.cities.values():
        if city.team == team:
            city_count += 1
            city_tile_count += len(city.city_cells)
        else:
            city_count_opponent += 1
            city_tile_count_opponent += len(city.city_cells)

    game_state_vec[GAME_STATE_IDX_DICT["cities_self"]] = city_tile_count / MAX_CITIES
    game_state_vec[GAME_STATE_IDX_DICT["cities_opponent"]] = city_tile_count_opponent / MAX_CITIES

    # Research
    idx = GAME_STATE_IDX_DICT['research_points']
    game_state_vec[idx] = min(game.state["teamStates"][team]["researchPoints"] / MAX_RESEARCH, 1)

    idx = GAME_STATE_IDX_DICT['coal_is_researched']
    game_state_vec[idx] = game.state["teamStates"][team]['researched']['coal']

    idx = GAME_STATE_IDX_DICT['uranium_is_researched']
    game_state_vec[idx] = game.state["teamStates"][team]['researched']['uranium']

    return game_state_vec


def get_unit_vec(game, unit, controlled_unit_vec):
    vec = np.zeros(UNIT_LEN)

    # Unit type, inventory, and position
    if isinstance(unit, CityTile):
        unit_type = 'city'
    elif isinstance(unit, Worker):
        unit_type = 'worker'
    else:
        unit_type = 'cart'

    x1, y1 = get_pos(game, unit)
    if controlled_unit_vec is not None:
        x2, y2 = controlled_unit_vec[UNIT_IDX_DICT['x']], controlled_unit_vec[UNIT_IDX_DICT['y']]
    else:
        x2, y2 = 0, 0

    vec[UNIT_IDX_DICT[unit_type]] = 1
    vec[UNIT_IDX_DICT['inventory']] = get_cargo(game, unit, unit_type)
    vec[UNIT_IDX_DICT['team']] = unit.team
    vec[UNIT_IDX_DICT['x']] = x1
    vec[UNIT_IDX_DICT['y']] = y1
    vec[UNIT_IDX_DICT['angle_to_controlled']] = np.arctan2(y2 - y1, x2 - x1) / (2 * np.pi)
    vec[UNIT_IDX_DICT['dist_to_controlled']] = calc_distance(x1, y1, x2, y2)

    return vec


def get_team_unit_vec(game, units, controlled_unit_vec):
    if len(units) == 0:
        return np.zeros((NUM_OBSERVATIONS, UNIT_LEN)).flatten()

    vec_list = np.zeros((len(units), UNIT_LEN))

    for idx, unit in enumerate(units):
        vec_list[idx] = get_unit_vec(game, unit, controlled_unit_vec)

    vec_list = sorted(
        vec_list,
        key=lambda vec: calc_distance(
            controlled_unit_vec[UNIT_IDX_DICT['x']],
            controlled_unit_vec[UNIT_IDX_DICT['y']],
            vec[UNIT_IDX_DICT['x']],
            vec[UNIT_IDX_DICT['y']]
        )
    )

    # Remove self from array of closest units
    for idx, vec in enumerate(vec_list):  # just in case units are stacked
        if np.array_equal(vec, controlled_unit_vec):
            vec_list = np.delete(vec_list, idx, axis=0)
            break

    vec_list = np.array(vec_list[:NUM_OBSERVATIONS])

    return format_array(vec_list, UNIT_LEN)


def get_team_city_vec(game, cities, controlled_unit_vec):
    if len(cities) == 0:
        return np.zeros((NUM_OBSERVATIONS, UNIT_LEN)).flatten()

    vec_list = np.zeros((len(cities), UNIT_LEN))
    for idx, city in enumerate(cities):
        cell = min(
            city.city_cells,
            key=lambda vec: calc_distance(
                controlled_unit_vec[UNIT_IDX_DICT['x']],
                controlled_unit_vec[UNIT_IDX_DICT['y']],
                vec.pos.x / game.map.width,
                vec.pos.y / game.map.width
            )
        )

        vec_list[idx] = get_unit_vec(game, cell.city_tile, controlled_unit_vec)

    vec_list = sorted(
        vec_list,
        key=lambda vec: calc_distance(
            controlled_unit_vec[UNIT_IDX_DICT['x']],
            controlled_unit_vec[UNIT_IDX_DICT['y']],
            vec[UNIT_IDX_DICT['x']],
            vec[UNIT_IDX_DICT['y']]
        )
    )

    vec_list = np.array(vec_list[:NUM_OBSERVATIONS])

    return format_array(vec_list, UNIT_LEN)


def get_resource_vec(game, controlled_unit_vec):
    if len(game.map.resources) == 0:
        return np.zeros((NUM_RESOURCE_OBSERVATIONS, RESOURCE_LEN)).flatten()

    resource_list = np.zeros((len(game.map.resources), RESOURCE_LEN))

    for idx, cell in enumerate(game.map.resources):
        resource_vec = np.zeros(RESOURCE_LEN)

        x1, y1 = get_pos(game, cell)
        x2, y2 = controlled_unit_vec[UNIT_IDX_DICT['x']], controlled_unit_vec[UNIT_IDX_DICT['y']]

        resource_vec[RESOURCE_IDX_DICT[cell.resource.type]] = 1
        resource_vec[RESOURCE_IDX_DICT['amount']] = get_cargo(game, cell, cell.resource.type)
        resource_vec[RESOURCE_IDX_DICT['x']] = x1
        resource_vec[RESOURCE_IDX_DICT['y']] = y1
        resource_vec[RESOURCE_IDX_DICT['angle_to_controlled']] = np.arctan2(y2 - y1, x2 - x1) / (2 * np.pi)
        resource_vec[RESOURCE_IDX_DICT['dist_to_controlled']] = calc_distance(x1, y1, x2, y2)

        resource_list[idx] = resource_vec

    resource_list = sorted(
        resource_list,
        key=lambda vec: calc_distance(
            controlled_unit_vec[UNIT_IDX_DICT['x']],
            controlled_unit_vec[UNIT_IDX_DICT['y']],
            vec[RESOURCE_IDX_DICT['x']],
            vec[RESOURCE_IDX_DICT['y']]
        )
    )

    vec_list = np.array(resource_list[:NUM_RESOURCE_OBSERVATIONS])

    if len(vec_list) == 0:
        vec_list = np.zeros((NUM_RESOURCE_OBSERVATIONS, RESOURCE_LEN))
    elif len(vec_list) < NUM_RESOURCE_OBSERVATIONS:
        vec_list = np.concatenate((vec_list, np.ones((NUM_RESOURCE_OBSERVATIONS - len(vec_list), RESOURCE_LEN)) * -1))

    return vec_list.flatten()


class LuxAgent(AgentWithModel):
    def __init__(self, mode="train", model=None):
        super().__init__(mode, model)

        # Initialize actions
        self.unit_actions = [
            partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # No Op
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),

            SpawnCityAction,
        ]

        self.action_space = spaces.Discrete(len(self.unit_actions))

        # Initialize observations
        self.observation_shape = OBSERVATION_SHAPE
        self.observation_space = spaces.Box(low=-1, high=1, shape=self.observation_shape, dtype=np.float16)

        self.object_nodes = {}
        self.observation = np.zeros(OBSERVATION_SHAPE[0])

    def game_start(self, game):
        self.last_unit_count = STARTING_UNITS
        self.last_unit_count_opponent = STARTING_UNITS
        
        self.last_city_tile_count = STARTING_CITIES
        self.last_city_tile_count_opponent = STARTING_CITIES

        self.coal_is_researched = False
        self.uranium_is_researched = False

        self.opponent_rewards = []

        starting_resource_dict = {'fuel_deposited': 0, 'last_wood': 0, 'last_coal': 0, 'last_uranium': 0}
        self.last_resource_dict = {
            self.team: starting_resource_dict.copy(),
            (self.team + 1) % 2: starting_resource_dict.copy()
        }

    def get_observation(self, game: Game, unit, city_tile, team, is_new_turn: bool):
        # game state [team, is_night, current_cycle_percent, game_complete_percent,
        # coal_research_progress, uranium_research_progress, team_worker_cap_reached]
        # controlled unit [is_city, is_worker, is_cart, inventory, abs_pos (x, y), team]
        # closest n allied units [is_city, is_worker, is_cart, inventory, abs_pos (x, y), team]
        # closest n enemy units [is_city, is_worker, is_cart, inventory, abs_pos (x, y), team]
        # closest n resources [is_wood, is_coal, is_uranium, amount, abs_pos]

        game_state_vec = get_game_state_vec(game, team)

        # Controlled unit
        controlled_unit = unit if unit is not None else city_tile
        controlled_unit_vec = get_unit_vec(game, unit=controlled_unit, controlled_unit_vec=None)
        player_team = team

        # n nearest units
        unit_vec_dict = {player_team: None, (player_team + 1) % 2: None}
        for team in TEAMS:
            units = game.get_teams_units(team).values()
            team_vec = get_team_unit_vec(game, units, controlled_unit_vec)
            unit_vec_dict[team] = team_vec

        # n nearest cities
        city_vec_dict = {player_team: None, (player_team + 1) % 2: None}
        for team in TEAMS:
            cities = [city for city in game.cities.values() if city.team == team]
            city_vec_dict[team] = get_team_city_vec(game, cities, controlled_unit_vec)

        # n nearest resources
        resource_vec = get_resource_vec(game, controlled_unit_vec)

        obs = np.concatenate([
            game_state_vec,
            controlled_unit_vec,
            unit_vec_dict[player_team], unit_vec_dict[(player_team + 1) % 2],
            city_vec_dict[player_team], city_vec_dict[(player_team + 1) % 2],
            resource_vec
        ])

        return obs
    
    def calculate_unit_reward(self, game):
        reward = 0

        # Number of cities spawned or destroyed
        city_count = 0
        city_tile_count = 0
        for city in game.cities.values():
            if city.team == self.team:
                city_count += 1
                city_tile_count += len(city.city_cells)

        city_growth = city_tile_count - self.last_city_tile_count
        self.last_city_tile_count = city_tile_count
        # city reward
        reward += city_growth * CITY_MADE

        # Number of units spawned or destroyed
        # unit_count = len(game.state["teamStates"][self.team]["units"])
        # unit_growth = unit_count - self.last_unit_count
        # self.last_unit_count = unit_count
        # Unit reward
        # reward += unit_growth * UNIT_MADE

        return reward
    
    def calculate_resource_reward(self, game):
        reward = 0
        opponent_reward = 0
        
        for team in TEAMS:
            # Amount of fuel deposited
            fuel_deposited = game.stats["teamStats"][team]["fuelGenerated"]
            fuel_deposited_growth = fuel_deposited - self.last_resource_dict[team]['fuel_deposited']
            self.last_resource_dict[team]['fuel_deposited'] = fuel_deposited
    
            # Amount of resource gathered
            resources_collected = game.stats["teamStats"][team]["resourcesCollected"]
    
            wood_gathered = resources_collected['wood'] - self.last_resource_dict[team]['last_wood']
            self.last_resource_dict[team]['last_wood'] = resources_collected['wood']
    
            coal_gathered = resources_collected['coal'] - self.last_resource_dict[team]['last_coal']
            self.last_resource_dict[team]['last_coal'] = resources_collected['coal']
    
            uranium_gathered = resources_collected['uranium'] - self.last_resource_dict[team]['last_uranium']
            self.last_resource_dict[team]['last_uranium'] = resources_collected['uranium']
        
            # Resource rewards
            team_reward = 0
            team_reward += fuel_deposited_growth * FUEL_DEPOSITED_REWARD_MODIFIER
            team_reward += wood_gathered * WOOD_GATHERED_REWARD_MODIFIER
            team_reward += coal_gathered * COAL_GATHERED_REWARD_MODIFIER
            team_reward += uranium_gathered * URANIUM_GATHERED_REWARD_MODIFIER
            
            if team == self.team:
                reward = team_reward
            else:
                opponent_reward = team_reward
        
        return reward, opponent_reward

    def get_reward(self, game, game_over: bool, is_new_turn: bool, game_errored: bool) -> float:
        """
        Returns the reward function for this step of the game. Reward should be a
        delta increment to the reward, not the total current reward.

        :param game: Game object for observations
        :param game_over:
        :param is_new_turn:
        :param game_errored:
        :return: reward for this time step
        """

        if not is_new_turn and not game_over:
            # Only apply rewards at the start of each turn or at game end
            return 0.0

        if game_errored:
            # Game environment step failed, assign a game lost reward to not incentivise this behaviour
            print("Game failed due to error")
            return -GAME_WIN

        if game_over:
            self.is_last_turn = True

            if game.get_winning_team() == self.team:
                return GAME_WIN

        reward = 0.0

        unit_reward = self.calculate_unit_reward(game)
        fuel_reward, fuel_reward_opponent = self.calculate_resource_reward(game)

        # Research rewards
        if game.state["teamStates"][self.team]["researched"]["coal"] and not self.coal_is_researched:
            self.coal_is_researched = True
            reward += COAL_UNLOCKED * (game.state["turn"] / MAX_DAYS)

        if game.state["teamStates"][self.team]["researched"]["uranium"] and not self.uranium_is_researched:
            self.uranium_is_researched = True
            reward += URANIUM_UNLOCKED * (game.state["turn"] / MAX_DAYS)

        self.opponent_rewards.append(fuel_reward_opponent)
        reward += unit_reward + fuel_reward
        reward -= np.mean(self.opponent_rewards)

        return reward

    def take_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            actionCode: Index of action to take into the action array.
        """
        action = self.action_code_to_action(action_code, game, unit, city_tile, team)
        self.match_controller.take_action(action)

    def action_code_to_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            action_code: Index of action to take into the action array.
        Returns: An action.
        """
        # Map action_code index into to a constructed Action object
        try:
            x = None
            y = None
            if city_tile is not None:
                x = city_tile.pos.x
                y = city_tile.pos.y
            elif unit is not None:
                x = unit.pos.x
                y = unit.pos.y

            if city_tile is not None:
                action = self.city_actions[action_code % len(self.city_actions)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )
            else:
                action = self.unit_actions[action_code % len(self.unit_actions)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )

            return action
        except Exception as e:
            # Not a valid action
            print(e)
            return None

    def process_turn(self, game, team):
        """
        Decides on a set of actions for the current turn. Not used in training, only inference. Generally
        don't modify this part of the code.
        Returns: Array of actions to perform.
        """
        start_time = time.time()
        actions = []
        new_turn = True

        # Inference the model per-unit
        units = game.state["teamStates"][team]["units"].values()
        for unit in units:
            if unit.can_act():
                obs = self.get_observation(game, unit, None, unit.team, new_turn)
                # IMPORTANT: You can change deterministic=True to disable randomness in model inference. Generally,
                # I've found the agents get stuck sometimes if they are fully deterministic.
                action_code, _states = self.model.predict(obs, deterministic=False)
                if action_code is not None:
                    actions.append(
                        self.action_code_to_action(action_code, game=game, unit=unit, city_tile=None, team=unit.team))
                new_turn = False

        # Inference the model per-city
        cities = game.cities.values()
        for city in cities:
            if city.team == team:
                for cell in city.city_cells:
                    city_tile = cell.city_tile
                    if city_tile.can_act():
                        obs = self.get_observation(game, None, city_tile, city.team, new_turn)
                        # IMPORTANT: You can change deterministic=True to disable randomness in model inference. Generally,
                        # I've found the agents get stuck sometimes if they are fully deterministic.
                        action_code, _states = self.model.predict(obs, deterministic=False)
                        if action_code is not None:
                            actions.append(
                                self.action_code_to_action(action_code, game=game, unit=None, city_tile=city_tile,
                                                           team=city.team))
                        new_turn = False

        time_taken = time.time() - start_time
        if time_taken > 0.5:  # Warn if larger than 0.5 seconds.
            print("WARNING: Inference took %.3f seconds for computing actions. Limit is 1 second." % time_taken)

        return actions

    def handle_city_actions(self, game, city_tile):
        if not game.worker_unit_cap_reached(self.team):
            self.match_controller.take_action(SpawnWorkerAction(
                    game=game,
                    unit_id=None,
                    unit=None,
                    city_id=city_tile.city_id,
                    citytile=city_tile,
                    team=self.team,
                    x=city_tile.pos.x,
                    y=city_tile.pos.y
                ))
        else:
            self.match_controller.take_action(ResearchAction(
                game=game,
                unit_id=None,
                unit=None,
                city_id=city_tile.city_id,
                citytile=city_tile,
                team=self.team,
                x=city_tile.pos.x,
                y=city_tile.pos.y
            ))

