import time
from functools import partial

from gym import spaces
from gym.spaces import flatten_space, flatten
from collections import deque
from constants import *
from luxai2021.env.agent import AgentWithModel
from luxai2021.game.actions import *
from luxai2021.game.city import CityTile, City
from luxai2021.game.game import Game
from luxai2021.game.unit import Worker, Cart


def get_scaled_pos(game, unit):
    """
    Scales position to be between 0 and 1 based on the game map size
    :param game:
    :param unit:
    :return:
    """

    x = unit.pos.x / game.map.width
    y = unit.pos.y / game.map.width

    return x, y


def get_direction_vec(unit_a, unit_b):
    """
    Returns a one hot encoded vector of the direction from a to b
    :param unit_a: a unit or city_tile
    :param unit_b: a unit or city_tile
    :return: direction vector
    """

    direction_vec = np.zeros(5)
    x_diff = unit_b.pos.x - unit_a.pos.x
    y_diff = unit_b.pos.y - unit_a.pos.y

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
    """
    Padds array with zero to desired_size
    :param array:
    :param desired_size:
    :return:
    """
    if len(array) < NUM_UNIT_OBSERVATIONS:
        array = np.concatenate((array, np.zeros((NUM_UNIT_OBSERVATIONS - len(array), desired_size))))

    return array


def get_cargo(game, unit):
    if isinstance(unit, CityTile):
        c = game.cities[unit.city_id]
        return min(c.fuel / (c.get_light_upkeep() * 200.0), 1.0)

    elif isinstance(unit, (Worker, Cart)):
        return min(unit.get_cargo_space_left() / 100, 1.0)

    else:
        return min(unit.amount / 500, 1.0)


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


def get_unit_vec(game, unit, controlled_unit):
    vec = np.zeros(UNIT_LEN)

    # Unit type, inventory, and position
    if isinstance(unit, CityTile):
        unit_type = 'city'
    elif isinstance(unit, Worker):
        unit_type = 'worker'
    else:
        unit_type = 'cart'

    x, y = get_scaled_pos(game, unit)

    vec[UNIT_IDX_DICT[unit_type]] = 1
    vec[UNIT_IDX_DICT['inventory']] = get_cargo(game, unit)
    vec[UNIT_IDX_DICT['team']] = unit.team
    vec[UNIT_IDX_DICT['x']] = x
    vec[UNIT_IDX_DICT['y']] = y
    vec[UNIT_IDX_DICT['dist_to_controlled']] = unit.pos.distance_to(controlled_unit.pos)
    vec[UNIT_IDX_DICT['angle_to_controlled']] = 0

    return vec


def get_team_vec(game, units, controlled_unit):
    if len(units) == 0:
        return np.zeros((NUM_UNIT_OBSERVATIONS, UNIT_LEN))

    team_vec = np.zeros((len(units), UNIT_LEN))

    # vectorize units
    for idx, unit in enumerate(units):
        if isinstance(unit, City):  # use closest city tile to represent city
            unit = min(
                unit.city_cells,
                key=lambda city_cell: city_cell.pos.distance_to(controlled_unit.pos)
            ).city_tile

        team_vec[idx] = get_unit_vec(game, unit, controlled_unit)

    # sort units by distance to controlled unit
    team_vec = sorted(
        team_vec,
        key=lambda vec: vec[UNIT_IDX_DICT['dist_to_controlled']]
    )

    team_vec = np.array(team_vec[:NUM_UNIT_OBSERVATIONS])

    return format_array(team_vec, UNIT_LEN)


def get_resource_vec(game, controlled_unit):
    if len(game.map.resources) == 0:
        return np.zeros((NUM_RESOURCE_OBSERVATIONS, RESOURCE_LEN))

    resource_list = np.zeros((len(game.map.resources), RESOURCE_LEN))

    for idx, cell in enumerate(game.map.resources):
        resource_vec = np.zeros(RESOURCE_LEN)

        x1, y1 = get_scaled_pos(game, cell)

        resource_vec[RESOURCE_IDX_DICT[cell.resource.type]] = 1
        resource_vec[RESOURCE_IDX_DICT['amount']] = get_cargo(game, cell.resource)
        resource_vec[RESOURCE_IDX_DICT['x']] = x1
        resource_vec[RESOURCE_IDX_DICT['y']] = y1
        resource_vec[RESOURCE_IDX_DICT['dist_to_controlled']] = cell.pos.distance_to(controlled_unit.pos)
        resource_vec[RESOURCE_IDX_DICT['angle_to_controlled']] = 0

        resource_list[idx] = resource_vec

    resource_list = sorted(
        resource_list,
        key=lambda vec: vec[RESOURCE_IDX_DICT['dist_to_controlled']]
    )

    resource_list = np.array(resource_list[:NUM_RESOURCE_OBSERVATIONS])

    if len(resource_list) < NUM_RESOURCE_OBSERVATIONS:
        resource_list = np.concatenate((resource_list, np.zeros((NUM_RESOURCE_OBSERVATIONS - len(resource_list), RESOURCE_LEN))))

    return resource_list


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
        self.observation_space = spaces.dict.Dict({
            "game_state": spaces.Box(low=0, high=1, shape=(GAME_STATE_LEN,)),
            "controlled_unit_vec": spaces.Box(low=0, high=1, shape=(UNIT_LEN,)),
            "closest_team_cities": spaces.Box(low=0, high=1, shape=(NUM_UNIT_OBSERVATIONS, UNIT_LEN)),
            "closest_opponent_cities": spaces.Box(low=0, high=1, shape=(NUM_UNIT_OBSERVATIONS, UNIT_LEN)),
            "closest_team_units": spaces.Box(low=0, high=1, shape=(NUM_UNIT_OBSERVATIONS, UNIT_LEN)),
            "closest_opponent_units": spaces.Box(low=0, high=1, shape=(NUM_UNIT_OBSERVATIONS, UNIT_LEN)),
            "closest_resources": spaces.Box(low=0, high=1, shape=(NUM_RESOURCE_OBSERVATIONS, RESOURCE_LEN))
        })

        self.observation_space = flatten_space(self.observation_space)
        self.observation_vector = deque([np.zeros(self.observation_space.shape[0]) for i in range(4)], maxlen=4)

    def game_start(self, game):
        self.observation_vector = deque([np.zeros(self.observation_space.shape[0]) for i in range(4)], maxlen=4)

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

    def get_observation(self, game: Game, unit, city_tile, controlled_unit_team, is_new_turn: bool):
        obs_dict = {"game_state": get_game_state_vec(game, controlled_unit_team)}

        # Controlled unit
        controlled_unit = unit if unit is not None else city_tile
        obs_dict["controlled_unit_vec"] = get_unit_vec(game, controlled_unit, controlled_unit)

        # Team observations
        for team in TEAMS:
            cities = [city for city in game.cities.values() if city.team == team]
            units = list(game.get_teams_units(team).values())

            if team == controlled_unit_team:
                city_key = "closest_team_cities"
                unit_key = "closest_team_units"
                units.remove(controlled_unit)
            else:
                city_key = "closest_opponent_cities"
                unit_key = "closest_opponent_units"

            obs_dict[city_key] = get_team_vec(game, cities, controlled_unit)
            obs_dict[unit_key] = get_team_vec(game, units, controlled_unit)

        # n nearest resources
        obs_dict["closest_resources"] = get_resource_vec(game, controlled_unit)

        obs = np.concatenate([np.array(elem).flatten() for elem in list(obs_dict.values())])

        # print(obs)
        self.observation_vector.append(obs)

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
        # prio early game getting wood and building cities

        """
        Returns the reward function for this step of the game. Reward should be a
        delta increment to the reward, not the total current reward.

        :param game: Game object for observations
        :param game_over:
        :param is_new_turn:
        :param game_errored:
        :return: reward for this time step
        """

        # if not is_new_turn and not game_over:
        #     # Only apply rewards at the start of each turn or at game end
        #     return 0.0

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
                obs = np.concatenate(self.observation_vector)
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
                        self.handle_city_actions(game, city_tile)
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
