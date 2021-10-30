from functools import partial

from gym import spaces

from constants import *
from luxai2021.env.agent import AgentWithModel
from luxai2021.game.actions import *
from luxai2021.game.city import CityTile
from luxai2021.game.constants import Constants
from luxai2021.game.game import Game
from luxai2021.game.unit import Worker


def calc_distance(x1, y1, x2, y2):
    a = abs(x2 - x1)
    b = abs(y2 - y1)
    return a + b


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

    # Research
    research = game.state["teamStates"][team]["researchPoints"]
    idx = GAME_STATE_IDX_DICT['coal_research_progress']
    game_state_vec[idx] = min(research / RESERACH_FOR_COAL, 1)
    idx = GAME_STATE_IDX_DICT['uranium_research_progress']
    game_state_vec[idx] = min(research / MAX_RESEARCH, 1)

    return game_state_vec


def get_unit_vec(game, unit):
    vec = np.zeros(UNIT_LEN)

    # Unit type, inventory, and position
    if isinstance(unit, CityTile):
        unit_type = 'city'
    elif isinstance(unit, Worker):
        unit_type = 'worker'
    else:
        unit_type = 'cart'

    x, y = unit.pos.x, unit.pos.y

    vec[UNIT_IDX_DICT[unit_type]] = 1
    vec[UNIT_IDX_DICT['inventory']] = get_cargo(game, unit, unit_type)
    vec[UNIT_IDX_DICT['team']] = unit.team
    vec[UNIT_IDX_DICT['x']] = x / game.map.width
    vec[UNIT_IDX_DICT['y']] = y / game.map.height

    return vec


def get_team_unit_vec(game, units, controlled_unit_vec):
    vec_list = np.zeros((len(units), UNIT_LEN))

    for idx, unit in enumerate(units):
        vec_list[idx] = get_unit_vec(game, unit)

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

    if len(vec_list) == 0:
        vec_list = np.ones((NUM_OBSERVATIONS, UNIT_LEN)) * -1
    elif len(vec_list) < NUM_OBSERVATIONS:
        vec_list = np.concatenate((vec_list, np.ones((NUM_OBSERVATIONS - len(vec_list), UNIT_LEN)) * -1))

    return vec_list.flatten()


def get_team_city_vec(game, cities, controlled_unit_vec):
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

        vec_list[idx] = get_unit_vec(game, cell.city_tile)

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

    if len(vec_list) == 0:
        vec_list = np.ones((NUM_OBSERVATIONS, UNIT_LEN)) * -1
    elif len(vec_list) < NUM_OBSERVATIONS:
        vec_list = np.concatenate((vec_list, np.ones((NUM_OBSERVATIONS - len(vec_list), UNIT_LEN)) * -1))

    return vec_list.flatten()


def get_resource_vec(game, controlled_unit_vec):
    resource_list = np.zeros((len(game.map.resources), RESOURCE_LEN))

    for idx, cell in enumerate(game.map.resources):
        resource_vec = np.zeros(RESOURCE_LEN)

        resource_vec[RESOURCE_IDX_DICT[cell.resource.type]] = 1
        resource_vec[RESOURCE_IDX_DICT['amount']] = get_cargo(game, cell, cell.resource.type)
        resource_vec[RESOURCE_IDX_DICT['x']] = cell.pos.x / game.map.width
        resource_vec[RESOURCE_IDX_DICT['y']] = cell.pos.y / game.map.height

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
        vec_list = np.ones((NUM_RESOURCE_OBSERVATIONS, RESOURCE_LEN)) * -1
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

            partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.CART),
            partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.WORKER),

            SpawnCityAction,
            PillageAction,
        ]

        self.city_actions = [
            SpawnWorkerAction,
            SpawnCartAction,
            ResearchAction,
        ]

        self.action_space = spaces.Discrete(max(len(self.unit_actions), len(self.city_actions)))

        # Initialize observations
        self.observation_shape = OBSERVATION_SHAPE
        self.observation_space = spaces.Box(low=-1, high=1, shape=self.observation_shape, dtype=np.float16)

        self.object_nodes = {}
        self.observation = np.zeros(OBSERVATION_SHAPE[0])

    def game_start(self, game):
        self.last_unit_count = STARTING_UNITS
        self.last_city_tile_count = STARTING_CITIES
        self.last_fuel_deposited = 0

        self.coal_is_researched = False
        self.uranium_is_researched = False
        self.last_wood = 0
        self.last_coal = 0
        self.last_uranium = 0
        self.old_research = 0

        self.total_amount_wood = 0
        for cell in game.map.resources:
            if cell.resource.type == Constants.RESOURCE_TYPES.WOOD:
                self.total_amount_wood += get_cargo(game, cell, Constants.RESOURCE_TYPES.WOOD)

        self.avg_team_reward_dict = {self.team: [], (self.team + 1) % 2: []}

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
        controlled_unit_vec = get_unit_vec(game, unit=controlled_unit)
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
            game_state_vec, controlled_unit_vec,
            unit_vec_dict[player_team], unit_vec_dict[(player_team + 1) % 2],
            city_vec_dict[player_team], city_vec_dict[(player_team + 1) % 2],
            resource_vec
        ])

        return obs

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

        """
        Prior checks
            [] Game error
            [] Game not start or end
            [] Game Over
        """

        if not is_new_turn and not game_over:
            # Only apply rewards at the start of each turn or at game end
            return 0.0

        if game_errored:
            # Game environment step failed, assign a game lost reward to not incentivise this behaviour
            print("Game failed due to error")
            return MIN_REWARD

        if game_over:
            self.is_last_turn = True

            if game.get_winning_team() == self.team:
                return GAME_WIN
            else:
                return GAME_LOSS

        """
        During Game
            [+] city spawned
            [-] city destroyed
            
            [+] unit spawned
            [-] unit destroyed
            
            [+] fuel collected
        """

        # Number of cities spawned or destroyed
        city_count = 0
        city_count_opponent = 0
        city_tile_count = 0
        city_tile_count_opponent = 0
        for city in game.cities.values():
            if city.team == self.team:
                city_count += 1
            else:
                city_count_opponent += 1

            for _ in city.city_cells:
                if city.team == self.team:
                    city_tile_count += 1
                else:
                    city_tile_count_opponent += 1

        city_growth = city_tile_count - self.last_city_tile_count
        self.last_city_tile_count = city_tile_count

        # Number of units spawned or destroyed
        unit_count = len(game.state["teamStates"][self.team]["units"])
        unit_count_opponent = len(game.state["teamStates"][(self.team + 1) % 2]["units"])
        unit_growth = unit_count - self.last_unit_count
        self.last_unit_count = unit_count

        # Amount of fuel deposited
        fuel_deposited = game.stats["teamStats"][self.team]["fuelGenerated"]
        fuel_deposited_growth = fuel_deposited - self.last_fuel_deposited
        self.last_fuel_deposited = fuel_deposited

        # Amount of resource gathered
        resources_collected = game.stats["teamStats"][self.team]["resourcesCollected"]

        wood_gathered = resources_collected['wood'] - self.last_wood
        self.last_wood = resources_collected['wood']

        coal_gathered = resources_collected['coal'] - self.last_coal
        self.last_coal = resources_collected['coal']

        uranium_gathered = resources_collected['uranium'] - self.last_uranium
        self.last_uranium = resources_collected['uranium']

        research_completed = game.state["teamStates"][self.team]["researchPoints"]
        research_growth = research_completed - self.old_research
        self.old_research = research_completed

        """
        End of game
            [] # ally cities - # enemy cities standing
            
            if above == 0
                # ally units - # enemy units
        
        """

        reward = 0.0

        # Research rewards
        if game.state["teamStates"][self.team]["researched"]["coal"] and not self.coal_is_researched:
            self.coal_is_researched = True
            reward += COAL_UNLOCKED

        if game.state["teamStates"][self.team]["researched"]["uranium"] and not self.uranium_is_researched:
            self.uranium_is_researched = True
            reward += URANIUM_UNLOCKED

        if self.coal_is_researched:
            research_growth *= RESEARCH_GOAL_MET_MODIFIER

        # reward += research_growth * RESEARCH_REWARD_MODIFIER * city_tile_count

        def calc_unit_reward(growth, count):
            # City / Unit rewards
            if count > 0:
                decayed_reward = (np.e ** (np.abs(growth) / count)) - 1
                decayed_reward = np.clip(decayed_reward, MIN_REWARD, MAX_REWARD)
                return decayed_reward if growth >= 0 else -decayed_reward

            elif growth != 0:  # Lost all units this turn
                return MIN_REWARD

            return 0

        reward += calc_unit_reward(city_growth, city_tile_count)  # city reward
        reward += calc_unit_reward(unit_growth, unit_count)  # unit reward

        # Resource rewards
        reward += fuel_deposited_growth * FUEL_DEPOSITED_REWARD_MODIFIER
        reward += wood_gathered * WOOD_GATHERED_REWARD_MODIFIER
        reward += coal_gathered * COAL_GATHERED_REWARD_MODIFIER
        reward += uranium_gathered * URANIUM_GATHERED_REWARD_MODIFIER

        # subtract the avg enemy team's reward to prevent positive sum situations
        # self.avg_team_reward_dict[self.team].append(reward)
        # reward -= np.mean(self.avg_team_reward_dict[(self.team + 1) % 2])

        return np.clip(reward, MIN_REWARD, MAX_REWARD)

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
