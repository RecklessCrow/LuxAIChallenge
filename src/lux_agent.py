from functools import partial

import numpy as np
from gym import spaces
from luxai2021.game.position import Position

from constants import *
from luxai2021.env.agent import AgentWithModel
from luxai2021.game.actions import *
from luxai2021.game.game_constants import GAME_CONSTANTS


class LuxAgent(AgentWithModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize actions
        self.unit_actions = [
            partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # This is the do-nothing action
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),

            # Transfer to nearby cart
            # partial(TransferAction, target_type_restriction=Constants.UNIT_TYPES.CART),
            # Transfer to nearby worker
            # partial(TransferAction, target_type_restriction=Constants.UNIT_TYPES.WORKER),

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
        self.observation_shape = (57,)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float16)

        self.object_nodes = {}

        # Initialize Reward Tracking
        self.last_unit_count = 0
        self.last_city_tile_count = 0
        self.last_fuel_collected = 0

    def get_map_contents(self, game, team):
        # Build a list of object nodes by type for quick distance-searches
        self.object_nodes = {}

        # Add resources
        for cell in game.map.resources:
            if cell.resource.type not in self.object_nodes:
                self.object_nodes[cell.resource.type] = np.array([[cell.pos.x, cell.pos.y]])
            else:
                self.object_nodes[cell.resource.type] = np.concatenate(
                    (self.object_nodes[cell.resource.type], [[cell.pos.x, cell.pos.y]]), axis=0)

        # Add your own and opponent units
        for t in [team, (team + 1) % 2]:
            for u in game.state["teamStates"][team]["units"].values():
                key = str(u.type)
                if t != team:
                    key = str(u.type) + "_opponent"

                if key not in self.object_nodes:
                    self.object_nodes[key] = np.array([[u.pos.x, u.pos.y]])
                else:
                    self.object_nodes[key] = np.concatenate(
                        (self.object_nodes[key], [[u.pos.x, u.pos.y]]), axis=0)

        # Add your own and opponent cities
        for city in game.cities.values():
            for cells in city.city_cells:
                key = "city"
                if city.team != team:
                    key = "city_opponent"

                if key not in self.object_nodes:
                    self.object_nodes[key] = np.array([[cells.pos.x, cells.pos.y]])
                else:
                    self.object_nodes[key] = np.concatenate(
                        (self.object_nodes[key], [[cells.pos.x, cells.pos.y]]), axis=0)

    @staticmethod
    def distance(node: Position, nodes):
        node = (node.x, node.y)
        return np.sum((nodes - node) ** 2, axis=1)

    @staticmethod
    def get_cargo(game, cell, unit_type):

        # cargo_amount = self.get_cargo(game.map.get_cell_by_pos(other_pos), entity_type, city_tile)
        RESOURCE_LIST = [Constants.RESOURCE_TYPES.WOOD, Constants.RESOURCE_TYPES.COAL, Constants.RESOURCE_TYPES.URANIUM]

        if unit_type == "city":
            # City fuel as % of upkeep for 200 turns
            c = game.cities[cell.city_tile.city_id]
            return min(c.fuel / (c.get_light_upkeep() * 200.0), 1.0)

        elif unit_type in RESOURCE_LIST:
            return min(cell.resource.amount / 500, 1.0)

        else:
            # Unit cargo
            return min(next(iter(cell.units.values())).get_cargo_space_left() / 100, 1.0)

    def get_observation(self, game, unit, city_tile, team, is_new_turn: bool):
        """
        Implements getting a observation from the current game for this unit or city
        """

        if is_new_turn:
            self.get_map_contents(game, team)

        """
        Features/Objects in Vector
        
        Turn Identifier  - 3x:
         - 1x Worker
         - 1x Cart
         - 1x City
         """
        turn_identifier = np.zeros(NUM_IDENTIFIERS)
        if unit is not None:
            if unit.type == Constants.UNIT_TYPES.WORKER:
                turn_identifier[0] = 1.0  # Worker
            else:
                turn_identifier[1] = 1.0  # Cart
        if city_tile is not None:
            turn_identifier[2] = 1.0  # CityTile

        """
        Game State - 12x:
         - 1x cargo size
         - 1x is night
         - 1x percent of game done
         - 2x citytile counts [cur player, opponent]
         - 2x worker counts [cur player, opponent]
         - 2x cart counts [cur player, opponent]
         
         - 1x research points [cur player]
         - 1x researched coal [cur player]
         - 1x researched uranium [cur player]
        """

        game_states = np.zeros(NUM_GAME_STATES)

        # 1x cargo size
        if unit is not None:
            game_states[0] = unit.get_cargo_space_left() / GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]

        # 1x is night
        game_states[1] = game.is_night()

        # 1x percent of game done
        game_states[2] = game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]

        # 6x unit counts
        for idx, key in enumerate(["city", str(Constants.UNIT_TYPES.WORKER), str(Constants.UNIT_TYPES.CART)]):
            if key in self.object_nodes:
                game_states[idx * 2 + 3] = len(self.object_nodes[key]) / MAX_UNIT_COUNT
            if (key + "_opponent") in self.object_nodes:
                game_states[idx * 2 + 4] = len(self.object_nodes[(key + "_opponent")]) / MAX_UNIT_COUNT

        # 1x research points
        game_states[9] = game.state["teamStates"][team]["researchPoints"] / MAX_RESEARCH

        # 1x researched coal
        game_states[10] = float(game.state["teamStates"][team]["researched"]["coal"])

        # 1x researched uranium
        game_states[11] = float(game.state["teamStates"][team]["researched"]["uranium"])

        if unit is not None:
            unit_pos = unit.pos
        else:
            unit_pos = city_tile.pos

        """
        Entity Detection - 42x:
        
        Nearest Cart - 3x:
         - 1x angle theta
         - 1x distance
         - 1x amount
         
        Nearest Worker - 3x:
         - 1x angle theta
         - 1x distance
         - 1x amount
        
        Worker with fullest inventory - 3x:
         - 1x angle theta
         - 1x distance
         - 1x amount
        
        Nearest City - 3x:
         - 1x angle theta
         - 1x distance
         - 1x amount fuel
        
        City with least amount of Fuel - 3x:
         - 1x angle theta
         - 1x distance
         - 1x amount fuel
        
        Resources - 27x:
        - 3x per resource 
         - 3x for 3 nearest resource piles
          - 1x angle theta
          - 1x distance
          - 1x amount
         
        """

        types = {
            Constants.RESOURCE_TYPES.WOOD: 3,
            Constants.RESOURCE_TYPES.COAL: 3,
            Constants.RESOURCE_TYPES.URANIUM: 3,
            "city": 1,
            str(Constants.UNIT_TYPES.WORKER): 1,
            str(Constants.UNIT_TYPES.CART): 1
        }

        entity_detection = np.zeros(NUM_RESOURCES)
        entity_idx = 0

        for idx, (entity_type, quantity) in enumerate(types.items()):
            if entity_type in self.object_nodes:

                nodes = self.object_nodes[entity_type]
                sorted_idx = np.argsort(self.distance(unit_pos, nodes))

                if (unit is not None and unit.type == entity_type) or (entity_type == "city" and city_tile is not None):
                    sorted_idx = np.delete(sorted_idx, 0)

                if len(nodes) == 0:
                    continue

                for i in range(quantity):

                    if i + 1 > len(sorted_idx):
                        entity_idx += 3
                        continue

                    node_idx = sorted_idx[i] if len(sorted_idx) > 0 else 0
                    other_pos = nodes[node_idx]
                    other_pos = Position(other_pos[0], other_pos[1])

                    # 1x angle theta
                    angle = np.arctan2(other_pos.y - unit_pos.y, other_pos.x - unit_pos.x)
                    entity_detection[entity_idx] = angle
                    entity_idx += 1

                    # 1x distance
                    distance = np.sqrt((other_pos.x - unit_pos.x)**2 + (other_pos.y - unit_pos.y)**2)
                    entity_detection[entity_idx] = distance
                    entity_idx += 1

                    # 1x amount
                    cell = game.map.get_cell_by_pos(other_pos)
                    cargo_amount = self.get_cargo(game, cell, entity_type)
                    entity_detection[entity_idx] = cargo_amount
                    entity_idx += 1

        # entity_detection = np.concatenate(entity_detection)

        return np.concatenate([turn_identifier, game_states, entity_detection])

    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.
        Args:
            game ([type]): Game.
        """

        # Reset reward tracking
        self.last_unit_count = 0
        self.last_city_tile_count = 0
        self.last_fuel_collected = 0

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
        Before Game
            [-] Game error
            [] Game not start or end
        """

        if not is_new_turn and not game_over:
            # Only apply rewards at the start of each turn or at game end
            return 0.0

        if game_errored:
            # Game environment step failed, assign a game lost reward to not incentivise this behaviour
            print("Game failed due to error")
            return -1.0

        """
        During Game
            [+++] city spawned
            [---] city destroyed
            
            [++] unit spawned
            [--] unit destroyed
            
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

            for cell in city.city_cells:
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

        # Amount of fuel collected
        fuel_collected = game.stats["teamStats"][self.team]["fuelGenerated"]
        fuel_growth = fuel_collected - self.last_fuel_collected
        self.last_fuel_collected = fuel_collected

        """
        End of game
            [] # ally cities - # enemy cities standing
            
            if above == 0
                # ally units - # enemy units
        
        """

        lead_amount = 0
        if game_over:
            self.is_last_turn = True

            lead_amount = city_tile_count - city_count_opponent

            if lead_amount == 0:
                lead_amount = unit_count - unit_count_opponent

            '''
            # game win/loss reward
            if game.get_winning_team() == self.team:
                # Win
            else:
                # Loss
            '''

        reward = 0.0

        reward += city_growth * CITY_REWARD_MODIFIER
        reward += unit_growth * UNIT_REWARD_MODIFIER
        reward += fuel_growth * FUEL_REWARD_MODIFIER
        reward += lead_amount * LEAD_REWARD_MODIFIER

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
