import numpy as np
from gym import spaces

from luxai2021.env.agent import AgentWithModel
from luxai2021.game.actions import *
from functools import partial

from luxai2021.game.game_constants import GAME_CONSTANTS


class LuxAgent(AgentWithModel):
    def __init__(self):
        super().__init__()

        self.unit_actions = [
            partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # This is the do-nothing action
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),

            # Transfer to nearby cart
            partial(TransferAction, target_type_restriction=Constants.UNIT_TYPES.CART),
            # Transfer to nearby worker
            partial(TransferAction, target_type_restriction=Constants.UNIT_TYPES.WORKER),

            SpawnCityAction,
            PillageAction,
        ]

        self.city_actions = [
            SpawnWorkerAction,
            SpawnCartAction,
            ResearchAction,
        ]

        self.action_space = spaces.Discrete(max(len(self.unit_actions), len(self.city_actions)))

        self.observation_shape = 51
        self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float16)

        self.object_nodes = {}

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
                        (self.object_nodes[key],[[cells.pos.x, cells.pos.y]]), axis=0)

    @staticmethod
    def distance(node, nodes):
        return np.sum((nodes - node) ** 2, axis=1)

    @staticmethod
    def get_cargo(cell, unit_type, city_tile):
        if unit_type == "city":
            # City fuel as % of upkeep for 200 turns
            c = cell.cities[city_tile.city_id]
            return min(c.fuel / (c.get_light_upkeep() * 200.0), 1.0)

        elif unit_type in \
                [Constants.RESOURCE_TYPES.WOOD, Constants.RESOURCE_TYPES.COAL, Constants.RESOURCE_TYPES.URANIUM]:
            # Resource amount
            return min(cell.resource.amount / 500, 1.0)

        else:
            # Unit cargo
            return min(next(iter(cell.units.values())).get_cargo_space_left() / 100, 1.0)

    def get_observation(self, game, unit, city_tile, team, is_new_turn):
        """
        Implements getting a observation from the current game for this unit or city
        """

        if is_new_turn:
            self.get_map_contents(game, team)


        """
        Features/Objects in Vector
        
        Turn Identifier
         - 1x Worker
         - 1x Cart
         - 1x City
         """

        turn_identifier = np.zeros(3)
        if unit is not None:
            if unit.type == Constants.UNIT_TYPES.WORKER:
                turn_identifier[0] = 1  # Worker
            else:
                turn_identifier[1] = 1.0  # Cart
        if city_tile is not None:
            turn_identifier[2] = 1.0  # CityTile

        """
        Game State
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

        game_states = np.zeros(12)

        # 1x cargo size
        if unit is not None:
            game_states[0] = unit.get_cargo_space_left() / GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]

        # 1x is night
        game_states[1] = game.is_night()

        # 1x percent of game done
        game_states[2] = game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]

        # 6x unit counts
        max_count = 30
        for idx, key in enumerate(["city", str(Constants.UNIT_TYPES.WORKER), str(Constants.UNIT_TYPES.CART)]):
            if key in self.object_nodes:
                game_states[idx * 2 + 3] = len(self.object_nodes[key]) / max_count
            if (key + "_opponent") in self.object_nodes:
                game_states[idx * 2 + 4] = len(self.object_nodes[(key + "_opponent")]) / max_count

        # 1x research points
        game_states[9] = game.state["teamStates"][team]["researchPoints"] / 200.0

        # 1x researched coal
        game_states[10] = float(game.state["teamStates"][team]["researched"]["coal"])

        # 1x researched uranium
        game_states[11] = float(game.state["teamStates"][team]["researched"]["uranium"])



        if unit is not None:
            unit_pos = unit.pos
        else:
            unit_pos = city_tile.pos

        """
        Entity Detection
        
        Nearest Cart
         - 1x angle theta
         - 1x distance
         - 1x amount
         
         Nearest Worker
         - 1x angle theta
         - 1x distance
         - 1x amount
        
        Worker with fullest inventory
         - 1x angle theta
         - 1x distance
         - 1x amount
        
        Nearest City:
         - 1x angle theta
         - 1x distance
         - 1x amount fuel
        
        City with least amount of Fuel:
         - 1x angle theta
         - 1x distance
         - 1x amount fuel
         
        - 3x per resource 
         - 3x for 3 nearest piles different piles
          - 1x angle theta to nearest wood pile
          - 1x distance to nearest wood pile
          - 1x amount in nearest wood pile
         
        """

        types = {
            Constants.RESOURCE_TYPES.WOOD: 3,
            Constants.RESOURCE_TYPES.COAL: 3,
            Constants.RESOURCE_TYPES.URANIUM: 3,
            "city": 1,
            str(Constants.UNIT_TYPES.WORKER): 1,
            str(Constants.UNIT_TYPES.CART): 1
        }

        entity_detection = []

        for idx, type in enumerate(types.keys()):
            if type in self.object_nodes:

                nodes = self.object_nodes[type]
                sorted_idx = np.argsort(self.distance(unit_pos, nodes))

                if unit.type == type or type == "city" and city_tile is not None:
                    sorted_idx = np.delete(sorted_idx, 0)

                if len(nodes) == 0:
                    continue

                n_closest_units = []
                for i in range(types[type]):
                    other_pos = nodes[sorted_idx[i]]

                    # 1x angle theta
                    angle = np.arctan2(other_pos[1] - unit_pos[1], other_pos[0] - unit_pos[0])

                    # 1x distance
                    distance = np.sqrt(other_pos[0] - unit_pos[0] + other_pos[1] - unit_pos[1])

                    # 1x amount
                    cargo_amount = self.get_cargo(game.map.get_cell_by_pos(other_pos), type, city_tile)

                    n_closest_units.append(np.array([angle, distance, cargo_amount]))

                entity_detection.append(np.concatenate(n_closest_units))

        entity_detection = np.concatenate(entity_detection)

        return np.concatenate([turn_identifier, game_states, entity_detection])

    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
        """
        Returns the reward function for this step of the game. Reward should be a
        delta increment to the reward, not the total current reward.
        """

        if is_game_error:
            # Game environment step failed, assign a game lost reward to not incentivise this
            print("Game failed due to error")
            return -1.0

        if not is_new_turn and not is_game_finished:
            # Only apply rewards at the start of each turn or at game end
            return 0

        """
        
        During Game
            [+++] city spawned
            [---] city destroyed
            
            [++] unit spawned
            [--] unit destroyed
            
            [+] fuel collected 
        
        End of game
            # ally cities - # enemy cities standing
            
            if above == 0
                # ally units - # enemy units
        
        """





        return 0

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
