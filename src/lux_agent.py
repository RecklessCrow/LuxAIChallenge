import numpy as np
from gym import spaces
from luxai2021.game.game import Game

from constants import *
from luxai2021.env.agent import AgentWithModel
from luxai2021.game.actions import *
from functools import partial

from luxai2021.game.game_constants import GAME_CONSTANTS
from luxai2021.game.position import Position


def smart_transfer_to_nearby(game, team, unit_id, unit, target_type_restriction=None, **kwarg):
    """
    Smart-transfers from the specified unit to a nearby neighbor. Prioritizes any
    nearby carts first, then any worker. Transfers the resource type which the unit
    has most of. Picks which cart/worker based on choosing a target that is most-full
    but able to take the most amount of resources.
    Args:
        team ([type]): [description]
        unit_id ([type]): [description]
    Returns:
        Action: Returns a TransferAction object, even if the request is an invalid
                transfer. Use TransferAction.is_valid() to check validity.
    """

    # Calculate how much resources could at-most be transferred
    resource_type = None
    resource_amount = 0
    target_unit = None

    if unit != None:
        for type, amount in unit.cargo.items():
            if amount > resource_amount:
                resource_type = type
                resource_amount = amount

        # Find the best nearby unit to transfer to
        unit_cell = game.map.get_cell_by_pos(unit.pos)
        adjacent_cells = game.map.get_adjacent_cells(unit_cell)

        for c in adjacent_cells:
            for id, u in c.units.items():
                # Apply the unit type target restriction
                if target_type_restriction == None or u.type == target_type_restriction:
                    if u.team == team:
                        # This unit belongs to our team, set it as the winning transfer target
                        # if it's the best match.
                        if target_unit is None:
                            target_unit = u
                        else:
                            # Compare this unit to the existing target
                            if target_unit.type == u.type:
                                # Transfer to the target with the least capacity, but can accept
                                # all of our resources
                                if (u.get_cargo_space_left() >= resource_amount and
                                        target_unit.get_cargo_space_left() >= resource_amount):
                                    # Both units can accept all our resources. Prioritize one that is most-full.
                                    if u.get_cargo_space_left() < target_unit.get_cargo_space_left():
                                        # This new target it better, it has less space left and can take all our
                                        # resources
                                        target_unit = u

                                elif (target_unit.get_cargo_space_left() >= resource_amount):
                                    # Don't change targets. Current one is best since it can take all
                                    # the resources, but new target can't.
                                    pass

                                elif (u.get_cargo_space_left() > target_unit.get_cargo_space_left()):
                                    # Change targets, because neither target can accept all our resources and
                                    # this target can take more resources.
                                    target_unit = u
                            elif u.type == Constants.UNIT_TYPES.CART:
                                # Transfer to this cart instead of the current worker target
                                target_unit = u

    # Build the transfer action request
    target_unit_id = None
    if target_unit is not None:
        target_unit_id = target_unit.id

        # Update the transfer amount based on the room of the target
        if target_unit.get_cargo_space_left() < resource_amount:
            resource_amount = target_unit.get_cargo_space_left()


class LuxAgent(AgentWithModel):
    def __init__(self, mode="train", model=None):
        super().__init__(mode, model)

        # Initialize actions
        self.unit_actions = [
            partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # No Op
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),

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
        self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float16)

        self.object_nodes = {}
        self.observation = np.zeros(OBSERVATION_SHAPE[0])

    def game_start(self, game):
        self.last_unit_count = 0
        self.last_city_tile_count = 0
        self.last_fuel_collected = 0

    def get_map_contents(self, game, team):
        # Build a list of object nodes by type for quick distance-searches
        self.object_nodes = {}

        # Add resources
        for cell in game.map.resources:
            if cell.resource.type not in self.object_nodes:
                self.object_nodes[cell.resource.type] = []
            self.object_nodes[cell.resource.type].append([cell.pos.x, cell.pos.y])

        # Add your own and opponent units
        for t in [team, (team + 1) % 2]:
            for u in game.state["teamStates"][team]["units"].values():
                key = str(u.type)
                if t != team:
                    key = str(u.type) + "_opponent"

                if key not in self.object_nodes:
                    self.object_nodes[key] = []
                self.object_nodes[key].append([u.pos.x, u.pos.y])

        # Add your own and opponent cities
        for city in game.cities.values():
            for cells in city.city_cells:
                key = "city"
                if city.team != team:
                    key = "city_opponent"

                if key not in self.object_nodes:
                    self.object_nodes[key] = []
                self.object_nodes[key].append([cells.pos.x, cells.pos.y])

        for key in self.object_nodes:
            self.object_nodes[key] = np.array(self.object_nodes[key])

    @staticmethod
    def distance(node, nodes):
        return np.sum((nodes - node) ** 2, axis=1)

    @staticmethod
    def get_cargo(game, cell, unit_type):
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

    @staticmethod
    def create_3Vector(observation, observation_idx, unit_pos, other_pos):
        if other_pos[0] - unit_pos.x == 0:  # center
            x_diff = .5
        elif other_pos[0] - unit_pos.x > 0:  # up
            x_diff = 1
        else:  # down
            x_diff = 0

        if other_pos[1] - unit_pos.y == 0:  # center
            y_diff = .5
        elif other_pos[1] - unit_pos.y > 0:  # up
            y_diff = 1
        else:  # down
            y_diff = 0

        # 1x distance
        distance = np.sqrt((other_pos[0] - unit_pos.x) ** 2 + (other_pos[1] - unit_pos.y) ** 2)

        observation[observation_idx + 0] = x_diff
        observation[observation_idx + 1] = y_diff
        observation[observation_idx + 2] = distance
        observation_idx += 3

    def get_observation(self, game: Game, unit, city_tile, team, is_new_turn: bool):
        """
        Implements getting a observation from the current game for this unit or city
        """

        if is_new_turn:
            self.get_map_contents(game, team)

        self.observation.fill(0)
        observation_idx = 0

        """
        Features/Objects in Vector
        
        Turn Identifier  - 3x:
         - 1x Worker
         - 1x Cart
         - 1x City
         """
        if unit is not None:
            if unit.type == Constants.UNIT_TYPES.WORKER:
                self.observation[observation_idx + 0] = 1.0  # Worker
            else:
                self.observation[observation_idx + 1] = 1.0  # Cart
        if city_tile is not None:
            self.observation[observation_idx + 2] = 1.0  # CityTile

        observation_idx += 3

        """
        Game State - 12x:
         - 1x cargo size
         - 1x percent of day state complete
         - 1x is night
         - 1x percent of day/night cycle complete
         - 2x city tile counts [cur player, opponent]
         - 2x worker counts [cur player, opponent]
         - 2x cart counts [cur player, opponent]
         
         - 1x research points [cur player]
         - 1x researched coal [cur player]
         - 1x researched uranium [cur player]
        """

        # 1x cargo size
        if unit is not None:
            self.observation[observation_idx] = unit.get_cargo_space_left() / \
                                          GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
        observation_idx += 1

        # 1x percent of day/night cycle complete
        if game.is_night():
            self.observation[observation_idx] = (game.state['turn'] % NUM_STEPS_IN_NIGHT) / NUM_STEPS_IN_NIGHT
        else:
            self.observation[observation_idx] = (game.state['turn'] % NUM_STEPS_IN_DAY) / NUM_STEPS_IN_DAY
        observation_idx += 1

        # 1x is night
        self.observation[observation_idx] = game.is_night()
        observation_idx += 1

        # 1x percent of game done
        self.observation[observation_idx] = game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
        observation_idx += 1

        # 6x unit counts
        for key in ["city", str(Constants.UNIT_TYPES.WORKER), str(Constants.UNIT_TYPES.CART)]:
            if key in self.object_nodes:
                self.observation[observation_idx] = len(self.object_nodes[key]) / MAX_UNIT_COUNT
            if (key + "_opponent") in self.object_nodes:
                self.observation[observation_idx + 1] = len(self.object_nodes[(key + "_opponent")]) / MAX_UNIT_COUNT
            observation_idx += 2

        # 1x research points
        self.observation[observation_idx] = game.state["teamStates"][team]["researchPoints"] / MAX_RESEARCH
        observation_idx += 1

        # 1x researched coal
        self.observation[observation_idx] = float(game.state["teamStates"][team]["researched"]["coal"])
        observation_idx += 1

        # 1x researched uranium
        self.observation[observation_idx] = float(game.state["teamStates"][team]["researched"]["uranium"])
        observation_idx += 1

        if unit is not None:
            unit_pos = unit.pos
        else:
            unit_pos = city_tile.pos

        """
        Entity Detection - 42x:
        
        Nearest Cart - 4x:
         - 2x [vector direction]
         - 1x distance
         - 1x amount
         
        Nearest Worker - 4x:
         - 2x [vector direction]
         - 1x distance
         - 1x amount
        
        Worker with fullest inventory - 4x:
         - 2x [vector direction]
         - 1x distance
         - 1x amount
        
        Nearest City - 4x:
         - 2x [vector direction]
         - 1x distance
         - 1x amount fuel
        
        City with least amount of Fuel - 4x:
         - 2x [vector direction]
         - 1x distance
         - 1x amount fuel
        
        Resources - 48x:
        - 3x per resource 
         - 3x for 3 nearest resource piles
          - 2x [vector direction]
          - 1x distance
          - 1x amount
         
        """

        assert observation_idx == 3 + 13

        # ToDo Check unit for team
        # game.get_teams_units(self.team) ?

        types = {
            Constants.RESOURCE_TYPES.WOOD: 3,
            Constants.RESOURCE_TYPES.COAL: 3,
            Constants.RESOURCE_TYPES.URANIUM: 3,
            "city": 1,
            str(Constants.UNIT_TYPES.WORKER): 1,
            str(Constants.UNIT_TYPES.CART): 1
        }

        # Nearest Entity
        for entity_type in types.keys():
            if entity_type not in self.object_nodes:
                sorted_idx = np.array([])
            else:
                nodes = self.object_nodes[entity_type]
                sorted_idx = np.argsort(self.distance(np.array([unit_pos.x, unit_pos.y]), nodes))

                if unit is not None and unit.type == entity_type or entity_type == "city" and city_tile is not None:
                    sorted_idx = np.delete(sorted_idx, 0)

            # todo check unit's team
            # sorted_idx = np.array([node for node in sorted_idx if node.team])

            n_closest_units = []
            for i in range(types[entity_type]):
                if i >= sorted_idx.size:
                    n_closest_units.append(np.array([0, 0, 0, 0]))
                    continue

                other_pos = nodes[sorted_idx[i]]

                self.create_3Vector(self.observation, observation_idx, unit_pos, other_pos)

                # 1x amount
                other_cell = game.map.get_cell_by_pos(Position(other_pos[0], other_pos[1]))
                cargo_amount = self.get_cargo(game, other_cell, entity_type)
                self.observation[observation_idx] = cargo_amount
                observation_idx += 1

        # ToDo Worker with fullest inventory (remove self)
        units = game.get_teams_units(self.team)

        if units:
            def get_unit_cargo(unit):
                return sum(game.get_unit(self.team, unit).cargo.values())

            max_unit_id = max(units, key=get_unit_cargo)
            max_unit = game.get_unit(self.team, max_unit_id)
            other_pos = [max_unit.pos.x, max_unit.pos.y]

            self.create_3Vector(self.observation, observation_idx, unit_pos, other_pos)

            # 1x amount
            cargo_amount = get_unit_cargo(max_unit_id)
            self.observation[observation_idx] = cargo_amount
            observation_idx += 1

        # ToDo City with least amount of fuel
        if "city" in self.object_nodes:
            def get_city_cargo(city_pos):
                cell = game.map.get_cell_by_pos(Position(city_pos[0], city_pos[1]))
                return self.get_cargo(game, cell, "city")

            starving_city_position = min(self.object_nodes["city"], key=get_city_cargo)
            self.create_3Vector(self.observation, observation_idx, unit_pos, starving_city_position)
            cargo_amount = get_city_cargo(starving_city_position)
            self.observation[observation_idx] = cargo_amount
            observation_idx += 1

        return self.observation

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

        # bigger negative reward than positive
        if city_growth < 0:
            city_growth *= 1.5

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
