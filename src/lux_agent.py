import numpy as np
from gym import spaces

from luxai2021.env.agent import AgentWithModel
from luxai2021.game.actions import *
from functools import partial


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

        self.observation_shape = num_obs
        self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float16)

    def get_observation(self, game, unit, city_tile, team, is_new_turn):
        """
        Implements getting a observation from the current game for this unit or city
        """

        """
        Features/Objects in Vector
        
        Turn Identifier
         - 1x Worker
         - 1x Cart
         - 1x City
        
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



        return np.zeros((10, 1))

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
