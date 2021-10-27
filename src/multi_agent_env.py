import numpy as np
from pettingzoo import AECEnv

from constants import *
from luxai2021.game.game import Game
from luxai2021.game.match_controller import MatchController, GameStepFailedException
from pettingzoo.utils import agent_selector


class MultiAgentLuxEnv(AECEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, configs, agent0, agent1,
                 replay_validate=None, replay_folder=None, replay_prefix="replay"):
        super().__init__()

        # Create the game
        self.game = Game(configs)
        self.match_controller = MatchController(
            self.game,
            agents=[agent0, agent1],
            replay_validate=replay_validate
        )

        self.replay_prefix = replay_prefix
        self.replay_folder = replay_folder

        self.possible_agents = {
            Constants.TEAM.A: agent0,
            Constants.TEAM.B: agent1
        }

        self.observation_spaces = {

        }

        self.current_step = 0
        self.match_generator = None

        self.last_observation_object = None

    def step(self, action_code):
        agent = self.agents[self.last_observation_object[2]]

        agent.take_action(
            action_code,
            self.game,
            unit=self.last_observation_object[0],
            city_tile=self.last_observation_object[1],
            team=self.last_observation_object[2]
        )

        self.current_step += 1

        # Get the next observation
        is_new_turn = True
        is_game_over = False
        is_game_error = False
        try:
            (unit, city_tile, team, is_new_turn) = next(self.match_generator)

            obs = agent.get_observation(self.game, unit, city_tile, team, is_new_turn)
            self.last_observation_object = (unit, city_tile, team, is_new_turn)
        except StopIteration:
            # The game episode is done.
            is_game_over = True
            obs = None
        except GameStepFailedException:
            # Game step failed, assign a game lost reward to not incentivise this
            is_game_over = True
            obs = None
            is_game_error = True

        # Calculate reward for this step
        reward = agent.get_reward(self.game, is_game_over, is_new_turn, is_game_error)

        return obs, reward, is_game_over, {}

    def reset(self):
        self.current_step = 0
        self.last_observation_object = None

        # Reset game + map
        self.match_controller.reset()
        if self.replay_folder:
            # Tell the game to log replays
            self.game.start_replay_logging(stateful=True, replay_folder=self.replay_folder,
                                           replay_filename_prefix=self.replay_prefix)

        self.match_generator = self.match_controller.run_to_next_observation()
        (unit, city_tile, team, is_new_turn) = next(self.match_generator)

        obs = self.learning_agent.get_observation(self.game, unit, city_tile, team, is_new_turn)
        self.last_observation_object = (unit, city_tile, team, is_new_turn)

        # Petting Zoo reset
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: NONE for agent in self.agents}
        self.observations = {agent: agent.get_observation(self.game, unit, city_tile, team, is_new_turn) for agent in
                             self.agents}
        self.num_moves = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent):
        pass

    def render(self, mode='human'):
        pass

    def state(self):
        pass
