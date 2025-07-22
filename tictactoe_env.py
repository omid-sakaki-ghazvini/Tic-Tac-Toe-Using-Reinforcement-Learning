import numpy as np
import gym
from gym import spaces
from enum import Enum

class Player(Enum):
    EMPTY = 0
    X = 1
    O = 2

class TicTacToeEnv(gym.Env):
    def __init__(self, invalid_move_penalty=-1):
        super(TicTacToeEnv, self).__init__()
        self.board_size = 3
        self.action_space = spaces.Discrete(self.board_size ** 2)
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.board_size, self.board_size), dtype=np.int32)
        self.invalid_move_penalty = invalid_move_penalty
        self.players = [Player.X.value, Player.O.value]
        self.current_player_index = 0
        self.history = []
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self.current_player = self.players[self.current_player_index]
        self.done = False
        self.winner = None
        self.history = []
        return self.board.copy()

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, {}

        row, col = action // self.board_size, action % self.board_size
        if self.board[row, col] != Player.EMPTY.value:
            return self.board.copy(), self.invalid_move_penalty, True, {}

        self.board[row, col] = self.current_player
        self.history.append(self.board.copy())

        if self._check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1 if self.current_player == Player.X.value else -1
            return self.board.copy(), reward, True, {"available_actions": self.get_available_actions()}

        if self._is_board_full():
            self.done = True
            return self.board.copy(), 0, True, {"available_actions": self.get_available_actions()}

        self.current_player_index = 1 - self.current_player_index
        self.current_player = self.players[self.current_player_index]
        return self.board.copy(), 0, False, {"available_actions": self.get_available_actions()}

    def _check_win(self, player):
        rows = np.all(self.board == player, axis=1)
        cols = np.all(self.board == player, axis=0)
        diag1 = np.all(np.diag(self.board) == player)
        diag2 = np.all(np.diag(np.fliplr(self.board)) == player)
        return any(rows) or any(cols) or diag1 or diag2

    def _is_board_full(self):
        return np.all(self.board != Player.EMPTY.value)

    def get_available_actions(self):
        return [i for i in range(9) if self.board[i // 3, i % 3] == Player.EMPTY.value]

    def render(self, mode='human'):
        if mode == 'human':
            for row in self.board:
                print([Player.X.name if cell == Player.X.value else Player.O.name if cell == Player.O.value else ' ' for cell in row])
            print()
