import numpy as np
import gym
from gym import spaces
from enum import Enum

class Player(Enum):
    EMPTY = 0
    X = 1
    O = 2

class TicTacToeEnv(gym.Env):
    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        self.board_size = 3
        self.action_space = spaces.Discrete(self.board_size ** 2)
        self.observation_space = spaces.Box(
            low=0, high=2,
            shape=(self.board_size, self.board_size),
            dtype=np.int32
        )
        self.reset()
        
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self.current_player = Player.X.value
        self.done = False
        self.winner = None
        return self.board.copy()
    
    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, {}
            
        row, col = action // self.board_size, action % self.board_size
        
        if self.board[row, col] != Player.EMPTY.value:
            return self.board.copy(), -10, True, {}
            
        self.board[row, col] = self.current_player
        
        if self._check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1 if self.current_player == Player.X.value else -1
            return self.board.copy(), reward, True, {}
            
        if self._is_board_full():
            self.done = True
            return self.board.copy(), 0, True, {}
            
        self.current_player = Player.O.value if self.current_player == Player.X.value else Player.X.value
        return self.board.copy(), 0, False, {}
    
    def _check_win(self, player):
        # Check rows
        for row in range(self.board_size):
            if all(self.board[row, :] == player):
                return True
                
        # Check columns
        for col in range(self.board_size):
            if all(self.board[:, col] == player):
                return True
                
        # Check diagonals
        if all(np.diag(self.board) == player):
            return True
        if all(np.diag(np.fliplr(self.board)) == player):
            return True
            
        return False
    
    def _is_board_full(self):
        return np.all(self.board != Player.EMPTY.value)
    
    def get_available_actions(self):
        return [i for i in range(9) if self.board[i // 3, i % 3] == 0]
