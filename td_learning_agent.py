import numpy as np
import pickle
from collections import defaultdict

class TDAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.state_values = defaultdict(float)
        
    def get_state_key(self, state):
        return str(state.flatten())
    
    def choose_action(self, state, available_actions):
        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
            
        state_key = self.get_state_key(state)
        max_value = -np.inf
        best_action = None
        
        for action in available_actions:
            next_state = state.copy()
            row, col = action // 3, action % 3
            next_state[row, col] = Player.X.value  # Agent plays X
            
            next_state_key = self.get_state_key(next_state)
            value = self.state_values.get(next_state_key, 0)
            
            if value > max_value:
                max_value = value
                best_action = action
                
        return best_action if best_action is not None else np.random.choice(available_actions)
    
    def update(self, state, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        current_value = self.state_values.get(state_key, 0)
        next_value = self.state_values.get(next_state_key, 0) if not done else 0
        
        td_target = reward + self.gamma * next_value
        td_error = td_target - current_value
        
        self.state_values[state_key] = current_value + self.alpha * td_error
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'state_values': dict(self.state_values),
                'params': {
                    'alpha': self.alpha,
                    'gamma': self.gamma,
                    'epsilon': self.epsilon
                }
            }, f)
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.state_values = defaultdict(float, data['state_values'])
            params = data['params']
            self.alpha = params['alpha']
            self.gamma = params['gamma']
            self.epsilon = params['epsilon']
