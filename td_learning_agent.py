import numpy as np
import pickle
from collections import defaultdict

class TDAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1, epsilon_min=0.01, epsilon_decay=0.995, player=1):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.player = player  # 1 for X, 2 for O
        self.state_values = defaultdict(float)
        self.state_action_counts = defaultdict(int)

    def get_canonical_state(self, state):
        min_state = state
        for _ in range(3):  # Rotate 0, 90, 180, 270 degrees
            rotated = np.rot90(state)
            flipped = np.fliplr(rotated)
            min_state = min(min_state.flatten(), rotated.flatten(), flipped.flatten())
            state = rotated
        return str(min_state)

    def choose_action(self, state, available_actions):
        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        state_key = self.get_canonical_state(state)
        max_value = -np.inf
        best_action = None
        for action in available_actions:
            next_state = state.copy()
            row, col = action // 3, action % 3
            next_state[row, col] = self.player
            next_state_key = self.get_canonical_state(next_state)
            value = self.state_values.get(next_state_key, 0)
            if value > max_value:
                max_value = value
                best_action = action
        return best_action if best_action is not None else max(available_actions, key=lambda a: self.state_action_counts.get(self.get_canonical_state(state.copy().flatten()[a // 3, a % 3] = self.player), 0))

    def update(self, state, reward, next_state, done):
        state_key = self.get_canonical_state(state)
        next_state_key = self.get_canonical_state(next_state)
        current_value = self.state_values.get(state_key, 0)
        next_value = self.state_values.get(next_state_key, 0) if not done else 0
        td_target = reward + self.gamma * next_value
        td_error = td_target - current_value
        self.state_values[state_key] = current_value + self.alpha * td_error
        self.state_action_counts[state_key] += 1

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'state_values': dict(self.state_values),
                    'params': {
                        'alpha': self.alpha,
                        'gamma': self.gamma,
                        'epsilon': self.epsilon,
                        'player': self.player
                    }
                }, f)
        except Exception as e:
            print(f"Error saving model: {e}")

    def load(self, filename):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.state_values = defaultdict(float, data['state_values'])
                params = data['params']
                self.alpha = params['alpha']
                self.gamma = params['gamma']
                self.epsilon = params['epsilon']
                self.player = params.get('player', 1)
        except FileNotFoundError:
            print(f"File {filename} not found.")
        except Exception as e:
            print(f"Error loading model: {e}")
