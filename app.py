import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import random

# Set up the page
st.set_page_config(page_title="Tic-Tac-Toe AI", layout="wide")
st.title("🎮 Tic-Tac-Toe AI with RL")

# Game Environment
class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # X starts
        self.done = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.board.copy()

    def step(self, action):
        row, col = action // 3, action % 3
        if self.board[row, col] != 0:
            return self.board.copy(), -10, True
        
        self.board[row, col] = self.current_player
        
        if self._check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1 if self.current_player == 1 else -1
            return self.board.copy(), reward, True
            
        if np.all(self.board != 0):
            self.done = True
            return self.board.copy(), 0, True
            
        self.current_player = 3 - self.current_player
        return self.board.copy(), 0, False
    
    def _check_win(self, player):
        for i in range(3):
            if all(self.board[i, :] == player) or all(self.board[:, i] == player):
                return True
        if all(np.diag(self.board) == player) or all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

# RL Agent
class TDAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_values = defaultdict(float)
    
    def get_state_key(self, state):
        return str(state.flatten())
    
    def choose_action(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)
            
        max_value = -np.inf
        best_action = None
        
        for action in available_actions:
            next_state = state.copy()
            row, col = action // 3, action % 3
            next_state[row, col] = 1  # Assume agent is X
            
            value = self.state_values.get(self.get_state_key(next_state), 0)
            if value > max_value:
                max_value = value
                best_action = action
                
        return best_action or random.choice(available_actions)
    
    def update(self, state, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_value = 0 if done else self.state_values.get(self.get_state_key(next_state), 0)
        td_target = reward + self.gamma * next_value
        self.state_values[state_key] += self.alpha * (td_target - self.state_values.get(state_key, 0))

# Streamlit UI
def main():
    # Initialize agent and environment
    if 'agent' not in st.session_state:
        st.session_state.agent = TDAgent()
        st.session_state.env = TicTacToeEnv()
        st.session_state.board = st.session_state.env.reset()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        if st.button("New Game"):
            st.session_state.board = st.session_state.env.reset()
        
        st.slider("AI Strength (ε)", 0.01, 0.5, 0.1, key="epsilon")
        st.session_state.agent.epsilon = st.session_state.epsilon
    
    # Game board
    st.header("Current Game")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Draw board
    for i in range(1, 3):
        ax.axvline(i, color='black', linewidth=2)
        ax.axhline(i, color='black', linewidth=2)
    
    # Draw X/O
    for row in range(3):
        for col in range(3):
            if st.session_state.board[row, col] == 1:
                ax.plot([col + 0.2, col + 0.8], [2 - row + 0.2, 2 - row + 0.8], 'r-', linewidth=4)
                ax.plot([col + 0.8, col + 0.2], [2 - row + 0.2, 2 - row + 0.8], 'r-', linewidth=4)
            elif st.session_state.board[row, col] == 2:
                circle = plt.Circle((col + 0.5, 2 - row + 0.5), 0.3, color='blue', fill=False, linewidth=4)
                ax.add_patch(circle)
    
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    st.pyplot(fig)
    
    # Game status
    if st.session_state.env.done:
        if st.session_state.env.winner == 1:
            st.success("AI wins!")
        elif st.session_state.env.winner == 2:
            st.error("You win!")
        else:
            st.info("It's a draw!")
    else:
        st.write(f"Current turn: {'AI (X)' if st.session_state.env.current_player == 1 else 'You (O)'}")
    
    # Human move input
    if not st.session_state.env.done and st.session_state.env.current_player == 2:
        cols = st.columns(3)
        for i in range(9):
            row, col = i // 3, i % 3
            if st.session_state.board[row, col] == 0:
                if cols[col].button(f"⬜ ({row}, {col})", key=f"btn_{i}"):
                    st.session_state.board, _, _ = st.session_state.env.step(i)
                    st.experimental_rerun()
    
    # AI move
    if not st.session_state.env.done and st.session_state.env.current_player == 1:
        available_actions = [i for i in range(9) if st.session_state.board[i // 3, i % 3] == 0]
        action = st.session_state.agent.choose_action(st.session_state.board, available_actions)
        st.session_state.board, _, _ = st.session_state.env.step(action)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
