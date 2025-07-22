import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import time
import pandas as pd

class TicTacToeGame:
    """Core game logic for Tic-Tac-Toe"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 for X (AI), 2 for O (Human)
        self.done = False
        self.winner = None
        return self.board.copy()
    
    def make_move(self, action):
        row, col = action // 3, action % 3
        
        if self.board[row, col] != 0:
            return False
            
        self.board[row, col] = self.current_player
        
        if self._check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
            return True
            
        if np.all(self.board != 0):
            self.done = True
            return True
            
        self.current_player = 3 - self.current_player
        return True
    
    def _check_win(self, player):
        for i in range(3):
            if all(self.board[i, :] == player) or all(self.board[:, i] == player):
                return True
        if all(np.diag(self.board) == player) or all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False


class RLAgent:
    """Reinforcement Learning Agent using TD Learning"""
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_values = defaultdict(float)
        self.learning_history = []
    
    def get_state_key(self, state):
        return str(state.flatten())
    
    def select_action(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)
            
        best_action = None
        max_value = -np.inf
        
        for action in available_actions:
            next_state = state.copy()
            row, col = action // 3, action % 3
            next_state[row, col] = 1
            
            value = self.state_values.get(self.get_state_key(next_state), 0)
            if value > max_value:
                max_value = value
                best_action = action
                
        return best_action if best_action is not None else random.choice(available_actions)
    
    def update_model(self, state, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_value = 0 if done else self.state_values.get(self.get_state_key(next_state), 0)
        td_target = reward + self.gamma * next_value
        td_error = td_target - self.state_values.get(state_key, 0)
        self.state_values[state_key] += self.alpha * td_error
        
        # Record learning progress
        self.learning_history.append({
            'episode': len(self.learning_history),
            'td_error': abs(td_error),
            'state_value': self.state_values[state_key]
        })


def render_game_board(board):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xticks([])
    ax.set_yticks([])
    
    for i in range(1, 3):
        ax.axvline(i, color='black', linewidth=2)
        ax.axhline(i, color='black', linewidth=2)
    
    for row in range(3):
        for col in range(3):
            if board[row, col] == 1:
                ax.plot([col + 0.2, col + 0.8], [2 - row + 0.2, 2 - row + 0.8], 'r-', linewidth=4)
                ax.plot([col + 0.8, col + 0.2], [2 - row + 0.2, 2 - row + 0.8], 'r-', linewidth=4)
            elif board[row, col] == 2:
                circle = plt.Circle((col + 0.5, 2 - row + 0.5), 0.3, color='blue', fill=False, linewidth=4)
                ax.add_patch(circle)
    
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    return fig


def initialize_session():
    if 'game' not in st.session_state:
        st.session_state.game = TicTacToeGame()
        st.session_state.game.reset()
    if 'agent' not in st.session_state:
        st.session_state.agent = RLAgent()
    if 'waiting_for_ai' not in st.session_state:
        st.session_state.waiting_for_ai = False
    if 'training_data' not in st.session_state:
        st.session_state.training_data = pd.DataFrame(columns=['episode', 'td_error', 'state_value'])


def plot_learning_progress():
    if len(st.session_state.agent.learning_history) > 0:
        df = pd.DataFrame(st.session_state.agent.learning_history)
        fig, ax = plt.subplots(figsize=(10, 4))
        
        ax.plot(df['episode'], df['td_error'], label='TD Error')
        ax.plot(df['episode'], df['state_value'], label='State Value')
        
        ax.set_xlabel('Training Episodes')
        ax.set_ylabel('Value')
        ax.set_title('Agent Learning Progress')
        ax.legend()
        ax.grid()
        
        st.pyplot(fig)


def main():
    st.set_page_config(
        page_title="Tic-Tac-Toe AI",
        page_icon="❎⭕",
        layout="wide"
    )
    
    initialize_session()
    
    st.title("Tic-Tac-Toe AI Challenge")
    st.markdown("Play against a reinforcement learning AI opponent")
    
    # Agent Settings Panel
    with st.sidebar:
        st.header("Agent Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.agent.alpha = st.slider(
                "Learning Rate (α)",
                min_value=0.01,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help="How quickly the agent adapts to new information"
            )
            
            st.session_state.agent.gamma = st.slider(
                "Discount Factor (γ)",
                min_value=0.1,
                max_value=0.99,
                value=0.9,
                step=0.01,
                help="How much the agent values future rewards"
            )
            
        with col2:
            st.session_state.agent.epsilon = st.slider(
                "Exploration Rate (ε)",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                help="Probability of random exploration"
            )
            
            training_episodes = st.slider(
                "Training Episodes",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                help="Number of training games to play"
            )
        
        if st.button("Train Agent"):
            with st.spinner(f"Training for {training_episodes} episodes..."):
                train_agent(training_episodes)
            st.success("Training completed!")
        
        if st.button("Reset Agent"):
            st.session_state.agent = RLAgent(
                alpha=st.session_state.agent.alpha,
                gamma=st.session_state.agent.gamma,
                epsilon=st.session_state.agent.epsilon
            )
            st.rerun()
    
    # Main Game Area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Game Board")
        board_fig = render_game_board(st.session_state.game.board)
        st.pyplot(board_fig)
        
        if st.session_state.game.done:
            if st.session_state.game.winner == 1:
                st.success("AI (X) wins!")
            elif st.session_state.game.winner == 2:
                st.error("You (O) win!")
            else:
                st.info("It's a draw!")
        else:
            st.write(f"Current turn: {'AI (X)' if st.session_state.game.current_player == 1 else 'You (O)'}")
        
        if st.button("New Game"):
            st.session_state.game.reset()
            st.rerun()
    
    with col2:
        st.header("Learning Analytics")
        plot_learning_progress()
        
        st.header("Make Your Move")
        if not st.session_state.game.done and st.session_state.game.current_player == 2:
            cols = st.columns(3)
            for i in range(9):
                row, col = i // 3, i % 3
                if st.session_state.game.board[row, col] == 0:
                    if cols[col].button(f"Row {row+1}, Col {col+1}", key=f"move_{i}"):
                        st.session_state.game.make_move(i)
                        st.rerun()
        
        if (not st.session_state.game.done and 
            st.session_state.game.current_player == 1 and 
            not st.session_state.waiting_for_ai):
            
            st.session_state.waiting_for_ai = True
            available_actions = [
                i for i in range(9) 
                if st.session_state.game.board[i // 3, i % 3] == 0
            ]
            
            time.sleep(0.5)
            ai_move = st.session_state.agent.select_action(
                st.session_state.game.board,
                available_actions
            )
            st.session_state.game.make_move(ai_move)
            st.session_state.waiting_for_ai = False
            st.rerun()


def train_agent(episodes):
    """Train the agent through self-play"""
    temp_env = TicTacToeGame()
    
    for episode in range(episodes):
        temp_env.reset()
        done = False
        
        while not done:
            available_actions = [
                i for i in range(9) 
                if temp_env.board[i // 3, i % 3] == 0
            ]
            
            if temp_env.current_player == 1:
                action = st.session_state.agent.select_action(
                    temp_env.board, 
                    available_actions
                )
            else:
                action = random.choice(available_actions)
                
            next_state = temp_env.board.copy()
            temp_env.make_move(action)
            reward = 1 if temp_env.winner == 1 else (-1 if temp_env.winner == 2 else 0)
            
            if temp_env.current_player == 1:
                st.session_state.agent.update_model(
                    next_state,
                    reward,
                    temp_env.board,
                    temp_env.done
                )
            
            done = temp_env.done


if __name__ == "__main__":
    main()
