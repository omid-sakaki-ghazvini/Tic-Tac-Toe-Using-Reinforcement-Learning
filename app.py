import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import time

class TicTacToeGame:
    """Core game logic for Tic-Tac-Toe"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Initialize or reset the game state"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 for X (AI), 2 for O (Human)
        self.done = False
        self.winner = None
        return self.board.copy()
    
    def make_move(self, action):
        """Execute a move and return game state"""
        row, col = action // 3, action % 3
        
        # Validate move
        if self.board[row, col] != 0:
            return False
            
        self.board[row, col] = self.current_player
        
        # Check win condition
        if self._check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
            return True
            
        # Check draw condition
        if np.all(self.board != 0):
            self.done = True
            return True
            
        self.current_player = 3 - self.current_player  # Switch player
        return True
    
    def _check_win(self, player):
        """Check if specified player has won"""
        # Check rows and columns
        for i in range(3):
            if all(self.board[i, :] == player) or all(self.board[:, i] == player):
                return True
        # Check diagonals
        if all(np.diag(self.board) == player) or all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False


class RLAgent:
    """Reinforcement Learning Agent using TD Learning"""
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.state_values = defaultdict(float)
    
    def get_state_key(self, state):
        """Convert board state to hashable key"""
        return str(state.flatten())
    
    def select_action(self, state, available_actions):
        """Choose action using ε-greedy policy"""
        if random.random() < self.epsilon:
            return random.choice(available_actions)
            
        # Evaluate possible moves
        best_action = None
        max_value = -np.inf
        
        for action in available_actions:
            next_state = state.copy()
            row, col = action // 3, action % 3
            next_state[row, col] = 1  # Assume agent is player 1 (X)
            
            value = self.state_values.get(self.get_state_key(next_state), 0)
            if value > max_value:
                max_value = value
                best_action = action
                
        return best_action if best_action is not None else random.choice(available_actions)
    
    def update_model(self, state, reward, next_state, done):
        """Update value function using TD learning"""
        state_key = self.get_state_key(state)
        next_value = 0 if done else self.state_values.get(self.get_state_key(next_state), 0)
        td_target = reward + self.gamma * next_value
        self.state_values[state_key] += self.alpha * (td_target - self.state_values.get(state_key, 0))


def render_game_board(board):
    """Create visual representation of game board"""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Draw grid lines
    for i in range(1, 3):
        ax.axvline(i, color='black', linewidth=2)
        ax.axhline(i, color='black', linewidth=2)
    
    # Draw X's and O's
    for row in range(3):
        for col in range(3):
            if board[row, col] == 1:  # X
                ax.plot([col + 0.2, col + 0.8], [2 - row + 0.2, 2 - row + 0.8], 'r-', linewidth=4)
                ax.plot([col + 0.8, col + 0.2], [2 - row + 0.2, 2 - row + 0.8], 'r-', linewidth=4)
            elif board[row, col] == 2:  # O
                circle = plt.Circle((col + 0.5, 2 - row + 0.5), 0.3, color='blue', fill=False, linewidth=4)
                ax.add_patch(circle)
    
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    return fig


def initialize_session():
    """Initialize Streamlit session state"""
    if 'game' not in st.session_state:
        st.session_state.game = TicTacToeGame()
        st.session_state.game.reset()
    if 'agent' not in st.session_state:
        st.session_state.agent = RLAgent()
    if 'waiting_for_ai' not in st.session_state:
        st.session_state.waiting_for_ai = False


def main():
    """Main application function"""
    # Configure page
    st.set_page_config(
        page_title="Tic-Tac-Toe AI",
        page_icon="❎⭕",
        layout="centered"
    )
    
    # Initialize session
    initialize_session()
    
    # Page header
    st.title("Tic-Tac-Toe AI Challenge")
    st.markdown("Play against a reinforcement learning AI opponent")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Game Settings")
        st.session_state.agent.epsilon = st.slider(
            "AI Difficulty (ε)",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            help="Higher values make the AI more random"
        )
        
        if st.button("New Game"):
            st.session_state.game.reset()
            st.session_state.waiting_for_ai = False
            st.rerun()
    
    # Game board display
    st.header("Game Board")
    board_fig = render_game_board(st.session_state.game.board)
    st.pyplot(board_fig)
    
    # Game status
    if st.session_state.game.done:
        if st.session_state.game.winner == 1:
            st.success("AI (X) wins!")
        elif st.session_state.game.winner == 2:
            st.error("You (O) win!")
        else:
            st.info("It's a draw!")
    else:
        st.write(f"Current turn: {'AI (X)' if st.session_state.game.current_player == 1 else 'You (O)'}")
    
    # Human move input
    if not st.session_state.game.done and st.session_state.game.current_player == 2:
        st.subheader("Your Move")
        cols = st.columns(3)
        for i in range(9):
            row, col = i // 3, i % 3
            if st.session_state.game.board[row, col] == 0:
                if cols[col].button(f"Row {row+1}, Col {col+1}", key=f"move_{i}"):
                    st.session_state.game.make_move(i)
                    st.rerun()
    
    # AI move logic
    if (not st.session_state.game.done and 
        st.session_state.game.current_player == 1 and 
        not st.session_state.waiting_for_ai):
        
        st.session_state.waiting_for_ai = True
        available_actions = [
            i for i in range(9) 
            if st.session_state.game.board[i // 3, i % 3] == 0
        ]
        
        # Add slight delay for better UX
        time.sleep(0.5)
        ai_move = st.session_state.agent.select_action(
            st.session_state.game.board,
            available_actions
        )
        st.session_state.game.make_move(ai_move)
        st.session_state.waiting_for_ai = False
        st.rerun()


if __name__ == "__main__":
    main()
