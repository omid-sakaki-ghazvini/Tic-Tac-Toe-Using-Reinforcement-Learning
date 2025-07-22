import streamlit as st
import numpy as np
from tictactoe_env import TicTacToeEnv, Player  # Make sure to import Player
from td_learning_agent import TDAgent
import pickle

# Initialize game and agent
env = TicTacToeEnv()
agent = TDAgent(alpha=0.5, gamma=0.9, epsilon=0.1)

# Load or train agent
try:
    agent.load('tictactoe_agent.pkl')
    st.success("AI agent loaded successfully!")
except:
    st.warning("Training a new agent...")
    # Quick training
    for _ in range(1000):
        state = env.reset()
        done = False
        while not done:
            available_actions = env.get_available_actions()
            if env.current_player == Player.X.value:  # Now properly referenced
                action = agent.choose_action(state, available_actions)
            else:
                action = np.random.choice(available_actions)
            next_state, reward, done, _ = env.step(action)
            if env.current_player == Player.X.value:  # Fixed reference
                agent.update(state, reward, next_state, done)
            state = next_state
    agent.save('tictactoe_agent.pkl')
    st.success("Training completed!")

# Streamlit UI
st.title("🎮 Tic-Tac-Toe AI")
st.markdown("Play against an AI trained with Reinforcement Learning")

# Initialize session state
if 'env' not in st.session_state:
    env.reset()
    st.session_state.env = env
    st.session_state.board = env.board.copy()
    st.session_state.done = env.done

# Game display
def display_board():
    cols = st.columns(3)
    for i in range(3):
        for j in range(3):
            with cols[j]:
                cell_value = st.session_state.board[i, j]
                display_text = " " if cell_value == Player.EMPTY.value else "❌" if cell_value == Player.X.value else "⭕"
                if st.button(
                    display_text,
                    key=f"cell_{i}_{j}",
                    disabled=(cell_value != Player.EMPTY.value or st.session_state.done),
                    on_click=handle_click,
                    args=(i*3 + j,),
                    height=100,
                    width=100
                ):
                    pass

def handle_click(action):
    if not st.session_state.done:
        # Human move (O)
        _, _, st.session_state.done, _ = st.session_state.env.step(action)
        st.session_state.board = st.session_state.env.board.copy()
        
        # AI move (X) if game not over
        if not st.session_state.done:
            available_actions = st.session_state.env.get_available_actions()
            action = agent.choose_action(st.session_state.env.board, available_actions)
            _, _, st.session_state.done, _ = st.session_state.env.step(action)
            st.session_state.board = st.session_state.env.board.copy()
        st.experimental_rerun()

# Game status
def show_status():
    if st.session_state.done:
        if st.session_state.env.winner == Player.X.value:
            st.error("🤖 AI wins!")
        elif st.session_state.env.winner == Player.O.value:
            st.success("🎉 You win!")
        else:
            st.info("🤝 It's a draw!")
        
        if st.button("New Game"):
            st.session_state.env.reset()
            st.session_state.board = st.session_state.env.board.copy()
            st.session_state.done = st.session_state.env.done
            st.experimental_rerun()
    else:
        current_player = "🤖 AI (X)" if st.session_state.env.current_player == Player.X.value else "You (O)"
        st.write(f"Current turn: {current_player}")

# Main game loop
display_board()
show_status()
