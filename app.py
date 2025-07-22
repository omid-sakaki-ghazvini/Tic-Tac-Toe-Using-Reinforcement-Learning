import streamlit as st
import numpy as np
from tictactoe_env import TicTacToeEnv
from td_learning_agent import TDAgent
import pickle

# Initialize game and agent
env = TicTacToeEnv()
agent = TDAgent(alpha=0.5, gamma=0.9, epsilon=0.1)

# Load or train agent
try:
    agent.load('tictactoe_agent.pkl')
except:
    st.warning("Training a new agent...")
    # Quick training
    for _ in range(1000):
        state = env.reset()
        done = False
        while not done:
            available_actions = env.get_available_actions()
            if env.current_player == Player.X.value:
                action = agent.choose_action(state, available_actions)
            else:
                action = np.random.choice(available_actions)
            next_state, reward, done, _ = env.step(action)
            if env.current_player == Player.X.value:
                agent.update(state, reward, next_state, done)
            state = next_state
    agent.save('tictactoe_agent.pkl')

# Streamlit UI
st.title("🎮 Tic-Tac-Toe AI")
st.markdown("Play against an AI trained with Temporal Difference Learning")

# Initialize session state
if 'board' not in st.session_state:
    env.reset()
    st.session_state.board = env.board.copy()
    st.session_state.done = env.done

# Game display
def display_board():
    cols = st.columns(3)
    for i in range(3):
        for j in range(3):
            with cols[j]:
                cell_value = st.session_state.board[i, j]
                display_text = " " if cell_value == 0 else "❌" if cell_value == 1 else "⭕"
                if st.button(
                    display_text,
                    key=f"cell_{i}_{j}",
                    disabled=(cell_value != 0 or st.session_state.done),
                    on_click=handle_click,
                    args=(i*3 + j,),
                    height=100,
                    width=100
                ):
                    pass

def handle_click(action):
    if not st.session_state.done:
        # Human move (O)
        _, _, st.session_state.done, _ = env.step(action)
        st.session_state.board = env.board.copy()
        
        # AI move (X) if game not over
        if not st.session_state.done:
            available_actions = env.get_available_actions()
            action = agent.choose_action(env.board, available_actions)
            _, _, st.session_state.done, _ = env.step(action)
            st.session_state.board = env.board.copy()

# Game status
def show_status():
    if st.session_state.done:
        if env.winner == Player.X.value:
            st.error("🤖 AI wins!")
        elif env.winner == Player.O.value:
            st.success("🎉 You win!")
        else:
            st.info("🤝 It's a draw!")
        
        if st.button("New Game"):
            env.reset()
            st.session_state.board = env.board.copy()
            st.session_state.done = env.done
            st.experimental_rerun()
    else:
        st.write(f"Current turn: {'🤖 AI (X)' if env.current_player == Player.X.value else 'You (O)'}")

# Main game loop
display_board()
show_status()

# Instructions
st.markdown("---")
st.markdown("""
### How to play:
1. Click on empty cells to place your ⭕
2. The AI will automatically place ❌
3. Try to get 3 in a row to win
4. Click 'New Game' to restart
""")
