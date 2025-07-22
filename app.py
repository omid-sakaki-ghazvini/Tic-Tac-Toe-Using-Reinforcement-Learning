import streamlit as st
import numpy as np
from tictactoe_env import TicTacToeEnv, Player
from td_learning_agent import TDAgent
import pickle

# Initialize game and agent
@st.cache_resource
def initialize_game():
    env = TicTacToeEnv()
    agent = TDAgent(alpha=0.5, gamma=0.9, epsilon=0.1)
    
    # Load or train agent
    try:
        agent.load('tictactoe_agent.pkl')
    except:
        # Train with reduced episodes for Streamlit Cloud
        for _ in range(500):  
            state = env.reset()
            done = False
            while not done:
                available_actions = env.get_available_actions()
                if env.current_player == Player.X.value:
                    action = agent.choose_action(state, available_actions)
                else:
                    action = np.random.choice(available_actions)
                next_state, _, done, _ = env.step(action)
                if env.current_player == Player.X.value:
                    reward = 1 if env.winner == Player.X.value else -1 if env.winner == Player.O.value else 0
                    agent.update(state, reward, next_state, done)
                state = next_state
        agent.save('tictactoe_agent.pkl')
    return env, agent

env, agent = initialize_game()

# Initialize session state
if 'board' not in st.session_state:
    env.reset()
    st.session_state.board = env.board.copy()
    st.session_state.done = env.done
    st.session_state.winner = env.winner

# Streamlit UI
st.title("🎮 Tic-Tac-Toe AI")
st.markdown("Play against an RL-trained AI")

# Game display with improved button handling
def display_board():
    cols = st.columns(3)
    button_styles = """
    <style>
        div[data-testid="column"] > button {
            height: 100px !important;
            width: 100px !important;
            font-size: 40px !important;
        }
    </style>
    """
    st.markdown(button_styles, unsafe_allow_html=True)
    
    for i in range(3):
        for j in range(3):
            with cols[j]:
                cell_value = st.session_state.board[i, j]
                display_text = " " if cell_value == Player.EMPTY.value else "❌" if cell_value == Player.X.value else "⭕"
                disabled = cell_value != Player.EMPTY.value or st.session_state.done
                
                if st.button(
                    display_text,
                    key=f"cell_{i}_{j}",
                    disabled=disabled,
                    on_click=handle_click,
                    args=(i*3 + j,)
                ):
                    pass

def handle_click(action):
    if not st.session_state.done:
        # Human move (O)
        _, _, st.session_state.done, _ = env.step(action)
        st.session_state.board = env.board.copy()
        st.session_state.winner = env.winner
        
        # AI move (X) if game not over
        if not st.session_state.done:
            available_actions = env.get_available_actions()
            action = agent.choose_action(env.board, available_actions)
            _, _, st.session_state.done, _ = env.step(action)
            st.session_state.board = env.board.copy()
            st.session_state.winner = env.winner

# Game status
def show_status():
    if st.session_state.done:
        if st.session_state.winner == Player.X.value:
            st.error("🤖 AI wins!")
        elif st.session_state.winner == Player.O.value:
            st.success("🎉 You win!")
        else:
            st.info("🤝 It's a draw!")
        
        if st.button("New Game"):
            env.reset()
            st.session_state.board = env.board.copy()
            st.session_state.done = env.done
            st.session_state.winner = env.winner
            st.experimental_rerun()
    else:
        current_player = "🤖 AI (X)" if env.current_player == Player.X.value else "You (O)"
        st.write(f"Current turn: {current_player}")

# Main game layout
def main():
    display_board()
    show_status()
    st.markdown("---")
    st.markdown("""
    ### How to play:
    1. Click any empty cell to place your ⭕
    2. The AI will automatically place ❌
    3. Get 3 in a row to win
    4. Click 'New Game' to restart
    """)

if __name__ == "__main__":
    main()
