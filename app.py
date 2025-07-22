import streamlit as st
import numpy as np
from tictactoe_env import TicTacToeEnv, Player
from td_learning_agent import TDAgent
import pickle
import time

# Initialize game and agent
@st.cache_resource
def initialize_game():
    env = TicTacToeEnv()
    agent = TDAgent(alpha=0.5, gamma=0.9, epsilon=0.1)
    
    try:
        agent.load('tictactoe_agent.pkl')
    except:
        # Train agent with reduced episodes for Streamlit Cloud
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

# Initialize session state
if 'game_data' not in st.session_state:
    env, agent = initialize_game()
    env.reset()
    st.session_state.game_data = {
        'env': env,
        'agent': agent,
        'board': env.board.copy(),
        'done': env.done,
        'winner': env.winner
    }

# Streamlit app
st.title("🎮 Tic-Tac-Toe AI")
st.markdown("Play against an AI trained with Reinforcement Learning")

# Improved board display using columns with proper keys
def display_board():
    board = st.session_state.game_data['board']
    cols = st.columns(3)
    
    # Add custom CSS for consistent button sizing
    st.markdown("""
    <style>
        div[data-testid="column"] {
            text-align: center;
        }
        div[data-testid="column"] button {
            width: 80px !important;
            height: 80px !important;
            font-size: 40px !important;
            padding: 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    for i in range(3):
        for j in range(3):
            with cols[j]:
                cell_value = board[i, j]
                display_text = " " if cell_value == Player.EMPTY.value else "❌" if cell_value == Player.X.value else "⭕"
                
                # Create unique key for each button
                button_key = f"btn_{i}_{j}_{int(time.time())}"
                
                if st.button(
                    display_text,
                    key=button_key,
                    disabled=(cell_value != Player.EMPTY.value or st.session_state.game_data['done']),
                    on_click=handle_click,
                    args=(i*3 + j,)
                ):
                    pass

def handle_click(action):
    game_data = st.session_state.game_data
    if not game_data['done'] and game_data['env'].current_player == Player.O.value:
        # Human move (O)
        _, _, game_data['done'], _ = game_data['env'].step(action)
        game_data['board'] = game_data['env'].board.copy()
        game_data['winner'] = game_data['env'].winner
        
        # AI move (X) if game continues
        if not game_data['done']:
            available_actions = game_data['env'].get_available_actions()
            action = game_data['agent'].choose_action(game_data['env'].board, available_actions)
            _, _, game_data['done'], _ = game_data['env'].step(action)
            game_data['board'] = game_data['env'].board.copy()
            game_data['winner'] = game_data['env'].winner
        
        st.session_state.game_data = game_data

# Game status display
def show_status():
    game_data = st.session_state.game_data
    if game_data['done']:
        if game_data['winner'] == Player.X.value:
            st.error("🤖 AI (❌) wins!")
        elif game_data['winner'] == Player.O.value:
            st.success("🎉 You (⭕) win!")
        else:
            st.info("🤝 It's a draw!")
        
        if st.button("New Game"):
            game_data['env'].reset()
            game_data['board'] = game_data['env'].board.copy()
            game_data['done'] = game_data['env'].done
            game_data['winner'] = game_data['env'].winner
            st.session_state.game_data = game_data
    else:
        current_player = "🤖 AI (❌)" if game_data['env'].current_player == Player.X.value else "You (⭕)"
        st.write(f"Current turn: {current_player}")

# Main app layout
def main():
    display_board()
    show_status()
    
    st.markdown("---")
    st.markdown("""
    ### How to play:
    1. You play as ⭕ (O)
    2. AI plays as ❌ (X)
    3. Click any empty cell to make your move
    4. The AI will respond automatically
    """)

if __name__ == "__main__":
    main()
