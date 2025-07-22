import streamlit as st
import numpy as np
from tictactoe_env import TicTacToeEnv, Player
from td_learning_agent import TDAgent
import pickle

# Custom CSS for styling
st.markdown("""
<style>
    .big-button {
        height: 100px !important;
        width: 100px !important;
        font-size: 40px !important;
    }
    .stButton>button {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize game
if 'env' not in st.session_state:
    st.session_state.env = TicTacToeEnv()
    st.session_state.env.reset()
    st.session_state.agent = TDAgent(alpha=0.5, gamma=0.9, epsilon=0.1)
    
    # Load or train agent
    try:
        st.session_state.agent.load('tictactoe_agent.pkl')
    except:
        # Train with reduced episodes for Streamlit
        for _ in range(500):
            state = st.session_state.env.reset()
            done = False
            while not done:
                available_actions = st.session_state.env.get_available_actions()
                if st.session_state.env.current_player == Player.X.value:
                    action = st.session_state.agent.choose_action(state, available_actions)
                else:
                    action = np.random.choice(available_actions)
                next_state, _, done, _ = st.session_state.env.step(action)
                if st.session_state.env.current_player == Player.X.value:
                    reward = 1 if st.session_state.env.winner == Player.X.value else -1 if st.session_state.env.winner == Player.O.value else 0
                    st.session_state.agent.update(state, reward, next_state, done)
                state = next_state
        st.session_state.agent.save('tictactoe_agent.pkl')

# Main app
st.title("🎮 Tic-Tac-Toe AI")
st.markdown("Play against an AI trained with Reinforcement Learning")

# Game board
cols = st.columns(3)
for i in range(3):
    for j in range(3):
        with cols[j]:
            cell_value = st.session_state.env.board[i, j]
            display_text = " " if cell_value == Player.EMPTY.value else "❌" if cell_value == Player.X.value else "⭕"
            
            if st.button(
                display_text,
                key=f"cell_{i}_{j}",
                disabled=(cell_value != Player.EMPTY.value or st.session_state.env.done),
                help="Make your move" if cell_value == Player.EMPTY.value else None
            ):
                # Human move (O)
                action = i * 3 + j
                st.session_state.env.step(action)
                
                # AI move (X) if game continues
                if not st.session_state.env.done:
                    available_actions = st.session_state.env.get_available_actions()
                    action = st.session_state.agent.choose_action(st.session_state.env.board, available_actions)
                    st.session_state.env.step(action)
                
                st.rerun()

# Game status
if st.session_state.env.done:
    if st.session_state.env.winner == Player.X.value:
        st.error("🤖 AI (❌) wins!")
    elif st.session_state.env.winner == Player.O.value:
        st.success("🎉 You (⭕) win!")
    else:
        st.info("🤝 It's a draw!")
    
    if st.button("New Game"):
        st.session_state.env.reset()
        st.rerun()
else:
    current_player = "🤖 AI (❌)" if st.session_state.env.current_player == Player.X.value else "You (⭕)"
    st.write(f"Current turn: {current_player}")

# Instructions
st.markdown("---")
st.markdown("""
### How to play:
1. You play as ⭕ (O)
2. AI plays as ❌ (X)
3. Click any empty cell to make your move
4. The AI will respond automatically
""")
