# -*- coding: utf-8 -*-
import streamlit as st
from tictactoe_env import TicTacToeEnv, Player
from td_learning_agent import TDAgent
import numpy as np
import pickle

# Initialize session state
if "board" not in st.session_state:
    st.session_state.board = [[Player.EMPTY.value for _ in range(3)] for _ in range(3)]
if "current_player" not in st.session_state:
    st.session_state.current_player = Player.X.value
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "winner" not in st.session_state:
    st.session_state.winner = None
if "agent" not in st.session_state:
    try:
        with open('trained_agent.pkl', 'rb') as f:
            st.session_state.agent = pickle.load(f)
    except FileNotFoundError:
        st.session_state.agent = TDAgent(player=Player.O.value)

# Function to handle human move
def handle_human_move(i, j):
    if not st.session_state.game_over and st.session_state.board[i][j] == Player.EMPTY.value:
        st.session_state.board[i][j] = st.session_state.current_player
        check_winner()
        if not st.session_state.game_over:
            st.session_state.current_player = Player.O.value
            agent_move()
        st.rerun()

# Function to handle agent move
def agent_move():
    env = TicTacToeEnv()
    env.board = np.array(st.session_state.board)
    available_actions = env.get_available_actions()
    if available_actions:
        action = st.session_state.agent.choose_action(env.board, available_actions)
        row, col = divmod(action, 3)
        st.session_state.board[row][col] = Player.O.value
        check_winner()
        st.rerun()

# Function to check for a winner
def check_winner():
    board = np.array(st.session_state.board)
    for player in [Player.X.value, Player.O.value]:
        if (np.all(board == player, axis=1).any() or
            np.all(board == player, axis=0).any() or
            np.all(np.diag(board) == player) or
            np.all(np.diag(np.fliplr(board)) == player)):
            st.session_state.game_over = True
            st.session_state.winner = player
            return
    if np.all(board != Player.EMPTY.value):
        st.session_state.game_over = True
        st.session_state.winner = None

# Function to display the game board
def display_board():
    board = st.session_state.board
    cols = st.columns(3)
    for i in range(3):
        with cols[i]:
            for j in range(3):
                cell_value = board[i][j]
                display_text = " " if cell_value == Player.EMPTY.value else ("X" if cell_value == Player.X.value else "O")
                disabled = cell_value != Player.EMPTY.value or st.session_state.game_over
                if st.button(display_text, key=f"cell_{i}_{j}", disabled=disabled):
                    handle_human_move(i, j)

# Function to show game status
def show_status():
    if st.session_state.game_over:
        if st.session_state.winner is not None:
            st.success(f"Player {'X' if st.session_state.winner == Player.X.value else 'O'} wins!")
        else:
            st.info("It's a draw!")
    else:
        st.info(f"Current turn: {'X' if st.session_state.current_player == Player.X.value else 'O'}")

# Main app
st.title("🎮 Tic-Tac-Toe AI")
st.write("Play against an AI trained with Reinforcement Learning!")
display_board()
show_status()
st.markdown("---")

# Reset button
if st.button("Reset Game"):
    for key in ["board", "current_player", "game_over", "winner"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Train or load agent button
if st.button("Train New Agent"):
    from train_agent import train_agent
    env = TicTacToeEnv()
    agent = TDAgent(player=Player.O.value)
    train_agent(env, agent, episodes=10000)
    st.session_state.agent = agent
    with open('trained_agent.pkl', 'wb') as f:
        pickle.dump(agent, f)
    st.success("Agent trained and saved!")
    st.rerun()
