import streamlit as st
from enum import Enum

# Define the Player enum
class Player(Enum):
    EMPTY = 0
    X = 1
    O = 2

# Initialize session state if not present
if "board" not in st.session_state:
    st.session_state.board = [[Player.EMPTY.value for _ in range(3)] for _ in range(3)]
if "current_player" not in st.session_state:
    st.session_state.current_player = Player.X.value
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "winner" not in st.session_state:
    st.session_state.winner = None

# Function to handle button click
def handle_click(i, j):
    if not st.session_state.game_over and st.session_state.board[i][j] == Player.EMPTY.value:
        st.session_state.board[i][j] = st.session_state.current_player
        check_winner()
        if not st.session_state.game_over:
            st.session_state.current_player = Player.O.value if st.session_state.current_player == Player.X.value else Player.X.value
        st.rerun()

# Function to check for a winner
def check_winner():
    board = st.session_state.board
    # Check rows
    for row in board:
        if row == [Player.X.value, Player.X.value, Player.X.value]:
            st.session_state.game_over = True
            st.session_state.winner = Player.X.value
            return
        if row == [Player.O.value, Player.O.value, Player.O.value]:
            st.session_state.game_over = True
            st.session_state.winner = Player.O.value
            return
    # Check columns
    for col in range(3):
        if [board[row][col] for row in range(3)] == [Player.X.value, Player.X.value, Player.X.value]:
            st.session_state.game_over = True
            st.session_state.winner = Player.X.value
            return
        if [board[row][col] for row in range(3)] == [Player.O.value, Player.O.value, Player.O.value]:
            st.session_state.game_over = True
            st.session_state.winner = Player.O.value
            return
    # Check diagonals
    if [board[i][i] for i in range(3)] == [Player.X.value, Player.X.value, Player.X.value]:
        st.session_state.game_over = True
        st.session_state.winner = Player.X.value
        return
    if [board[i][i] for i in range(3)] == [Player.O.value, Player.O.value, Player.O.value]:
        st.session_state.game_over = True
        st.session_state.winner = Player.O.value
        return
    if [board[i][2-i] for i in range(3)] == [Player.X.value, Player.X.value, Player.X.value]:
        st.session_state.game_over = True
        st.session_state.winner = Player.X.value
        return
    if [board[i][2-i] for i in range(3)] == [Player.O.value, Player.O.value, Player.O.value]:
        st.session_state.game_over = True
        st.session_state.winner = Player.O.value
        return
    # Check for draw
    if all(cell != Player.EMPTY.value for row in board for cell in row):
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
                    handle_click(i, j)

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
display_board()
show_status()
st.markdown("---")

# Reset button
if st.button("Reset Game"):
    if "board" in st.session_state:
        del st.session_state.board
    if "current_player" in st.session_state:
        del st.session_state.current_player
    if "game_over" in st.session_state:
        del st.session_state.game_over
    if "winner" in st.session_state:
        del st.session_state.winner
    st.rerun()
