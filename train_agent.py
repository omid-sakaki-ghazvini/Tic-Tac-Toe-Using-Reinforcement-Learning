# -*- coding: utf-8 -*-
import numpy as np
from tictactoe_env import TicTacToeEnv
from td_learning_agent import TDAgent

def train_agent(env, agent, episodes=10000):
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            available_actions = env.get_available_actions()
            action = agent.choose_action(state, available_actions)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, reward, next_state, done)
            state = next_state
        agent.decay_epsilon()
        if episode % 1000 == 0:
            print(f"Episode {episode}, Epsilon: {agent.epsilon:.4f}")
    agent.save('trained_agent.pkl')

if __name__ == "__main__":
    env = TicTacToeEnv()
    agent = TDAgent(player=Player.O.value)
    train_agent(env, agent)
