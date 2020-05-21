import gym
from dqn_linear import Agent
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import plotLearning


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, lr=.003, input_dims=env.observation_space.shape, batch_size=64, n_actions=4)
    scores, eps_history = [], []
    n_games = 100

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            screen = env.render(mode='rgb_array')
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
    x = [i + 1 for i in range(n_games)]
    filename = 'lunar_lander.png'
    plotLearning(x, scores, eps_history, filename)