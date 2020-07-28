import torch
from unityagents import UnityEnvironment
import numpy as np
from collections import namedtuple, deque
from agent import Agent
import matplotlib.pyplot as plt
from environment import env
from util import OUNoise


def ddpg(env, agent, brain_name, action_size, n_episodes=2000, max_t=1000, n_agent=20):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    best_score = 0
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[
            brain_name]  # reset the environment
        states = env_info.vector_observations
        agent.noise_reset()
        agent_scores = [0]*n_agent
        for step in range(max_t):
            actions = agent.act(states, step)
            env_info = env.step(actions)[brain_name]     # send the action to the environment
            next_states = env_info.vector_observations   # get the next state
            rewards = env_info.rewards                   # get the reward
            dones = env_info.local_done                  # see if episode has finished
            for i_agent in range(n_agent):
                agent_scores[i_agent] += rewards[i_agent]
                agent.step(states[i_agent], actions[i_agent], rewards[i_agent],
                           next_states[i_agent], dones[i_agent], i_agent)
            states = next_states
            if any(dones):
                break
        score = np.mean(agent_scores)
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        if best_score < score:
            best_score = score
        print('\rEpisode {}\t Episode score: {:.2f}\t Average Score: {:.2f}\t Best Score: {:.2f}'.format(
            i_episode, score, np.mean(scores_window), best_score), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\t Current score: {:.2f}\t Average Score: {:.2f}'.format(
                i_episode, score, np.mean(scores_window)))
        if np.mean(scores_window) >= 30:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-100, np.mean(scores_window)))
            agent.save_model()
            break
    env.close()
    return scores


if __name__ == "__main__":
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    print("Brain name: ", brain_name)
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Action size', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    state_size = len(state)
    print('States have length:', state_size)
    agent = Agent(state_size=state_size,
                  action_size=action_size, seed=2, n_agent=20)
    scores = ddpg(env, agent, brain_name, action_size)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
