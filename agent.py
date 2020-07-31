import numpy as np
import random
from collections import namedtuple, deque


import torch
import torch.nn.functional as F
import torch.optim as optim

from settings import *
from network import Actor, Critic
from util import OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size: int, action_size: int, seed: int, n_agent: int):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.n_agent = n_agent
        self.seed = random.seed(seed)

        # Initialize actor and critic local and target networks
        self.actor = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(
            state_size, action_size, seed).to(device)
        self.critic = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(
            state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=CRITIC_LEARNING_RATE)
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = [0] * n_agent
        self.noises = [OUNoise(action_size, seed*i) for i in range(self.n_agent)]

        # Copy parameters from local network to target network
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def step(self, state: np.array, action, reward, next_state, done, i_agent):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step[i_agent] = (self.t_step[i_agent] + 1) % UPDATE_EVERY
        # Learn, if enough samples are available in memory every UPDATE_EVERY
        if len(self.memory) > BATCH_SIZE and (not any(self.t_step)):
            for _ in range(LEARN_TIMES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def noise_reset(self):
        [self.noises[i_agent].reset() for i_agent in range(self.n_agent)]

    def save_model(self, checkpoint_path: str = "./checkpoints/"):
        torch.save(self.actor.state_dict(), f"{checkpoint_path}/actor.pt")
        torch.save(self.critic.state_dict(), f"{checkpoint_path}/critic.pt")

    def load_model(self, checkpoint_path: str = "./checkpoints/checkpoint.pt"):
        self.actor.load_state_dict(torch.load(f"{checkpoint_path}/actor.pt"))
        self.critic.load_state_dict(torch.load(f"{checkpoint_path}/critic.pt"))

    def act(self, states: np.array, step:int):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
        """
        states = torch.from_numpy(states).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action_values = self.actor(states).cpu().data.numpy()
        self.actor.train()
        for action, noise in zip(action_values, self.noises):
            action = noise.get_action(action,t=step)
        return np.clip(action_values, -1, 1)

    def learn(self, experiences: tuple, gamma=GAMMA):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # Critic loss
        mask = torch.tensor(1-dones).detach().to(device)
        Q_values = self.critic(states, actions)
        next_actions = self.actor_target(next_states)
        next_Q = self.critic_target(next_states, next_actions.detach())
        Q_prime = rewards + gamma * next_Q * mask
        critic_loss = F.mse_loss(Q_values, Q_prime.detach())

        # Update critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if CRITIC_GRADIENT_CLIPPING_VALUE:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), CRITIC_GRADIENT_CLIPPING_VALUE)
        self.critic_optimizer.step()

        # Actor loss
        policy_loss =  -self.critic(states, self.actor(states)).mean()

        # Update actor network
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        if ACTOR_GRADIENT_CLIPPING_VALUE:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), ACTOR_GRADIENT_CLIPPING_VALUE)
        self.actor_optimizer.step()


        self.actor_soft_update()
        self.critic_soft_update()

    def actor_soft_update(self, tau: float = TAU):
        """Soft update for actor target network

        Args:
            tau (float, optional). Defaults to TAU.
        """
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data)

    def critic_soft_update(self, tau: float = TAU):
        """Soft update for critic target network

        Args:
            tau (float, optional). Defaults to TAU.
        """
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.detach_()
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
