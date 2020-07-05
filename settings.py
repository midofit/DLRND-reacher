"""Module for project hyper parameters"""

ENV_PATH = './data/Reacher.app'
ACTOR_NETWORK_LINEAR_SIZES = "300,200"
CRITIC_NETWORK_LINEAR_SIZES = "100,100"
ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 1e-3
BUFFER_SIZE = int(1e5)          # replay buffer size
BATCH_SIZE = 64                 # minibatch size
UPDATE_EVERY = 4                # how often to update the network
GAMMA = 0.99                    # discount factor
TAU = 1e-3                      # for soft update of target parameters
CRITIC_BATCH_NORM = False        # apply batch norm for critic network
ACTOR_BATCH_NORM = True         # apply batch norm for actor network