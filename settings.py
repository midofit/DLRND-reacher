"""Module for project hyper parameters"""

ENV_PATH = './data/Reacher_One_Linux_NoVis.x86_64'
ACTOR_NETWORK_LINEAR_SIZES = "1024,512,256"
CRITIC_NETWORK_LINEAR_SIZES = "1024,200"
LEARNING_RATE = 1e-3
BUFFER_SIZE = 1e5       # replay buffer size
BATCH_SIZE = 256        # minibatch size
UPDATE_EVERY = 10       # how often to update the network
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
