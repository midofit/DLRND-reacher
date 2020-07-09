"""Module for project hyper parameters"""

ENV_PATH = './data/Reacher2.app'
ACTOR_NETWORK_LINEAR_SIZES = "512,384"
CRITIC_NETWORK_LINEAR_SIZES = "512,384"
ACTOR_LEARNING_RATE = 5e-4
CRITIC_LEARNING_RATE = 5e-4
BUFFER_SIZE = int(2e6)          # replay buffer size
BATCH_SIZE = 512                # minibatch size
UPDATE_EVERY = 20               # how often to update the network
GAMMA = 0.99                    # discount factor
TAU = 1e-3                      # for soft update of target parameters
CRITIC_BATCH_NORM = True        # apply batch norm for critic network
ACTOR_BATCH_NORM = True         # apply batch norm for actor network
LEARN_TIMES = 10

def printvars():
   tmp = globals().copy()
   [print(k,' = ',v) for k,v in tmp.items() if not k.startswith('_') and k!='tmp' and k!='In' and k!='Out' and not hasattr(v, '__call__')]

printvars()