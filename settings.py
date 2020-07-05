"""Module for project hyper parameters"""

ENV_PATH = './data/Reacher.app'
ACTOR_NETWORK_LINEAR_SIZES = "300,200"
CRITIC_NETWORK_LINEAR_SIZES = "200,100"
ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 1e-3
BUFFER_SIZE = int(1e5)          # replay buffer size
BATCH_SIZE = 128                 # minibatch size
UPDATE_EVERY = 1                # how often to update the network
GAMMA = 0.99                    # discount factor
TAU = 1e-3                      # for soft update of target parameters
CRITIC_BATCH_NORM = True        # apply batch norm for critic network
ACTOR_BATCH_NORM = True         # apply batch norm for actor network


def printvars():
   tmp = globals().copy()
   [print(k,' = ',v) for k,v in tmp.items() if not k.startswith('_') and k!='tmp' and k!='In' and k!='Out' and not hasattr(v, '__call__')]

printvars()