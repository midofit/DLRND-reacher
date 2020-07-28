"""Module for project hyper parameters"""

ENV_PATH = './data/Reacher_2/Reacher.exe'
ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 1e-3
BUFFER_SIZE = int(1e6)          # replay buffer size
BATCH_SIZE = 256               # minibatch size
UPDATE_EVERY = 2               # how often to update the network
GAMMA = 0.99                    # discount factor
TAU = 1e-3                      # for soft update of target parameters
CRITIC_BATCH_NORM = True        # apply batch norm for critic network
ACTOR_BATCH_NORM = True         # apply batch norm for actor network
LEARN_TIMES = 1
CRITIC_GRADIENT_CLIPPING_VALUE = 1
ACTOR_GRADIENT_CLIPPING_VALUE = 0

def printvars():
   tmp = globals().copy()
   [print(k,' = ',v) for k,v in tmp.items() if not k.startswith('_') and k!='tmp' and k!='In' and k!='Out' and not hasattr(v, '__call__')]

printvars()