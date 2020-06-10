""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import navdb, scr, settings, sim, stack, tools, traf

from mlres.agent import ActorCritic
from mlres.data__loader import Data_Loader
from mlres.traff_management import Traff_Management as TM

import torch
import torch.optim as optim

# Initialization function of your plugin. Do not change the name of this
# function, as it is the way BlueSky recognises this file as a plugin.

EPISODES = 50

# Hyper params:
HIDDEN_SIZE = 256
LR = 3e-4
MIN_BATCH_SIZE = 5
PPO_EPOCHS = 4

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def init_plugin():
    global tm
    global dl
    global model
    global optimizer
    global actions
    global closest
    global current_epoch

    # The number of closest ac
    closest = 4

    current_epoch = 0

    # tm = TM(
    #     verbose=True, routes_file="routes/sim2.npy", time_spawn=True)
    # Use this for distance based spawning

    tm = TM(
        verbose=True, routes_file="routes/sim2.npy", time_spawn=False)

    dl = Data_Loader(closest=4, verbose=True)

    actions = [24000, 26000, 28000, 30000, 32000]

    temp = torch.zeros(closest + 1, 5).flatten()
    model = ActorCritic(
        temp.shape[0], len(actions), HIDDEN_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Addtional initilisation code

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'ML_TRAFF',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': 12,

        # The update function is called after traffic is updated. Use this if you
        # want to do things as a result of what happens in traffic. If you need to
        # something before traffic is updated please use preupdate.
        'update':          update,

        # The preupdate function is called before traffic is updated. Use this
        # function to provide settings that need to be used by traffic in the current
        # timestep. Examples are ASAS, which can give autopilot commands to resolve
        # a conflict.
        'preupdate':       preupdate,

        # If your plugin has a state, you will probably need a reset function to
        # clear the state in between simulations.
        'reset':         reset
    }

    stackfunctions = {
    }

    # init_plugin() should always return these two dicts.
    return config, stackfunctions


# Periodic update functions that are called by the simulation. You can replace
# this by anything, so long as you communicate this in init_plugin

def update():
    global tm
    global dl

    # Update the traffic
    tm.update()

    for ac in traf.id:
        ac_data = dl.get_data(traf, ac)
        closest_n = dl.get_context_data(traf, ac)

        needed = 4 - len(closest_n)

        for i in range(needed):
            closest_n.append(np.zeros(5))
        closest_n = np.array(closest_n)

    # Check to see if the episode has finished
    if tm.check_state():
        check_over()


def check_over():
    if current_epoch >= EPISODES:
        pass

    reset()


def reset():
    global tm

    tm.reset()
