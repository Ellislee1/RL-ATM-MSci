import math
import numpy as np

import torch
from bluesky import traf
from bluesky.tools import geo


class Data_Loader():
    def __init__(self, closest, verbose=True):
        self.verbose = verbose
        self.initilized = False
        self.maxhdg = 0.0
        self.minhdg = 360.0
        self.maxalt = None
        self.minalt = None
        self.maxvs = None
        self.minvs = None
        self.maxtas = None
        self.mintas = 0.0
        self.num_closest = closest
        self.c_mindist = 0
        self.c_maxdist = 0

        if self.verbose:
            print("###############\nPreparing Data Loader\n###############")

    def get_data(self, traf, callsign):
        hdg = None
        alt = None
        vs = None
        tas = None

        idx = traf.id2idx(callsign)

        hdg = traf.trk[idx]
        alt = traf.alt[idx]
        vs = traf.vs[idx]
        tas = traf.tas[idx]

        if not self.initilized:
            self.minalt = alt
            self.maxalt = alt
            self.minvs = vs
            self.maxvs = vs
            self.maxtas = tas
            self.initilized = True
        else:
            self.minalt = min(self.minalt, alt)
            self.maxalt = max(self.maxalt, alt)
            self.minvs = min(self.minvs, vs)
            self.maxvs = max(self.maxvs, vs)
            self.maxtas = max(self.maxtas, tas)

        return self.normalise_ac([hdg, alt, vs, tas, 0])

    def normalise_ac(self, ac_data, is_context=False):
        ac_data[0] = (ac_data[0] - self.minhdg) / (self.maxhdg - self.minhdg)
        ac_data[1] = (ac_data[1] - self.minalt) / (self.maxalt - self.minalt)
        ac_data[2] = (ac_data[2] - self.minvs) / (self.maxvs - self.minvs)
        ac_data[3] = (ac_data[3] - self.mintas) / (self.maxtas - self.mintas)

        if is_context:
            ac_data[4] = (ac_data[0] - self.c_mindist) / \
                (self.c_maxdist - self.c_mindist)

        for i in range(len(ac_data)):
            if math.isnan(ac_data[i]):
                ac_data[i] = 1.0

        return np.array(ac_data)

    def get_context_data(self, traf, callsign):
        _id = traf.id2idx(callsign)
        dist_matrix = self.get_dist_matrix(_id, traf)[_id]
        dist_matrix = np.array(dist_matrix).flatten()

        closest = self.get_closest(traf, _id)

        context = []

        for element in closest:
            idx, dist = element
            hdg = traf.trk[idx]
            alt = traf.alt[idx]
            vs = traf.vs[idx]
            tas = traf.tas[idx]

            context.append([hdg, alt, vs, tas, dist])

        if len(context) > 0:
            for i, val in enumerate(context):
                context[i] = self.normalise_ac(context[i], is_context=True)

        return context

    # Get the distance matrix for all aircraft in the system

    def get_dist_matrix(self, _id, traf):
        size = traf.lat.shape[0]
        dist_matrix = geo.latlondist_matrix(np.repeat(traf.lat, size),
                                            np.repeat(traf.lon, size),
                                            np.tile(traf.lat, size),
                                            np.tile(traf.lon, size)).reshape(size, size)
        return dist_matrix

    # The the closest n aircraft to a given ac
    def get_closest(self, traf, _id):
        # Get the distance matrix and prepare it
        dist_matrix = self.get_dist_matrix(_id, traf)[_id]
        dist_matrix = np.array(dist_matrix).flatten()

        closest = []

        # Loop through all elements in the matrix
        for i, dist in enumerate(dist_matrix):
            # Discard the curren ac
            if not i == _id:
                # Ensure that the closest array is full
                if len(closest) < self.num_closest:
                    closest.append([i, dist])
                    self.c_maxdist = max(self.c_maxdist, dist)
                else:
                    # Min max calcuation
                    for x, item in enumerate(closest):
                        c_dist = item[1]
                        if dist < c_dist:
                            self.c_maxdist = max(self.c_maxdist, dist)
                            closest[x] = [i, dist]
                            break

        return closest
