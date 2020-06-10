import random
import time

import numpy as np
from bluesky import stack, traf
from geopy import distance as dist


class Traff_Management:
    def __init__(self, routes_file, max_ac=20, time_spawn=True, times=[20, 25, 30], verbose=False, distances=[5, 6, 7], iata="BAW",
                 types=['A320'], max_speed=310, min_speed=310, max_alt=28000, min_alt=28000):
        print("Setting up Traffic Management")

        """
        Default variables
        """
        # Current number of ac in the sim
        self.active = 0
        # Total aircraft that have existed in the sim
        self.existed = 0
        # Count how many update calls there have been
        self.update_timer = 0
        # Track the routes each aircraft is on
        self.on_route = {}

        """
        User defined settings
        Default times [20,25,30] = 4,5,6 min seperation
        Default distances in NM [5,6,7]
        """
        # Maximum aircraft allowed in the system
        self.max_ac = max_ac
        # File for the routes path
        # Format: [spawn.lat, spawn.long, heading, goal.lat, goal.long]
        self.routes = np.load(routes_file)
        # Time based spawning for the ac
        self.time_spawn = time_spawn
        # The time offsets for creating aircraft
        self.times = times
        # The lateral separation spawning
        self.distances = distances
        # Prefix IATA for aircraft
        self.iata = iata
        # Types of creatable aircraft
        self.types = types
        # Min speed
        self.min_speed = min_speed
        # Max speed
        self.max_speed = max_speed
        # Min alt
        self.min_alt = min_alt
        # max alt
        self.max_alt = max_alt

        """
        Other Variables
        """
        # Used for distance based spawning
        self.last_spawned = []
        # Queue for time based spawning
        if self.time_spawn:
            self.spawn_queue = random.choices(
                self.times, k=self.routes.shape[0])
        else:
            self.spawn_queue = random.choices(
                self.distances, k=self.routes.shape[0])

        print(self.spawn_queue)

        # Output for debugging
        self.verbose = verbose

        # Run first ac spawn
        self.first()

    # Reset the variables to the origional
    def reset(self):
        # Current number of ac in the sim
        self.active = 0
        # Total aircraft that have existed in the sim
        self.existed = 0
        # Reset array
        self.last_spawned = []
        # Reset timer
        self.update_timer = 0
        # Reset routes
        self.on_route = {}

        # Queue for time based spawning
        if self.time_spawn:
            self.spawn_queue = random.choices(
                self.times, k=self.routes.shape[0])
        else:
            self.spawn_queue = random.choices(
                self.distances, k=self.routes.shape[0])

        print(self.spawn_queue)

        # Run first ac spawn
        self.first()

    # First run to get aircraft into the system
    def first(self):
        if self.verbose:
            print("Creating an aircraft at each point")

        for i, route in enumerate(self.routes):
            self.last_spawned.append(self.spawn_ac(route, i))

    # Code to spawn an aircraft
    def spawn_ac(self, route, route_idx):
        s_lat, s_lon, hdg, g_lat, g_lon = route
        callsign = self.iata + str(self.existed)
        ac_type = np.random.choice(self.types, 1)[0]
        # Generate random altitude between two altitudes
        alt = np.random.randint(self.min_alt / 1000,
                                high=(self.max_alt / 1000) + 1) * 1000
        # Generate random speed between two speeds
        speed = np.random.randint(
            self.min_speed / 10, high=(self.max_speed / 10) + 1)*10

        if self.verbose:
            print(
                f'Creating {callsign} {ac_type} at [{s_lat}, {s_lon}] for [{g_lat}, {g_lon}] heading {hdg}')

        # Create aircraft
        stack.stack('CRE, {}, {}, {}, {}, {}, {}, {}'.format(
            callsign, ac_type, s_lat, s_lon, hdg, alt, speed))
        # Set aircrafts destination
        stack.stack('ADDWPT {} {}, {}'.format(callsign, g_lat, g_lon))

        # Update last spawned aircraft
        if len(self.last_spawned) >= len(self.routes):
            self.last_spawned[route_idx] = callsign

        # Add aircraft to the enroute aircraft
        self.on_route[callsign] = route_idx

        # increatse counts
        self.existed += 1
        self.active += 1

        return callsign

    # Update the system
    def update(self):
        if self.time_spawn:
            self.time_update()
        else:
            self.dist_update()

        self.update_timer += 1

    # Update based on time
    def time_update(self):
        # Ensure that the max_ac hasnt been reached
        if self.existed <= self.max_ac:
            # Loop through spawn queue
            for k in range(len(self.spawn_queue)):
                # Check to see if the timers match
                if self.update_timer == self.spawn_queue[k]:
                    self.spawn_ac(self.routes[k], k)
                    self.spawn_queue[k] = self.update_timer + \
                        random.choices(self.times, k=1)[0]

                    if self.verbose:
                        print(self.on_route)

                if self.existed >= self.max_ac:
                    break

    # Update based on distance
    def dist_update(self):
        if self.existed <= self.max_ac:
            for k in range(len(self.spawn_queue)):
                # Get aircraft as well as current position and start position
                callsign = self.last_spawned[k]
                idx = traf.id2idx(callsign)
                lat, lon = traf.lat[idx], traf.lon[idx]
                s_lat, s_lon, _, _, _ = self.routes[self.on_route[callsign]]

                ac = (lat, lon)
                start = (s_lat, s_lon)

                # Get the distance from the starting position in nm
                distance = dist.distance(ac, start).nm

                if distance >= self.spawn_queue[k]:
                    self.spawn_ac(self.routes[k], k)

                    if self.verbose:
                        print(self.on_route)

                self.spawn_queue[k] = random.choices(self.distances, k=1)[0]

    def get_route(self, callsign):
        return self.on_route[callsign]

    def check_state(self):
        return (self.max_ac <= self.existed) and (self.active == 0)
