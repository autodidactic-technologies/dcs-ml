import numpy as np

class Aircraft:
    """ Represents an aircraft in the DCS environment """
    def __init__(self, name, coalition, lat, lon, alt, heading, speed, munitions=None):
        self.name = name
        self.coalition = coalition  # 1 for Allies, 2 for Enemies
        self.lat = lat
        self.lon = lon
        self.alt = alt  # Altitude in meters
        self.heading = heading  # Heading in degrees
        self.speed = speed  # Speed in m/s
        self.munitions = munitions if munitions else {}

    def get_distance(self, other):
        """ Calculates distance to another aircraft using the Haversine formula """
        R = 6371e3  # Earth radius in meters
        phi1, phi2 = np.radians(self.lat), np.radians(other.lat)
        delta_phi = np.radians(other.lat - self.lat)
        delta_lambda = np.radians(other.lon - self.lon)

        a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c  # Distance in meters

    def to_dict(self):
        """ Convert to dictionary for JSON export """
        return {
            "name": self.name,
            "coalition": self.coalition,
            "lat": self.lat,
            "lon": self.lon,
            "alt": self.alt,
            "heading": self.heading,
            "speed": self.speed,
            "munitions": self.munitions
        }


class GroundUnit:
    """ Represents a ground unit in DCS (e.g., SAM sites, tanks) """
    def __init__(self, name, coalition, lat, lon, heading):
        self.name = name
        self.coalition = coalition  # 1 for Allies, 2 for Enemies
        self.lat = lat
        self.lon = lon
        self.heading = heading  # Heading in degrees

    def to_dict(self):
        """ Convert to dictionary for JSON export """
        return {
            "name": self.name,
            "coalition": self.coalition,
            "lat": self.lat,
            "lon": self.lon,
            "heading": self.heading
        }


class Observer:
    """ Observer class that collects aircraft and ground unit data and constructs state representation """
    def __init__(self, ego_aircraft, enemies, allies, ground_units=None):
        self.ego_aircraft = ego_aircraft
        self.enemies = sorted(enemies, key=lambda e: ego_aircraft.get_distance(e))[:3]  # Closest 3 enemies
        self.allies = sorted(allies, key=lambda a: ego_aircraft.get_distance(a))[:3]  # Closest 3 allies
        self.ground_units = ground_units if ground_units else []

    def get_state(self):
        """ Constructs the state representation as a flattened NumPy array """
        state = [
            self.ego_aircraft.speed,
            self.ego_aircraft.alt,
            self.ego_aircraft.heading,
            sum(self.ego_aircraft.munitions.values())  # Total number of munitions
        ]

        # Add enemy aircraft states (distance, speed, heading, coalition)
        for enemy in self.enemies:
            state += [self.ego_aircraft.get_distance(enemy), enemy.speed, enemy.heading, enemy.coalition]

        # Add ally aircraft states
        for ally in self.allies:
            state += [self.ego_aircraft.get_distance(ally), ally.speed, ally.heading, ally.coalition]

        return np.array(state, dtype=np.float32)

    def to_dict(self):
        """ Convert all objects to JSON format """
        return {
            "ego": self.ego_aircraft.to_dict(),
            "enemies": [e.to_dict() for e in self.enemies],
            "allies": [a.to_dict() for a in self.allies],
            "ground_units": [g.to_dict() for g in self.ground_units]
        }




# Create own aircraft
ego = Aircraft(name="Su-27", coalition=1, lat=41.7, lon=41.9, alt=5000, heading=90, speed=250, munitions={"R-60": 2})

# Create enemies
enemy1 = Aircraft(name="F-16", coalition=2, lat=41.6, lon=41.8, alt=5500, heading=180, speed=300)
enemy2 = Aircraft(name="F-15", coalition=2, lat=41.5, lon=41.7, alt=5200, heading=270, speed=320)
enemy3 = Aircraft(name="F/A-18", coalition=2, lat=41.8, lon=42.0, alt=4800, heading=60, speed=290)

# Create allies
ally1 = Aircraft(name="MiG-29", coalition=1, lat=41.9, lon=42.1, alt=5000, heading=30, speed=270)
ally2 = Aircraft(name="Su-30", coalition=1, lat=41.7, lon=41.8, alt=5100, heading=120, speed=260)
ally3 = Aircraft(name="MiG-21", coalition=1, lat=41.6, lon=41.6, alt=4950, heading=200, speed=250)

# Create ground units
sam_site = GroundUnit(name="SAM SA-10", coalition=2, lat=41.4, lon=41.9, heading=0)
tank = GroundUnit(name="M1 Abrams", coalition=2, lat=41.3, lon=42.0, heading=180)

# Create Observer
observer = Observer(ego, [enemy1, enemy2, enemy3], [ally1, ally2, ally3], [sam_site, tank])

# Get state as a NumPy array
state = observer.get_state()
print("State Shape:", state.shape)
print("State Vector:", state)

# Get JSON formatted output
json_data = observer.to_dict()
import json
print(json.dumps(json_data, indent=2))

