# SPDX-License-Identifier: BSD-3-Clause

# flake8: noqa F401
from collections.abc import Callable
from random import random

import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw

from vendeeglobe import (
    Checkpoint,
    Heading,
    Instructions,
    Location,
    Vector,
    config,
)
from vendeeglobe.utils import distance_on_surface, wind_force


SCALE = 4
WINDOW = 1
EARTH_RADIUS = 6_371.00
DEG_TO_KM = 111
SHIFT = -30
SIZE = (360 * SCALE, 360 * SCALE)

np.set_printoptions(linewidth=320)

def make_image(courses: list[Checkpoint]) -> np.ndarray:
    img = Image.new('F', SIZE, color=0.0)
    drw = ImageDraw.Draw(img)
    distance = 0.0
    for a, b in zip(courses[0:], courses[1:]):
        a_lat = np.radians(a.latitude)
        b_lat = np.radians(b.latitude)
        a_lon = np.radians(a.longitude)
        b_lon = np.radians(b.longitude)

        _xa = np.cos(a_lat) * np.cos(a_lon)
        _xb = np.cos(b_lat) * np.cos(b_lon)
        _ya = np.cos(a_lat) * np.sin(a_lon)
        _yb = np.cos(b_lat) * np.sin(b_lon)
        _za = np.sin(a_lat)
        _zb = np.sin(b_lat)

        diff_radius = (b.radius - a.radius) / DEG_TO_KM * SCALE
        ab_distance = distance_on_surface(a.longitude, a.latitude, b.longitude, b.latitude)

        d_lat = b_lat - a_lat
        d_lon = b_lon - b_lon

        calc_a = np.sin(d_lat / 2) ** 2 + np.cos(a_lat) * np.cos(b_lat) * np.sin(d_lon / 2) ** 2
        calc_b = 2 * np.atan2(np.sqrt(calc_a), np.sqrt(1 - calc_a))

        rest_distance = ab_distance
        while rest_distance > 0.0:
            ratio = rest_distance / ab_distance
            radius = a.radius  / DEG_TO_KM * SCALE + diff_radius * (1 - ratio)

            _a = np.sin(ratio * calc_b) / np.sin(calc_b)
            _b = np.sin((1 - ratio) * calc_b) / np.sin(calc_b)

            x = _a * _xa + _b * _xb
            y = _a * _ya + _b * _yb
            z = _a * _za + _b * _zb

            lat = np.atan2(z, np.sqrt(x**2 + y**2))
            lon = np.atan2(y, x)

            # mx = np.pi + lon
            # my = np.pi - np.log(np.tan(np.pi / 4 + lat / 2))
            v = SCALE * ((np.degrees(np.array([lon, lat])) + [SHIFT, 180]) % 360)
            # print(np.degrees(lat), np.degrees(lon), v)

            drw.circle(v, fill=distance, radius=radius)

            distancelet = min(rest_distance, 10.0)
            distance += distancelet
            rest_distance -= distancelet

    return np.asarray(img)


class Bot:
    """
    This is the ship-controlling bot that will be instantiated for the competition.
    """

    def __init__(self):
        self.team = "Firemark"  # This is your team name
        self.course = [
            Checkpoint(
                latitude=config.start.latitude,
                longitude=config.start.longitude,
                radius=22.0,
            ),
            Checkpoint(44.076538, -18.292936, radius=250.0),
            Checkpoint(29.481453, -25.596465, radius=350.0),
            Checkpoint(8.944915, -27.085383, radius=850.0),
            Checkpoint(-54.721834, -47.141871, radius=850.0),
            Checkpoint(-59.969150, -84.645305, radius=550.0),
            Checkpoint(-8.329906, -105.442150, radius=800.0),
            Checkpoint(2.806318, -168.943864, radius=1800.0),
            Checkpoint(-11.105436, 171.859256, radius=400.0),
            Checkpoint(-31.290107, 169.046756, radius=450.0),
            Checkpoint(-51.022261, 149.886600, radius=450.0),
            Checkpoint(-15.668984, 77.674694, radius=1000.0),
        ]

        self.map = make_image(self.course)
        self.escape = 0.0
        self.old_m = [0, 0]

    def run(
        self,
        t: float,
        dt: float,
        longitude: float,
        latitude: float,
        heading: float,
        speed: float,
        vector: np.ndarray,
        forecast: Callable,
        world_map: Callable,
    ) -> Instructions:
        """
        This is the method that will be called at every time step to get the
        instructions for the ship.

        Parameters
        ----------
        t:
            The current time in hours.
        dt:
            The time step in hours.
        longitude:
            The current longitude of the ship.
        latitude:
            The current latitude of the ship.
        heading:
            The current heading of the ship.
        speed:
            The current speed of the ship.
        vector:
            The current heading of the ship, expressed as a vector.
        forecast:
            Method to query the weather forecast for the next 5 days.
            Example:
            current_position_forecast = forecast(
                latitudes=latitude, longitudes=longitude, times=0
            )
        world_map:
            Method to query map of the world: 1 for sea, 0 for land.
            Example:
            current_position_terrain = world_map(
                latitudes=latitude, longitudes=longitude
            )

        Returns
        -------
        instructions:
            A set of instructions for the ship. This can be:
            - a Location to go to
            - a Heading to point to
            - a Vector to follow
            - a number of degrees to turn Left
            - a number of degrees to turn Right

            Optionally, a sail value between 0 and 1 can be set.
        """
        # Initialize the instructions
        instructions = Instructions()
        instructions.sail = 1.0

        mx = int(SCALE * ((longitude + SHIFT) % 360))
        my = int(SCALE * ((latitude + 180) % 360))
        m = [longitude, latitude]

        if self.escape > 0.0:
            self.escape -= dt
            self.old_m = m
            return instructions

        if distance_on_surface(*m, *self.old_m) <= 0.1:
            # Try to escape
            instructions.location = Location(
                latitude=latitude + 2 * random() - 1.0,
                longitude=longitude + 2 * random() - 1.0,
            )
            self.escape = 0.5
            self.old_m = m
            return instructions

        self.old_m = m
        lon_range = np.arange(-WINDOW, WINDOW, 1/SCALE) + longitude

        w = self.map.shape[0]
        h = self.map.shape[1]
        part_map_x = slice(np.mod(mx-WINDOW*SCALE, w), np.mod(mx+WINDOW*SCALE, w))
        part_map_y = slice(np.mod(my-WINDOW*SCALE, h), np.mod(my+WINDOW*SCALE, h))

        if part_map_x.start > part_map_x.stop:
            part_map_x = slice(part_map_x.stop, part_map_x.start)
        if part_map_y.start > part_map_y.stop:
            part_map_y = slice(part_map_y.stop, part_map_y.start)

        part_map = self.map[part_map_y, part_map_x].copy()

        try:
            world = np.zeros(part_map.shape)
            for col in range(WINDOW * 2 * SCALE):
                _lat = latitude  + (col/SCALE - WINDOW)
                # print(part_map[col], current_position_terrain)
                # part_map[col] *= current_position_terrain
                world[col] = world_map(latitudes=_lat, longitudes=lon_range)
            part_map *= world
        except (ValueError, IndexError):
            pass
        

        # Filter too big values
        #part_map[part_map > part_map_nonzero_min + 1000.0] = 0.0

        best_y, best_x = np.unravel_index(np.argmax(part_map), part_map.shape)

        gradient_u = np.gradient(part_map, axis=0)
        gradient_v = np.gradient(part_map, axis=1)
        gradient = np.dstack([gradient_u, gradient_v]).astype(np.float64)
        try:
            wind = np.zeros(part_map.shape)
            for col in range(WINDOW * 2 * SCALE):
                _lat = latitude  + (col/SCALE - WINDOW)
                uwind, vwind = forecast(latitudes=_lat, longitudes=lon_range, times=0)
                fort = np.column_stack([uwind, vwind])
                arr = [wind_force(g, f) for f, g in zip(fort, gradient[col])]
                wind[col] = np.linalg.norm(np.array(arr), axis=1)
            mask = part_map > 0
            if np.any(mask):
                part_map_nonzero_min = part_map[mask].min()
                part_map[mask] -= part_map_nonzero_min
            part_map *= (wind / wind.max() * 0.9) + 0.1
        except (ValueError, IndexError):
            pass


        part_map[ndimage.binary_dilation(part_map == 0, iterations=2)] = 0.0


        best_y, best_x = np.unravel_index(np.argmax(part_map), part_map.shape)
        best_lon = longitude + best_x/SCALE - WINDOW
        best_lat = latitude + best_y/SCALE - WINDOW

        instructions.location = Location(
           latitude=best_lat,
           longitude=best_lon,
        )
        return instructions


if __name__ == "__main__":
    b = Bot()
    map = (255 * b.map / b.map.max()).astype(np.uint8)
    img = Image.fromarray(map, 'L')
    img.save(open("wtf.png", "bw"))