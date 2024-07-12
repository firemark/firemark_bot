# SPDX-License-Identifier: BSD-3-Clause

# flake8: noqa F401
from collections.abc import Callable
from functools import partial
from random import random

import numpy as np
from scipy import ndimage

from vendeeglobe import (
    Checkpoint,
    Instructions,
    Location,
    Vector,
    config,
)
from vendeeglobe.utils import distance_on_surface, wind_force, goto

from .mapee import Mapee, SHIFT


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
            # Checkpoint(-8.329906, -105.442150, radius=800.0),
            Checkpoint(2.806318, -168.943864, radius=1800.0),
            Checkpoint(-11.105436, 171.859256, radius=400.0),
            Checkpoint(-31.290107, 169.046756, radius=450.0),
            Checkpoint(-51.022261, 149.886600, radius=450.0),
            Checkpoint(-15.668984, 77.674694, radius=1000.0),
        ]

        self.mapees = [
            Mapee.make(self.course, scale=1.0, window=5),
            Mapee.make(self.course, scale=2.0, window=2),
            Mapee.make(self.course, scale=4.0, window=2),
        ]
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
        # Initialize the instructions
        instructions = Instructions()
        instructions.sail = 1.0
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
        get_vec_from_mapee = partial(self._get_vec_from_mapee, latitude, longitude, world_map, forecast)
        a, b, c = [get_vec_from_mapee(mapee) for mapee in self.mapees]
        v = a + b * 0.5 + c * 0.25
        instructions.vector = Vector(u=v[0], v=v[1])
        return instructions

    def _get_vec_from_mapee(self, latitude: float, longitude: float, world_map, forecast, mapee: Mapee) -> np.ndarray:
        window = mapee.window
        scale = mapee.scale
        lon_range = np.arange(-window, window, 1/scale) + longitude

        mx = int(scale * ((longitude + SHIFT) % 360))
        my = int(scale * ((latitude + 180) % 360))

        w = mapee.map.shape[0]
        h = mapee.map.shape[1]

        def slice_(m_, d_):
            s = slice(
                np.mod(m_-window*scale, d_).astype(int), 
                np.mod(m_+window*scale, d_).astype(int),
            )
            if s.start > s.stop:
                return slice(s.stop, s.start)
            return s

        part_map_x = slice_(mx, w)
        part_map_y = slice_(my, h)
        part_map = mapee.map[part_map_y, part_map_x].copy()

        try:
            world = np.zeros(part_map.shape)
            for col in range(int(window * 2 * scale)):
                _lat = latitude  + (col/scale - window)
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
            for col in range(int(window * 2 * scale)):
                _lat = latitude  + (col/scale - window)
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
        best_lon = longitude + best_x/scale - window
        best_lat = latitude + best_y/scale - window

        bearing = np.radians(goto(
            origin=Location(longitude, latitude),
            to=Location(best_lon, best_lat),
        ))

        return np.array([np.cos(bearing), np.sin(bearing)])


if __name__ == "__main__":
    # For testing generated map only. Go away.
    from PIL import Image
    b = Bot()
    map = b.mapees[0].map
    map = (255 * map / map.max()).astype(np.uint8)
    img = Image.fromarray(map, 'L')
    img.save(open("wtf.png", "bw"))