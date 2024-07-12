from dataclasses import dataclass
from PIL import Image, ImageDraw

import numpy as np

from vendeeglobe.utils import distance_on_surface
from vendeeglobe import Checkpoint

SHIFT = -30
DEG_TO_KM = 111


@dataclass
class Mapee:
    scale: float
    window: int
    map: np.ndarray

    @classmethod
    def make(cls, courses: list[Checkpoint], scale: float, window: int):
        size = (int(360 * scale), int(360 * scale))
        img = Image.new('F', size, color=0.0)
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

            diff_radius = (b.radius - a.radius) / DEG_TO_KM * scale
            ab_distance = distance_on_surface(a.longitude, a.latitude, b.longitude, b.latitude)

            d_lat = b_lat - a_lat
            d_lon = b_lon - b_lon

            calc_a = np.sin(d_lat / 2) ** 2 + np.cos(a_lat) * np.cos(b_lat) * np.sin(d_lon / 2) ** 2
            calc_b = 2 * np.atan2(np.sqrt(calc_a), np.sqrt(1 - calc_a))

            rest_distance = ab_distance
            while rest_distance > 0.0:
                ratio = rest_distance / ab_distance
                radius = a.radius  / DEG_TO_KM * scale + diff_radius * (1 - ratio)

                _a = np.sin(ratio * calc_b) / np.sin(calc_b)
                _b = np.sin((1 - ratio) * calc_b) / np.sin(calc_b)

                x = _a * _xa + _b * _xb
                y = _a * _ya + _b * _yb
                z = _a * _za + _b * _zb

                lat = np.atan2(z, np.sqrt(x**2 + y**2))
                lon = np.atan2(y, x)

                v = scale * ((np.degrees(np.array([lon, lat])) + [SHIFT, 180]) % 360)

                drw.circle(v, fill=distance, radius=radius)

                distancelet = min(rest_distance, 10.0)
                distance += distancelet
                rest_distance -= distancelet

        return cls(scale, window, np.asarray(img))