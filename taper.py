import numpy as np


class Taper:
    def __init__(
        self,
        parameters_mirror,
        parameters_taper,
        context,
        n_taper
    ):
        
        self.parameters_mirror = parameters_mirror
        self.parameters_taper = parameters_taper
        self.context = context
        self.n_taper = n_taper

        self.taper_layout = self.generate_taper_layout()

    def generate_taper_layout(self):


        t = np.linspace(0, 1, self.n_taper)

        taper_lat = self.parameters_mirror["lattice"] + t * (self.parameters_taper["lattice"] - self.parameters_mirror["lattice"])
        taper_par = np.array([self.parameters_mirror["hole_params"] + ti * (self.parameters_taper["hole_params"] - self.parameters_mirror["hole_params"]) for ti in t])

        taper_pos = np.zeros(self.n_taper)
        for i in range(1, self.n_taper):
            gap = (taper_lat[i] + taper_lat[i - 1]) / 2
            taper_pos[i] = taper_pos[i - 1] + gap

        taper_layout = {
            "positions": taper_pos,
            "lattice": taper_lat,
            "hole_params": taper_par,
        }

        return taper_layout

