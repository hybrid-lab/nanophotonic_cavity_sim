import numpy as np


class Mirror:
    def __init__(
        self,
        parameters,
        context,
        n_mirror
    ):
        
        self.parameters = parameters
        self.context = context
        self.n_mirror = n_mirror

        self.mirror_layout = self.generate_mirrors_layout()

    def generate_mirrors_layout(self):

        # ── Mirror templates ──────────────────────────────────────────────
        mirror_lat = np.full(self.n_mirror, self.parameters["lattice"])
        mirror_par = np.tile(self.parameters["hole_params"], (self.n_mirror, 1))
        mirror_pos = np.arange(0, self.n_mirror) * self.parameters["lattice"]

        mirror_layout = {
            "positions": mirror_pos,
            "lattice": mirror_lat,
            "hole_params": mirror_par,
        }

        return mirror_layout



