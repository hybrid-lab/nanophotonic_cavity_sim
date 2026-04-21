import numpy as np


class Defect:
    def __init__(
        self,
        parameters,
        context,
        n_defect
    ):
        
        self.parameters = parameters
        self.context = context
        self.n_defect = n_defect

        self.has_center = (self.n_defect % 2 != 0)
        self.n_half = self.n_defect // 2

        self.central_cavity = 0                         # central space between the two halves of the taper, set to 0 for a continuous taper

        self.defect_layout = self.generate_defect_layout()

    # ──────────────────────────────────────────────────────────────────────
    # Defect Taper computation
    # ──────────────────────────────────────────────────────────────────────

    def defect_function(self, parameter_mirror, parameter_defect, structure_index,
                      cubic_coeff=2, quadratic_coeff=-3, linear_coeff=0, offset_coeff=1):

        mirror_params = np.asarray(parameter_mirror, dtype=float)
        defect_params = np.asarray(parameter_defect, dtype=float)
        
        x = structure_index / (self.n_defect // 2)
        blend = cubic_coeff * x**3 + quadratic_coeff * x**2 + linear_coeff * x + offset_coeff
        
        return mirror_params - (mirror_params - defect_params) * blend

    def _build_defect_half(self, parameters_mirror, parameters_defect): 

        indices = np.arange(1, self.n_half + 1)

        lattices = np.array([self.defect_function(parameters_mirror["lattice"], parameters_defect["lattice"], i) for i in indices])
        params = np.array([self.defect_function(parameters_mirror["hole_params"], parameters_defect["hole_params"], i) for i in indices])
        params = np.atleast_2d(params)

        positions = np.cumsum(lattices) - lattices / 2
        return lattices, positions, params

    # ──────────────────────────────────────────────────────────────────────
    # Layout
    # ──────────────────────────────────────────────────────────────────────

    def generate_defect_layout(self):

        # ── Defect taper halves ───────────────────────────────────────
        r_lat, r_pos, r_par = self._build_defect_half(parameters_mirror=self.parameters["parameters_mirrors_right"], parameters_defect=self.parameters["parameters_defect"])
        l_lat, l_pos, l_par = self._build_defect_half(parameters_mirror=self.parameters["parameters_mirrors_left"], parameters_defect=self.parameters["parameters_defect"])

        if self.has_center:
            r_pos += self.parameters["parameters_defect"]["lattice"] / 2
            l_pos = -(l_pos + self.parameters["parameters_defect"]["lattice"] / 2)

            taper_pos = np.concatenate([l_pos[::-1], [0.0], r_pos])
            taper_lat = np.concatenate([l_lat[::-1], [self.parameters["parameters_defect"]["lattice"]], r_lat])
            taper_par = np.concatenate([l_par[::-1],  np.atleast_2d(self.parameters["parameters_defect"]["hole_params"]), r_par])

        else:
            half_gap = self.central_cavity / 2
            r_pos = r_pos + half_gap
            l_pos = -(l_pos + half_gap)
            taper_pos = np.concatenate([l_pos[::-1], r_pos])
            taper_lat = np.concatenate([l_lat[::-1], r_lat])
            taper_par = np.concatenate([l_par[::-1], r_par])
        
        order = np.argsort(taper_pos)

        defect_layout = {
            "positions": taper_pos[order],
            "lattice": taper_lat[order],
            "hole_params": taper_par[order],
        }

        return defect_layout
