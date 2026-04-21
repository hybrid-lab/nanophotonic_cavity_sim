import os
import json
import numpy as np
import matplotlib.pyplot as plt

try:
    import gdsfactory as gf
except ImportError:
    gf = None

from defect import Defect
from simulation import Cavity_simulation, _make_serializable
from taper import Taper
from mirror import Mirror
from hole import hole_polygon_2d

class Cavity:
    """Photonic crystal nanobeam cavity assembled from tapers, mirrors, and a defect."""

    def __init__(
        self,
        n_cells=None,
        parameters=None,
        context=None,
    ):
        self.n_cells = n_cells or {
            "N_left_taper":    5,
            "N_left_mirror":  20,
            "N_defect":       20,
            "N_right_mirror": 10,
            "N_right_taper":  10,
        }

        self.parameters = parameters or {
            "parameters_taper_left":    {"lattice": 0.40, "hole_params": np.array([0.12, 0.12])},
            "parameters_mirrors_left":  {"lattice": 0.54, "hole_params": np.array([0.18, 0.18])},
            "parameters_defect":        {"lattice": 0.4, "hole_params": np.array([0.1, 0.1])},
            "parameters_mirrors_right": {"lattice": 0.54, "hole_params": np.array([0.18, 0.18])},
            "parameters_taper_right":   {"lattice": 0.40, "hole_params": np.array([0.12, 0.12])},
        }

        self.context = context or {
            "freq0": 0.38,
            "fwidth": 0.2,
            "thickness": 0.15,
            "width": 0.8,
            "polarization": "TE",
            "medium": "SiN",
            "mode": "dielectric",
            "sidewall_angle": 0,
            "geometry": "circular",
        }

        # Sub-components
        self.taper_left = Taper(
            parameters_mirror=self.parameters["parameters_mirrors_left"],
            parameters_taper=self.parameters["parameters_taper_left"],
            context=self.context,
            n_taper=self.n_cells["N_left_taper"],
        )
        self.mirror_left = Mirror(
            parameters=self.parameters["parameters_mirrors_left"],
            context=self.context,
            n_mirror=self.n_cells["N_left_mirror"],
        )
        self.defect = Defect(
            parameters=self.parameters,
            context=self.context,
            n_defect=self.n_cells["N_defect"],
        )
        self.mirror_right = Mirror(
            parameters=self.parameters["parameters_mirrors_right"],
            context=self.context,
            n_mirror=self.n_cells["N_right_mirror"],
        )
        self.taper_right = Taper(
            parameters_mirror=self.parameters["parameters_mirrors_right"],
            parameters_taper=self.parameters["parameters_taper_right"],
            context=self.context,
            n_taper=self.n_cells["N_right_taper"],
        )

        # Full beam layout
        self.beam_layout = self._generate_beam_layout()

        # Simulation is created but not built until .build_simulation()
        self.simulation = None

    # ──────────────────────────────────────────────────────────────────
    # Beam layout assembly
    # ──────────────────────────────────────────────────────────────────

    def _generate_beam_layout(self):
        """Concatenate sub-component layouts into a single beam.

        All sub-layouts are ordered left-to-right starting from 0 to +x,
        except the defect which spans -x to +x. This method flips and
        shifts to assemble the full cavity.
        """
  
        d_layout = self.defect.defect_layout
        ml_layout = self.mirror_left.mirror_layout
        mr_layout = self.mirror_right.mirror_layout
        tl_layout = self.taper_left.taper_layout
        tr_layout = self.taper_right.taper_layout

        # Defect edges
        pos_defect_left = d_layout["positions"][0]
        pos_defect_right = d_layout["positions"][-1]

        # Mirror anchors (first hole is adjacent to defect)
        start_ml = pos_defect_left - (d_layout["lattice"][0] + ml_layout["lattice"][0]) / 2
        end_ml = start_ml - ml_layout["positions"][-1]

        start_mr = pos_defect_right + (d_layout["lattice"][-1] + mr_layout["lattice"][0]) / 2
        end_mr = start_mr + mr_layout["positions"][-1]

        # Taper anchors
        start_tl = end_ml - (tl_layout["lattice"][0] + ml_layout["lattice"][-1]) / 2
        start_tr = end_mr + (tr_layout["lattice"][0] + mr_layout["lattice"][-1]) / 2

        # Concatenate: left taper | left mirror | defect | right mirror | right taper
        sections_pos = [
            start_tl - tl_layout["positions"][::-1],
            start_ml - ml_layout["positions"][::-1],
            d_layout["positions"],
            start_mr + mr_layout["positions"],
            start_tr + tr_layout["positions"],
        ]
        sections_lat = [
            tl_layout["lattice"][::-1],
            ml_layout["lattice"][::-1],
            d_layout["lattice"],
            mr_layout["lattice"],
            tr_layout["lattice"],
        ]
        sections_params = [
            tl_layout["hole_params"][::-1],
            ml_layout["hole_params"][::-1],
            d_layout["hole_params"],
            mr_layout["hole_params"],
            tr_layout["hole_params"],
        ]

        return {
            "positions": np.concatenate(sections_pos),
            "lattice": np.concatenate(sections_lat),
            "hole_params": np.concatenate(sections_params),
        }
        

    # ──────────────────────────────────────────────────────────────────
    # Section helpers (used by plots)
    # ──────────────────────────────────────────────────────────────────

    def _section_boundaries(self):
        """Return list of (start_idx, end_idx, label, color) tuples."""
        n_lt = self.n_cells["N_left_taper"]
        n_lm = self.n_cells["N_left_mirror"]
        n_d = self.n_cells["N_defect"]
        n_rm = self.n_cells["N_right_mirror"]
        n_rt = self.n_cells["N_right_taper"]

        cumulative = np.cumsum([0, n_lt, n_lm, n_d, n_rm, n_rt])
        labels = ["Left Taper", "Left Mirror", "Defect", "Right Mirror", "Right Taper"]
        colors = ["gold", "royalblue", "crimson", "royalblue", "gold"]

        return [
            (cumulative[i], cumulative[i + 1], labels[i], colors[i])
            for i in range(5)
        ]

    # ──────────────────────────────────────────────────────────────────
    # Simulation lifecycle
    # ──────────────────────────────────────────────────────────────────

    def build_simulation(self, **build_kwargs):
        """Create and build the Simulation (medium fitting, mode solve, etc.)."""
        self.simulation = Cavity_simulation(
            parameters=self.parameters,
            n_cells=self.n_cells,
            context=self.context,
            beam_layout=self.beam_layout,
        ).build(**build_kwargs)
        return self.simulation

    @staticmethod
    def _ensure_numpy_params(parameters):
        for region in parameters.values():
            if isinstance(region, dict) and "hole_params" in region:
                if isinstance(region["hole_params"], list):
                    region["hole_params"] = np.array(region["hole_params"])
        return parameters

    @classmethod
    def from_saved_simulation_file(cls, filepath):
        """Reconstruct a Cavity from a saved simulation HDF5,
        using the attrs embedded in the simulation."""
        sim = Cavity_simulation.from_saved_simulation_file(filepath)
        instance = cls(
            n_cells=sim.n_cells,
            parameters=cls._ensure_numpy_params(sim.parameters),
            context=sim.context,
        )
        instance.simulation = sim
        return instance
    
    # ──────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────

    def get_name(self):
        """Generate a descriptive name string from n_cells and parameters."""
        name = "cavity"
 
        for k, v in self.n_cells.items():
            name += f"__{k}={v}"
        name += "\n"
 
        for region_name, region_params in self.parameters.items():
            for param_key, param_val in region_params.items():
                if isinstance(param_val, (int, float)):
                    name += f"__{region_name}_{param_key}={param_val:.4f}"
                elif isinstance(param_val, (list, np.ndarray)):
                    vals = ",".join(
                        f"{v:.4f}" for v in np.asarray(param_val).flat
                    )
                    name += f"__{region_name}_{param_key}=[{vals}]"
                else:
                    name += f"__{region_name}_{param_key}={param_val}"
            name += "\n"
 
        return name.strip("\n")
    

    # ──────────────────────────────────────────────────────────────────
    # Physical rendering
    # ──────────────────────────────────────────────────────────────────
 
    def render_gdsfactory(self, name=None, layer=(1, 0),
                          hole_layer=(2, 0), label_layer=(1, 2),
                          n_pts=128, waveguide_extension=5.0):
        """Export the cavity as a gdsfactory Component.
 
        The slab is placed on ``layer`` and holes on ``hole_layer``
        so that a boolean subtraction can be applied in the fab flow
        if needed.
 
        Parameters
        ----------
        name : str, optional
            Component name. Defaults to ``self.get_name()``.
        layer : tuple
            GDS layer for the waveguide slab.
        hole_layer : tuple
            GDS layer for the air holes.
        label_layer : tuple
            GDS layer for the text label.
        n_pts : int
            Polygon resolution for curved holes.
        waveguide_extension : float
            Extra slab length (µm) beyond the outermost holes.
 
        Returns
        -------
        gf.Component
        """
        if gf is None:
            raise ImportError(
                "gdsfactory is not installed. "
                "Install with: pip install gdsfactory"
            )
 
        positions = np.asarray(self.beam_layout["positions"])
        lattices = np.asarray(self.beam_layout["lattice"])
        hole_params = np.atleast_2d(self.beam_layout["hole_params"])
        width = self.context["width"]
        geometry = self.context["geometry"]
 
        if name is None:
            name = self.get_name()
 
        c = gf.Component()
 
        # ── Waveguide slab ────────────────────────────────────────
        x_left = positions[0] - lattices[0] / 2 - waveguide_extension
        x_right = positions[-1] + lattices[-1] / 2 + waveguide_extension
 
        c.add_polygon(
            [
                (x_left,   width / 2),
                (x_right,  width / 2),
                (x_right, -width / 2),
                (x_left,  -width / 2),
            ],
            layer=layer,
        )
 
        # ── Air holes ─────────────────────────────────────────────
        for i, x_pos in enumerate(positions):
            poly = hole_polygon_2d(geometry, hole_params[i], n_pts=n_pts)
            poly_shifted = poly + np.array([x_pos, 0.0])
            c.add_polygon(poly_shifted.tolist(), layer=hole_layer)
 
        # ── Ports ─────────────────────────────────────────────────
        c.add_port(
            name="left",
            center=(x_left, 0),
            width=width,
            orientation=180,
            layer=layer,
        )
        c.add_port(
            name="right",
            center=(x_right, 0),
            width=width,
            orientation=0,
            layer=layer,
        )
 
        # ── Label ─────────────────────────────────────────────────
        c.add_label(text=name, layer=label_layer, position=(0, 0))
 
        return c