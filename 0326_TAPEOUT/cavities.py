import numpy as np
import matplotlib.pyplot as plt

import tidy3d as td
from tidy3d import web
from tidy3d.plugins.dispersion import FastDispersionFitter
from matplotlib.patches import Rectangle
import os
from tidy3d.plugins.resonance import ResonanceFinder
from scipy.optimize import curve_fit

from tidy3d.plugins.mode import ModeSolver
from tidy3d.plugins.mode.web import run as run_mode_solver

import matplotlib.animation as animation

try:
    import gdsfactory as gf
except ImportError:
    gf = None

from mirrors import mirror_labels, context, ROUNDING, render_mirror

c0 = td.constants.C_0

np.random.seed(15)

# ---------------------------------------------------------------------------
# Keys in mirror_labels that represent hole shape parameters (not period/width).
# ---------------------------------------------------------------------------
_HOLE_PARAM_KEYS = None  # Auto-detected at first use


def _get_hole_param_keys():
    """Return the subset of mirror_labels keys that are hole-shape parameters."""
    global _HOLE_PARAM_KEYS
    if _HOLE_PARAM_KEYS is None:
        _HOLE_PARAM_KEYS = [k for k in mirror_labels if k not in ("period", "width")]
    return _HOLE_PARAM_KEYS


class Cavity(object):
    def __init__(
        self,
        mirrors,
        n_mirrors={
            "left_taper" : 0,
            "left" : 20,
            "defect" : 20,
            "right" : -1,
            "right_taper" : 10,
        },
        shift={
            "left" : 0,
            "defect" : 0,
            "right" : 0,
        },
        context=context,
    ):
        self.mirrors = mirrors
        self.n_mirrors = n_mirrors
        self.shift = shift

        self.context = context

        self.path = None

    def __len__(self):
        return np.sum(list(self.n_mirrors.values()))

    def generate_path(self):
        N_total = len(self)

        mirror_keys = self.mirrors["mirror"].keys()

        self.path = {
            key : np.zeros(N_total)
            for key in mirror_keys
        }
        self.path["name"] = ["",] * N_total

        # defect
        x = np.linspace(-1, 1, self.n_mirrors["defect"], endpoint=True)
        r = x * x   # Quadratic tapering

        defect_start = self.n_mirrors["left_taper"] + max(self.n_mirrors["left"], 0)
        defect_end = N_total - (self.n_mirrors["right_taper"] + max(self.n_mirrors["right"], 0))

        defect_start_clip = max(-self.n_mirrors["left"], 0)
        defect_end_clip = self.n_mirrors["defect"] + min(self.n_mirrors["right"], 0)

        print(defect_start, defect_end, defect_start_clip, defect_end_clip)

        for key in mirror_keys:
            path = r * self.mirrors["mirror"][key] + (1-r) * self.mirrors["defect"][key]
            self.path[key][defect_start:defect_end] = path[defect_start_clip:defect_end_clip]

        self.path["name"][defect_start:defect_end] = ["defect",] * (defect_end - defect_start)

        # mirrors and tapers
        for key in mirror_keys:
            # mirrors
            self.path[key][:defect_start] = self.path[key][defect_start]
            self.path[key][defect_end:] = self.path[key][defect_end-1]

            # tapers
            if self.n_mirrors["left_taper"] > 0:
                self.path[key][:self.n_mirrors["left_taper"]] = np.linspace(
                    self.mirrors["taper"][key],
                    self.path[key][0],
                    self.n_mirrors["left_taper"],
                    endpoint=False
                )
            if self.n_mirrors["right_taper"] > 0:
                self.path[key][-1:-self.n_mirrors["right_taper"]-1:-1] = np.linspace(
                    self.mirrors["taper"][key],
                    self.path[key][-1],
                    self.n_mirrors["right_taper"],
                    endpoint=False
                )
            # FUTURE: provide nonlinear options.

        self.path["name"][:defect_start] = ["mirror",] * defect_start
        self.path["name"][defect_end:] = ["mirror",] * (len(self.path["name"]) - defect_end)

        if self.n_mirrors["left_taper"] > 0:
            self.path["name"][:self.n_mirrors["left_taper"]] = ["taper",] * self.n_mirrors["left_taper"]
        if self.n_mirrors["right_taper"] > 0:
            self.path["name"][-self.n_mirrors["right_taper"]:] = ["taper",] * self.n_mirrors["right_taper"]

        # Build numbered labels for each hole
        labels = [""] * N_total
        idx = 0
        for i in range(self.n_mirrors["left_taper"]):
            labels[idx] = f"taper_l{i+1}_idx{idx}"
            idx += 1
        for i in range(self.n_mirrors["left"]):
            labels[idx] = f"mirror_l{i+1}_idx{idx}"
            idx += 1
        for i in range(self.n_mirrors["defect"]):
            labels[idx] = f"defect_{i+1}_idx{idx}"
            idx += 1
        for i in range(self.n_mirrors["right"]):
            labels[idx] = f"mirror_r{i+1}_idx{idx}"
            idx += 1
        for i in range(self.n_mirrors["right_taper"]):
            labels[idx] = f"taper_r{i+1}_idx{idx}"
            idx += 1
        self.path["label"] = labels

        # If enabled, round all parameters to the nearest specified value (e.g. 1 nm).
        if ROUNDING is not None:
            for key in mirror_keys:
                self.path[key] = np.round(self.path[key] / ROUNDING / 2) * ROUNDING * 2

    def get_mirror(self, index):
        parameters = {key : float(self.path[key][index]) for key in mirror_labels}

        if index == 0:
            parameters["width_left"] = parameters["width"]
        else:
            parameters["width_left"] = float(parameters["width"] + self.path["width"][index-1]) / 2

        if index == len(self) - 1:
            parameters["width_right"] = parameters["width"]
        else:
            parameters["width_right"] = float(parameters["width"] + self.path["width"][index+1]) / 2

        return parameters

    def get_path_x(self, edge=0):
        """
        Integrate the periods, centered around the defect.

        Parameters
        ----------
        edge : int
            0 for center of holes, -1 for left edge of holes, +1 for right edge of holes.
        """
        # x positions of hole centers relative to the cavity start
        x = np.concatenate(([0,], np.cumsum(self.path["period"][1:] + self.path["period"][:-1])/2))

        # print(x)

        di = self.n_mirrors["left_taper"] + self.n_mirrors["left"] + self.n_mirrors["defect"] // 2    # index of the center of the defect region

        # Now find the center of the defect region
        if self.n_mirrors["defect"] % 2:    # odd
            x -= x[di]
        else:                       # even
            x -= (x[di-1] + x[di]) / 2

        return x + edge * (self.path["period"] / 2)

    def _shade_regions(self, ax, x_microns=True):
        if x_microns:
            xs = self.get_path_x(edge=-1) + .01
            xe = self.get_path_x(edge=+1) - .01
        else:
            x = np.arange(len(self.path["name"]))
            xs = x - 0.45
            xe = x + 0.45

        region_boundaries = np.where(np.array(self.path["name"][:-1]) != np.array(self.path["name"][1:]))[0] + 1

        for start, end in zip([0,] + region_boundaries.tolist(), region_boundaries.tolist() + [len(self.path["name"]),]):
            region_type = self.path["name"][start]

            if region_type == "mirror":
                color = "lightcoral"
            elif region_type == "defect":
                color = "sandybrown"
            elif region_type == "taper":
                color = "gold"
            else:
                color = "white"

            ax.axvspan(xs[start], xe[end-1], color=color, zorder=-10, alpha=0.3, linestyle="none")

        ax.set_xlim(xs[0]-1, xe[-1]+1)

    def render_path(self, x_microns=True, render_cavity=True):
        plot_keys = [k for k in self.path.keys() if k not in ("name", "label")]
        M = len(plot_keys) + render_cavity

        if x_microns:
            x = self.get_path_x(edge=0)
        else:
            x = np.arange(len(self.path["name"]))
            render_cavity = False

        fig, axs = plt.subplots(M, 1, figsize=(8,8), sharex=True)

        I = np.arange(len(self.path["name"])) + render_cavity

        if render_cavity:
            self._shade_regions(axs[0], x_microns=x_microns)
            self.render_matplotlib(ax=axs[0])
            axs[0].set_aspect('equal', adjustable='datalim')
            axs[0].set_ylabel("Position $y$ [µm]")

        for i, key in zip(I, plot_keys):
            # if np.isscalar(self.path[key][0]):
            axs[i].scatter(
                x,
                self.path[key],
                marker=".",
                color="k"
            )
            if key in mirror_labels:
                axs[i].set_ylabel(mirror_labels[key])

            self._shade_regions(axs[i], x_microns=x_microns)

            if i == M-1:
                if x_microns:
                    axs[i].set_xlabel("Position $x$ [µm]")
                else:
                    axs[i].set_xlabel("Hole Position $x$")

        plt.tight_layout()
        plt.show()

    def render_matplotlib(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,4))

        x = self.get_path_x(edge=0)

        draw_kwargs = {
            "color" : "k",
            "edgecolor" : "k",
            "linewidth" : .5,
        }

        # Render the holes
        for i in range(len(self)):
            mirror = self.get_mirror(i)

            points = render_mirror(mirror)["polygon"]

            # Fill the polygon black
            ax.fill(points[:,0] + x[i], points[:,1], **draw_kwargs)

        # Add waveguide on either side
        dy=.01
        w = self.path["width"]
        x_left = x[0] - self.path["period"][0] / 2
        x_right = x[-1] + self.path["period"][-1] / 2
        ax.fill([x_left, x_left-10, x_left-10, x_left], np.flip([-w[0]/2, -w[0]/2, w[0]/2, w[0]/2])+dy, **draw_kwargs)
        ax.fill([x_right, x_right+10, x_right+10, x_right], np.flip([-w[-1]/2, -w[-1]/2, w[-1]/2, w[-1]/2])+dy, **draw_kwargs)

    def get_name(self):
        name = "cavity"

        # Parse `mirrors` and `n_mirrors` to generate a descriptive name for the cavity.
        for (k,v) in self.n_mirrors.items():
            name += f"__{k}={v}"
        name += "\n"

        for region_name, region_params in self.mirrors.items():
            for param_key, param_val in region_params.items():
                if isinstance(param_val, float):
                    name += f"__{region_name}_{param_key}={param_val:.4f}"
                else:
                    name += f"__{region_name}_{param_key}={param_val}"
            name += "\n"

        return name.strip("\n")

    def render_gdsfactory(self):
        if gf is None:
            raise ImportError("gdsfactory is not installed. Please install gdsfactory to use this method.")

        x = self.get_path_x(edge=0)

        c = gf.Component(self.get_name())

        # Add all the mirror cells
        for i in range(len(self)):
            mirror = self.get_mirror(i)

            mirror_rendered = render_mirror(mirror)

            c.add_ref(mirror_rendered["cell"]).move((x[i], 0))

        # Define ports on either side of the cavity
        c.add_port(
            name="left",
            center=(x[0] - self.path["period"][0] / 2, 0),
            width=self.path["width"][0],
            orientation=180,
            layer=1,
        )
        c.add_port(
            name="right",
            center=(x[-1] + self.path["period"][-1] / 2, 0),
            width=self.path["width"][-1],
            orientation=0,
            layer=1,
        )

        c.add_label(
            text=self.get_name(),
            layer=(1, 2),
            position=(0,0),
        )

        return c

    # ------------------------------------------------------------------
    # Tidy3D integration: populate Cavity path from a Cavity_Tidy object
    # ------------------------------------------------------------------

    @classmethod
    def from_cavity_tidy(cls, cavity_tidy, context=None):
        """Create a Cavity instance with its path populated from a Cavity_Tidy object.

        This is the primary integration point: take a fully constructed (or
        loaded-from-file) ``Cavity_Tidy`` and produce a ``Cavity`` whose
        ``.path`` and ``.n_mirrors`` are ready for rendering, GDS export, etc.

        Region mapping
        --------------
        Cavity_Tidy concept       →  Cavity region
        ────────────────────────────────────────────
        linear_taper_left holes   →  "taper"  (left)
        mirrors_left holes        →  "mirror" (left)
        tapered_structures        →  "defect"
        mirrors_right holes       →  "mirror" (right)
        linear_taper_right holes  →  "taper"  (right)

        Parameters
        ----------
        cavity_tidy : Cavity_Tidy
            A constructed Cavity_Tidy instance (with .positions, .lattices,
            .params, and region counts already computed).
        context : dict or None, optional
            Context dict passed to the Cavity constructor. If None (default),
            the context is built from the Cavity_Tidy parameters.

        Returns
        -------
        Cavity
            A new Cavity with ``.path`` fully populated.
        """
        if context is None:
            context = {
                "wavelength": cavity_tidy.wl0,
                "thickness": cavity_tidy.thickness,
                "width": cavity_tidy.width,
            }

        instance = cls.__new__(cls)
        instance.context = context
        instance.mirrors = None
        instance.shift = {"left": 0, "defect": 0, "right": 0}
        instance._cavity_tidy = cavity_tidy

        instance._populate_path_from_tidy(cavity_tidy)

        return instance

    def import_from_cavity_tidy(self, cavity_tidy):
        """Populate this Cavity's path from a Cavity_Tidy object (in-place).

        Overwrites any existing ``self.path`` and ``self.n_mirrors``.

        Parameters
        ----------
        cavity_tidy : Cavity_Tidy
            A constructed Cavity_Tidy instance.
        """
        self._cavity_tidy = cavity_tidy
        self._populate_path_from_tidy(cavity_tidy)

    def _populate_path_from_tidy(self, ct):
        """Internal: build self.path, self.mirrors, self.n_mirrors from Cavity_Tidy arrays.

        Cavity_Tidy stores (after generate_beam_layout):
          - ct.positions   : (N,)   hole centre x-positions, sorted left→right
          - ct.lattices    : (N,)   lattice constant per hole
          - ct.params      : (N, P) hole shape semi-axes per hole
          - ct.n_mirrors_left, ct.n_mirrors_right, ct.n_tapered_structures
          - ct.n_linear_taper_left, ct.n_linear_taper_right
          - ct.width       : waveguide width (scalar)

        Cavity.path needs:
          - "period"  : (N,) lattice constant  — direct copy of ct.lattices
          - "width"   : (N,) waveguide width   — constant ct.width
          - hole keys : (N,) hole diameters    — ct.params[:, col]
          - "name"    : (N,) region labels
          - "label"   : (N,) unique hole identifiers

        """
        n_taper_left  = ct.n_linear_taper_left
        n_left        = ct.n_mirrors_left
        n_defect      = ct.n_tapered_structures
        n_right       = ct.n_mirrors_right
        n_taper_right = ct.n_linear_taper_right
        N_total       = n_taper_left + n_left + n_defect + n_right + n_taper_right

        hole_keys = _get_hole_param_keys()   # e.g. ["hx", "hy"]
        all_keys  = ["period", "width"] + hole_keys

        # Sanity checks
        assert len(ct.positions) == N_total, (
            f"Cavity_Tidy has {len(ct.positions)} holes but expected "
            f"n_taper_left({n_taper_left}) + n_left({n_left}) + "
            f"n_defect({n_defect}) + n_right({n_right}) + "
            f"n_taper_right({n_taper_right}) = {N_total}"
        )
        assert ct.params.shape[1] == len(hole_keys), (
            f"Cavity_Tidy params has {ct.params.shape[1]} columns but "
            f"mirror_labels expects {len(hole_keys)} hole keys: {hole_keys}"
        )

        # --- Build path arrays directly from the sorted Cavity_Tidy data ---
        self.path = {}
        self.path["period"] = ct.lattices.copy()
        self.path["width"]  = np.full(N_total, float(ct.width))
        for col, key in enumerate(hole_keys):
            self.path[key] = ct.params[:, col].copy()   # semi-axis → diameter

        # --- Region names and labels ---
        names  = [""] * N_total
        labels = [""] * N_total
        idx = 0

        for i in range(n_taper_left):
            names[idx]  = "taper"
            labels[idx] = f"taper_l{i+1}_idx{idx}"
            idx += 1
        for i in range(n_left):
            names[idx]  = "mirror"
            labels[idx] = f"mirror_l{i+1}_idx{idx}"
            idx += 1
        for i in range(n_defect):
            names[idx]  = "defect"
            labels[idx] = f"defect_{i+1}_idx{idx}"
            idx += 1
        for i in range(n_right):
            names[idx]  = "mirror"
            labels[idx] = f"mirror_r{i+1}_idx{idx}"
            idx += 1
        for i in range(n_taper_right):
            names[idx]  = "taper"
            labels[idx] = f"taper_r{i+1}_idx{idx}"
            idx += 1

        self.path["name"]  = names
        self.path["label"] = labels

        # --- n_mirrors dict ---
        
        self.n_mirrors = {
            "left_taper":  n_taper_left,
            "left":        n_left,
            "defect":      n_defect,
            "right":       n_right,
            "right_taper": n_taper_right,
        }
        

        hole_keys_only = _get_hole_param_keys()

        def _mirror_dict_from_tidy(lattice, hole_params_semi):
            """Build a Cavity-style parameter dict from Cavity_Tidy values."""
            d = {"period": float(lattice), "width": float(ct.width)}
            for col, key in enumerate(hole_keys_only):
                d[key] = float(hole_params_semi[col])  # semi-axis → diameter
            return d

        self.mirrors = {
            "mirror_left":  _mirror_dict_from_tidy(ct.lattice_mirrors_left,  ct.parameters_mirrors_left),
            "mirror_right": _mirror_dict_from_tidy(ct.lattice_mirrors_right, ct.parameters_mirrors_right),
            "defect":       _mirror_dict_from_tidy(ct.lattice_taper_center,  ct.parameters_taper_center),
        }
        if n_taper_left > 0:
            self.mirrors["taper_left"] = _mirror_dict_from_tidy(
                ct.lattice_linear_taper_left, ct.parameters_linear_taper_left
            )
        if n_taper_right > 0:
            self.mirrors["taper_right"] = _mirror_dict_from_tidy(
                ct.lattice_linear_taper_right, ct.parameters_linear_taper_right
            )

class Cavity_Tidy(object):

    @staticmethod
    def _parse_params(params):
        """Parse a parameter specification into (lattice, hole_params).

        Parameters
        ----------
        params : dict
            Dictionary with keys ``"lattice"`` (float) and
            ``"hole_params"`` (list/array of hole shape values).
            Example: ``{"lattice": 0.37, "hole_params": [0.07, 0.12]}``

        Returns
        -------
        lattice : float
            Lattice constant.
        hole_params : np.ndarray
            1-D array of hole shape parameters.
        """
        if not isinstance(params, dict):
            raise TypeError(
                f"Expected a dict with keys 'lattice' and 'hole_params', "
                f"got {type(params).__name__}. "
                f"Example: {{'lattice': 0.37, 'hole_params': [0.07, 0.12]}}"
            )
        lattice = float(params["lattice"])
        hole_params = np.atleast_1d(np.asarray(params["hole_params"], dtype=float))
        return lattice, hole_params

    def __init__(self, freq0, fwidth, polarization, thickness, width, nanobeam_medium,
                 geometry, sidewall_angle,
                 n_mirrors_left,
                 n_mirrors_right,
                 n_tapered_structures,
                 n_linear_taper_left,
                 n_linear_taper_right,
                 parameters_mirrors_left,
                 parameters_mirrors_right,
                 parameters_taper_center,
                 parameters_linear_taper_left,
                 parameters_linear_taper_right,
                 central_cavity=0,
                 t_start = 1e-8,
                 run_time = 5e-12,
                 mode = 'dielectric'
                 ):
        """
        Parameters
        ----------
        parameters_mirrors_left : dict
            ``{"lattice": float, "hole_params": [hx, hy, ...]}``
        parameters_mirrors_right : dict
            Same format as above.
        parameters_taper_center : dict
            Same format as above.
        parameters_linear_taper_left : dict
            Same format as above.
        parameters_linear_taper_right : dict
            Same format as above.
        """

        self.thickness = thickness
        self.width = width
        self.geometry = geometry
        self.sidewall_angle = sidewall_angle

        self.n_mirrors_left = n_mirrors_left
        self.n_mirrors_right = n_mirrors_right
        self.n_tapered_structures = n_tapered_structures
        self.n_holes = n_mirrors_left + n_mirrors_right + n_tapered_structures + n_linear_taper_left + n_linear_taper_right
        self.n_linear_taper_left = n_linear_taper_left
        self.n_linear_taper_right = n_linear_taper_right

        # Parse dict parameters → (lattice, hole_params) pairs
        self.lattice_mirrors_left, self.parameters_mirrors_left = \
            self._parse_params(parameters_mirrors_left)
        self.lattice_mirrors_right, self.parameters_mirrors_right = \
            self._parse_params(parameters_mirrors_right)
        self.lattice_taper_center, self.parameters_taper_center = \
            self._parse_params(parameters_taper_center)
        self.lattice_linear_taper_left, self.parameters_linear_taper_left = \
            self._parse_params(parameters_linear_taper_left)
        self.lattice_linear_taper_right, self.parameters_linear_taper_right = \
            self._parse_params(parameters_linear_taper_right)

        self.central_cavity = central_cavity

        # Source parameters
        self.freq0 = freq0
        self.fwidth = fwidth
        self.wl0 = c0 / freq0
        self.wlM = c0 / (freq0 - self.fwidth/2)
        self.polarization = polarization
        self.t_start = t_start
        self.run_time = run_time
        self.mode = mode

        # Material
        self.nanobeam_medium = (
            self.material(nanobeam_medium)
            if isinstance(nanobeam_medium, (int, float))
            else self.material_from_file(nanobeam_medium)
        )

        # Beam layout: dict with "positions", "lattice", "hole_params"
        self.beam_layout = self.generate_beam_layout()
        self.positions = self.beam_layout["positions"]
        self.lattices = self.beam_layout["lattice"]
        self.params = self.beam_layout["hole_params"]

        self.nanobeam_length = self.positions[-1] + self.lattice_mirrors_right/2 - self.positions[0] - self.lattice_mirrors_left/2
        self.taper_start = self.positions[self.n_mirrors_left - 1] - self.lattice_mirrors_left/2
        self.taper_end = self.positions[-self.n_mirrors_right] + self.lattice_mirrors_right/2
        self.taper_center_x = (self.taper_start + self.taper_end) / 2
        self.taper_length = self.taper_end - self.taper_start

        # Simulation domain (depends on positions)
        self.sim_center, self.sim_size = self.simulation_center_and_size()

        # Source and monitors (depend on sim_size)
        self.sources = self.define_source()
        self.monitors = self.define_monitors(n_point_monitors=5, deviation=(0.2, 0, 0))

        # Structures
        self.nanobeam = self.generate_nanobeam_structure()

        # Full simulation
        self.sim = self.create_simulation()
        print('SIMULATION CREATED')

        self.sim_data = None
        self.task_id = None
        self.resonance_df = None
        self.combined_signal = None
        self.resonant_frequency = None
        self.Q = None
        self.energy_density = None
        self.eps = None
        self.Vmode = None
        self.Q_directional = None

        self.n_eff = None
        self.k_wg = None
        self.a_last = None

    # ──────────────────────────────────────────────────────────────────────
    # Create Instance if I have already simulated the cavity
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, filepath):
        """Create a Cavity instance from a previously saved simulation HDF5 file.

        Parameters
        ----------
        filepath : str
            Path to the .hdf5 file saved by a previous run.
        """
        sim_data = td.SimulationData.from_file(filepath)
        sim = sim_data.simulation

        # Extract source parameters
        source = sim.sources[0]
        freq0 = source.source_time.freq0
        fwidth = source.source_time.fwidth
        polarization = source.polarization

        # Create instance without running __init__
        instance = cls.__new__(cls)

        instance.freq0 = freq0
        instance.fwidth = fwidth
        instance.wl0 = c0 / freq0
        instance.wlM = c0 / (freq0 - fwidth / 2)
        instance.polarization = polarization
        instance.run_time = sim.run_time

        # Simulation objects
        instance.sim = sim
        instance.sim_data = sim_data
        instance.sim_center = sim.center
        instance.sim_size = sim.size

        # Analysis results (not yet computed)
        instance.task_id = None
        instance.resonance_df = None
        instance.combined_signal = None
        instance.resonant_frequency = None
        instance.Q = None
        instance.energy_density = None
        instance.eps = None
        instance.Vmode = None
        instance.Q_directional = None

        print(f"Loaded simulation from: {filepath}")
        print(f"  freq0 = {freq0:.4e} Hz, fwidth = {fwidth:.4e} Hz")
        print(f"  polarization = {polarization}")
        print(f"  monitors: {list(sim_data.monitor_data.keys())}")

        return instance

    def save_params(self, directory, save_name):
        """Save design parameters as a JSON file alongside the simulation."""
        import json
        params = {
            'n_mirrors_left': self.n_mirrors_left,
            'n_mirrors_right': self.n_mirrors_right,
            'n_tapered_structures': self.n_tapered_structures,
            'n_linear_taper_left': self.n_linear_taper_left,
            'n_linear_taper_right': self.n_linear_taper_right,
            'parameters_mirrors_left': {
                'lattice': self.lattice_mirrors_left,
                'hole_params': self.parameters_mirrors_left.tolist(),
            },
            'parameters_mirrors_right': {
                'lattice': self.lattice_mirrors_right,
                'hole_params': self.parameters_mirrors_right.tolist(),
            },
            'parameters_taper_center': {
                'lattice': self.lattice_taper_center,
                'hole_params': self.parameters_taper_center.tolist(),
            },
            'parameters_linear_taper_left': {
                'lattice': self.lattice_linear_taper_left,
                'hole_params': self.parameters_linear_taper_left.tolist(),
            },
            'parameters_linear_taper_right': {
                'lattice': self.lattice_linear_taper_right,
                'hole_params': self.parameters_linear_taper_right.tolist(),
            },
            'thickness': self.thickness,
            'width': self.width,
            'geometry': self.geometry,
            'sidewall_angle': self.sidewall_angle,
        }
        path = os.path.join(directory, f"{save_name}_params.json")
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"Parameters saved: {path}")

    # ──────────────────────────────────────────────────────────────────────
    # Simulation domain
    # ──────────────────────────────────────────────────────────────────────

    def simulation_center_and_size(self):
        """Compute simulation center and size from the actual hole positions."""
        padding = 2 * self.wlM

        xmin = self.positions[0] - self.lattice_linear_taper_left - padding
        xmax = self.positions[-1] + self.lattice_linear_taper_right + padding

        sim_center = (
            (xmin + xmax) / 2,
            0.0,
            0.0,
        )
        sim_size = (
            xmax - xmin,
            self.width + 2 * padding,
            self.thickness + 2 * padding,
        )
        return sim_center, sim_size

    # ──────────────────────────────────────────────────────────────────────
    # Materials
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def material(index):
        """Non-dispersive medium from refractive index."""
        return td.Medium(permittivity=float(index)**2)

    @staticmethod
    def material_from_file(filename, plot=False):
        """Fit a dispersive medium from a tab-separated (wavelength, n) file."""
        fitter = FastDispersionFitter.from_file(
            filename, skiprows=1, delimiter="\t", usecols=(0, 1),
        )
        medium, rms = fitter.fit()
        if plot:
            fitter.plot(medium)
            plt.show()
        return medium

    # ──────────────────────────────────────────────────────────────────────
    # Symmetry
    # ──────────────────────────────────────────────────────────────────────

    def get_symmetry(self):
        SYMMETRY_MAP = {
            "Ex": [-1,  1,  1],
            "Ey": [ 1, -1,  1],
            "Ez": [ 1,  1, -1],
            "Hx": [ 1, -1, -1],
            "Hy": [-1,  1, -1],
            "Hz": [-1, -1,  1],
        }
        try:
            return list(SYMMETRY_MAP[self.polarization])
        except KeyError:
            raise ValueError(f"Unknown polarization '{self.polarization}'. Must be one of {list(SYMMETRY_MAP)}")

    # ──────────────────────────────────────────────────────────────────────
    # Hole geometry
    # ──────────────────────────────────────────────────────────────────────

    def hole_geometry(self, geometry, hole_center, params):
        """Create a single hole geometry at the given center."""
        if geometry == "square":
            return td.Box(center=hole_center, size=(params[0], params[1], self.thickness))

        if geometry == "ellipse":
            theta = np.linspace(0, 2 * np.pi, 128, endpoint=False)
            x = params[0]/2 * np.cos(theta) + hole_center[0]
            y = params[1]/2 * np.sin(theta) + hole_center[1]
            return td.PolySlab(
                vertices=np.column_stack([x, y]),
                slab_bounds=(-self.thickness / 2 + hole_center[2],
                             self.thickness / 2 + hole_center[2]),
                axis=2,
            )
        raise ValueError(f"Unknown geometry: '{geometry}'")

    # ──────────────────────────────────────────────────────────────────────
    # Source & Monitors
    # ──────────────────────────────────────────────────────────────────────

    def define_source(self):

        if self.n_tapered_structures % 2 != 0 and self.mode == 'dielectric':
            source_center = (self.lattice_taper_center/2, 0, 0)
        elif self.n_tapered_structures % 2 == 0 and self.mode == 'dielectric':
            source_center = (0,0,0)
        elif self.n_tapered_structures % 2 != 0 and self.mode == 'air':
            source_center = (0,0,0)
        elif self.n_tapered_structures % 2 == 0 and self.mode == 'air':
            source_center = (self.lattice_taper_center/2, 0, 0)


        return td.PointDipole(
            center=source_center,
            name="Point_Dipole_Source",
            polarization=self.polarization,
            source_time=td.GaussianPulse(
                freq0=self.freq0, fwidth=self.fwidth,
                phase=2 * np.pi * np.random.random(),
            ),
        )

    def define_monitors(self, n_point_monitors, deviation):
        monitors = []

        # Random point monitors
        positions = np.random.random((n_point_monitors, 3)) * np.array(deviation)
        for i in range(n_point_monitors):
            monitors.append(
                td.FieldTimeMonitor(
                    center=tuple(positions[i]),
                    name=f"Point_Monitor_{i}",
                    start=0, size=(0, 0, 0), interval=1,
                )
            )

        start = self.run_time - 1 / self.freq0

        # 3D field monitor
        mon_size_3d = tuple(np.array(self.sim_size) - 2 * self.wlM)
        monitors.append(
            td.FieldTimeMonitor(
                center=self.sim_center, size=mon_size_3d, start=start,
                name="Field_Time_Monitor",
                interval_space=(5, 5, 5), interval=10,
            )
        )

        # 2D field profile
        for axis in ("x", "y", "z"):
            center = list(self.sim_center)
            size = [td.inf, td.inf, td.inf]
            idx = "xyz".index(axis)
            center[idx] = 0.0
            size[idx] = 0
            monitors.append(
                td.FieldTimeMonitor(
                    center=tuple(center),
                    size=tuple(size),
                    start=start,
                    name=f"Field_Profile_Monitor_{axis}",
                )
            )

        # Six planar flux monitors
        axis_labels = ["x", "y", "z"]
        for axis_idx in range(3):
            for sign in (-1, +1):
                mon_center = list(self.sim_center)
                mon_center[axis_idx] += sign * (self.sim_size[axis_idx]/2 - self.wlM)
                mon_size = list(self.sim_size)
                mon_size[axis_idx] = 0.0
                name_prefix = "-" if sign == -1 else "+"
                monitors.append(
                    td.FluxTimeMonitor(
                        center=tuple(mon_center), size=tuple(mon_size),
                        start=start, name=f"{name_prefix}{axis_labels[axis_idx]}",
                    )
                )

        return monitors

    def mode_solve(self, num_modes, plot=False):

        mode_size = [0, self.sim_size[1]-self.wlM, self.sim_size[2]-self.wlM]

        if self.n_tapered_structures % 2 != 0:
            mode_center = (self.lattice_taper_center/2, 0, 0)
        else:
            mode_center = (0,0,0)

        mode_plane = td.Box(center=mode_center, size=mode_size)

        mode_spec = td.ModeSpec(num_modes=num_modes, target_neff=self.nanobeam_medium.n_cfl)
        mode_solve = ModeSolver(simulation=self.sim, 
                                plane=mode_plane, 
                                mode_spec=mode_spec, 
                                freqs=[self.freq0])
        mode_data = mode_solve.solve()

        if plot==True:
            mode_solve.plot()
            plt.show()
            print(mode_data.to_dataframe())    

        self.n_eff = mode_data.n_eff.values.flatten() 
        self.k_wg = 2*np.pi*self.n_eff*self.freq0/c0
        self.a_last = self.wl0/(2 * self.n_eff)

        print('n_eff: ', self.n_eff)
        print('k_wg: ', self.k_wg)
        print('a_last: ', self.a_last)


    # ──────────────────────────────────────────────────────────────────────
    # Defect Taper computation
    # ──────────────────────────────────────────────────────────────────────

    def param_tapered(self, mirror_params, defect_params, structure_index, n_tapered_structures,
                      cubic_coeff=2, quadratic_coeff=-3, linear_coeff=0, offset_coeff=1):
        """Hermite smooth-step taper. At x=0 → defect_params, at x=1 → mirror_params."""
        mirror_params = np.asarray(mirror_params, dtype=float)
        defect_params = np.asarray(defect_params, dtype=float)
        x = structure_index / (n_tapered_structures // 2)
        blend = cubic_coeff * x**3 + quadratic_coeff * x**2 + linear_coeff * x + offset_coeff
        return mirror_params - (mirror_params - defect_params) * blend

    def _build_taper_half(self, mirror_lattice, center_lattice, mirror_params, center_params):
        """Build one half of the taper. Returns (lattices, positions, params)."""
        n_half = self.n_tapered_structures // 2

        indices = np.arange(1, n_half + 1)

        lattices = np.array([
            self.param_tapered(mirror_lattice, center_lattice, i, self.n_tapered_structures)
            for i in indices
        ])
        params = np.array([
            self.param_tapered(mirror_params, center_params, i, self.n_tapered_structures)
            for i in indices
        ])
        params = np.atleast_2d(params)

        positions = np.cumsum(lattices) - lattices / 2
        return lattices, positions, params

    # ──────────────────────────────────────────────────────────────────────
    # Beam layout
    # ──────────────────────────────────────────────────────────────────────

    def generate_beam_layout(self):
        """
        Compute positions, lattice constants, and hole parameters for every hole.

        Layout order (left to right):
            linear_taper_left | mirrors_left | defect_taper | mirrors_right | linear_taper_right

        The linear tapers interpolate linearly from the outermost mirror
        parameters toward the linear taper target parameters specified in
        the constructor (``parameters_linear_taper_left/right``).

        Returns
        -------
        dict
            "positions"   : (N,) array of hole centre x-positions, sorted.
            "lattice"     : (N,) array of lattice constant per hole.
            "hole_params" : (N, P) array of hole shape parameters per hole.
        """
        # ── Mirror templates ──────────────────────────────────────────────
        mirror_right_lat = np.full(self.n_mirrors_right, self.lattice_mirrors_right)
        mirror_left_lat = np.full(self.n_mirrors_left, self.lattice_mirrors_left)
        mirror_right_par = np.tile(self.parameters_mirrors_right, (self.n_mirrors_right, 1))
        mirror_left_par = np.tile(self.parameters_mirrors_left, (self.n_mirrors_left, 1))

        # ── No defect taper case ──────────────────────────────────────────
        if self.n_tapered_structures == 0:
            mirror_right_pos = np.arange(1, self.n_mirrors_right + 1) * self.lattice_mirrors_right
            mirror_left_pos = np.arange(1, self.n_mirrors_left + 1) * self.lattice_mirrors_left

            core_pos = np.concatenate([-mirror_left_pos[::-1], mirror_right_pos])
            core_lat = np.concatenate([mirror_left_lat[::-1], mirror_right_lat])
            core_par = np.concatenate([mirror_left_par[::-1], mirror_right_par])
            order = np.argsort(core_pos)
            core_pos, core_lat, core_par = core_pos[order], core_lat[order], core_par[order]

        else:
            # ── Defect taper halves ───────────────────────────────────────
            has_center = (self.n_tapered_structures % 2 != 0)

            r_lat, r_pos, r_par = self._build_taper_half(
                self.lattice_mirrors_right, self.lattice_taper_center,
                self.parameters_mirrors_right, self.parameters_taper_center,
            )
            l_lat, l_pos, l_par = self._build_taper_half(
                self.lattice_mirrors_left, self.lattice_taper_center,
                self.parameters_mirrors_left, self.parameters_taper_center,
            )

            if has_center:
                r_pos += self.lattice_taper_center / 2
                l_pos = -(l_pos + self.lattice_taper_center / 2)

                taper_pos = np.concatenate([l_pos[::-1], [0.0], r_pos])
                taper_lat = np.concatenate([l_lat[::-1], [self.lattice_taper_center], r_lat])
                taper_par = np.concatenate([
                    l_par[::-1],
                    np.atleast_2d(self.parameters_taper_center),
                    r_par,
                ])
            else:
                half_gap = self.central_cavity / 2
                r_pos = r_pos + half_gap
                l_pos = -(l_pos + half_gap)
                taper_pos = np.concatenate([l_pos[::-1], r_pos])
                taper_lat = np.concatenate([l_lat[::-1], r_lat])
                taper_par = np.concatenate([l_par[::-1], r_par])
            
            # ── Stitch mirrors to defect taper edges ──────────────────────
            right_edge = taper_pos[-1] if taper_pos.size > 0 else 0
            left_edge = taper_pos[0] if taper_pos.size > 0 else 0
            right_edge_lattice = taper_lat[-1] if taper_lat.size > 0 else self.lattice_mirrors_right
            left_edge_lattice = taper_lat[0] if taper_lat.size > 0 else self.lattice_mirrors_left

            if self.n_mirrors_right > 0:
                first_gap_right = (right_edge_lattice + self.lattice_mirrors_right) / 2
                mirror_right_pos = first_gap_right + np.arange(self.n_mirrors_right) * self.lattice_mirrors_right
                mirror_right_pos = right_edge + mirror_right_pos
            else:
                mirror_right_pos = np.array([])

            if self.n_mirrors_left > 0:
                first_gap_left = (left_edge_lattice + self.lattice_mirrors_left) / 2
                mirror_left_pos = first_gap_left + np.arange(self.n_mirrors_left) * self.lattice_mirrors_left
                mirror_left_pos = left_edge - mirror_left_pos
            else:
                mirror_left_pos = np.array([])

            core_pos = np.concatenate([mirror_left_pos[::-1], taper_pos, mirror_right_pos])
            core_lat = np.concatenate([mirror_left_lat[::-1], taper_lat, mirror_right_lat])
            core_par = np.concatenate([mirror_left_par[::-1], taper_par, mirror_right_par])

            order = np.argsort(core_pos)
            core_pos, core_lat, core_par = core_pos[order], core_lat[order], core_par[order]

        # ── Linear taper: left side ───────────────────────────────────────
        # Interpolate from mirror params (at the mirror edge) to the linear
        # taper target params.  Lattice and hole_params are both ramped.
        # endpoint=True so the outermost hole reaches the target exactly.
        if self.n_linear_taper_left > 0:
            n_lt = self.n_linear_taper_left
            # Build exactly like the right taper, then flip.
            # t=0 → mirror value (closest to core), t=1 → target (outermost)
            t = np.linspace(0, 1, n_lt)
 
            lt_left_lat = self.lattice_mirrors_left + t * (self.lattice_linear_taper_left - self.lattice_mirrors_left)
            lt_left_par = np.array([
                self.parameters_mirrors_left + ti * (self.parameters_linear_taper_left - self.parameters_mirrors_left)
                for ti in t
            ])
 
            # Positions: stitch to the leftmost core hole using average spacing.
            # Build positions going leftward from core edge (same pattern as
            # right taper going rightward, but subtracting gaps).
            left_core_edge = core_pos[0]
            left_core_lattice = core_lat[0]
            first_gap = (left_core_lattice + lt_left_lat[0]) / 2
 
            lt_left_pos = np.zeros(n_lt)
            lt_left_pos[0] = left_core_edge - first_gap
            for i in range(1, n_lt):
                gap = (lt_left_lat[i] + lt_left_lat[i - 1]) / 2
                lt_left_pos[i] = lt_left_pos[i - 1] - gap
 
            # Reverse so that arrays are ordered left-to-right
            # (outermost first, closest-to-core last)
            lt_left_lat = lt_left_lat[::-1]
            lt_left_par = lt_left_par[::-1]
            lt_left_pos = lt_left_pos[::-1]
        else:
            lt_left_pos = np.array([])
            lt_left_lat = np.array([])
            lt_left_par = np.empty((0, core_par.shape[1]))
 
        # ── Linear taper: right side ──────────────────────────────────────
        if self.n_linear_taper_right > 0:
            n_rt = self.n_linear_taper_right
            # t=0 → mirror value (closest to mirror), t=1 → target (outermost)
            # So hole 0 is identical to the rightmost mirror, hole n_rt-1
            # has the linear taper target.
            t = np.linspace(0, 1, n_rt)
 
            lt_right_lat = self.lattice_mirrors_right + t * (self.lattice_linear_taper_right - self.lattice_mirrors_right)
            lt_right_par = np.array([
                self.parameters_mirrors_right + ti * (self.parameters_linear_taper_right - self.parameters_mirrors_right)
                for ti in t
            ])
 
            right_core_edge = core_pos[-1]
            right_core_lattice = core_lat[-1]
            first_gap = (right_core_lattice + lt_right_lat[0]) / 2
 
            lt_right_pos = np.zeros(n_rt)
            lt_right_pos[0] = right_core_edge + first_gap
            for i in range(1, n_rt):
                gap = (lt_right_lat[i] + lt_right_lat[i - 1]) / 2
                lt_right_pos[i] = lt_right_pos[i - 1] + gap
        else:
            lt_right_pos = np.array([])
            lt_right_lat = np.array([])
            lt_right_par = np.empty((0, core_par.shape[1]))
 
        # ── Assemble everything ───────────────────────────────────────────
        positions = np.concatenate([lt_left_pos, core_pos, lt_right_pos])
        lattices = np.concatenate([lt_left_lat, core_lat, lt_right_lat])
        params = np.concatenate([lt_left_par, core_par, lt_right_par])
 
        order = np.argsort(positions)
        return {
            "positions": positions[order],
            "lattice": lattices[order],
            "hole_params": params[order],
        }

    # ──────────────────────────────────────────────────────────────────────
    # Nanobeam structure
    # ──────────────────────────────────────────────────────────────────────

    def generate_nanobeam_structure(self):
        """Build waveguide slab + air holes as Tidy3D Structures."""
        geometry_wvg = td.PolySlab(
            vertices=[
                [-2*self.sim_size[0], self.width / 2],
                [ 2*self.sim_size[0], self.width / 2],
                [ 2*self.sim_size[0], -self.width / 2],
                [-2*self.sim_size[0], -self.width / 2],
            ],
            axis=2,
            slab_bounds=(-self.thickness / 2, self.thickness / 2),
            sidewall_angle=self.sidewall_angle,
        )
        waveguide = td.Structure(
            geometry=geometry_wvg,
            medium=self.nanobeam_medium,
            name="Nanobeam",
        )

        hole_geometries = [
            self.hole_geometry(self.geometry, (x, 0, 0), self.params[i])
            for i, x in enumerate(self.positions)
        ]
        holes = td.Structure(
            geometry=td.GeometryGroup(geometries=hole_geometries),
            medium=td.Medium(permittivity=1),
            name="Nanobeam_holes",
        )

        return [waveguide, holes]

    # ──────────────────────────────────────────────────────────────────────
    # Simulation
    # ──────────────────────────────────────────────────────────────────────

    def create_simulation(self, grid_size_override=(0.01, 0.01, 0.01)):
        """Assemble and return the full Tidy3D Simulation."""
        mesh_override = td.MeshOverrideStructure(
            geometry=td.Box(center=(0, 0, 0), size=(self.taper_length, self.width, self.thickness)),
            dl=grid_size_override,
        )
        grid_spec = td.GridSpec.auto(min_steps_per_wvl=15, override_structures=[mesh_override])

        boundary_spec = td.BoundarySpec(
            x=td.Boundary.pml(), y=td.Boundary.pml(), z=td.Boundary.pml(),
        )

        sym = self.get_symmetry()

        # x-symmetry requires the cavity to be mirror-symmetric about x=0.
        # Break it if any left/right mismatch in counts or parameters.
        is_symmetric = (
            self.n_mirrors_left == self.n_mirrors_right
            and self.n_linear_taper_left == self.n_linear_taper_right
            and np.isclose(self.lattice_mirrors_left, self.lattice_mirrors_right)
            and np.allclose(self.parameters_mirrors_left, self.parameters_mirrors_right)
            and np.isclose(self.lattice_linear_taper_left, self.lattice_linear_taper_right)
            and np.allclose(self.parameters_linear_taper_left, self.parameters_linear_taper_right)
        )
        if not is_symmetric:
            sym[0] = 0

        return td.Simulation(
            center=self.sim_center,
            size=self.sim_size,
            structures=self.nanobeam,
            sources=[self.sources],
            monitors=self.monitors,
            run_time=self.run_time,
            boundary_spec=boundary_spec,
            grid_spec=grid_spec,
            symmetry=tuple(sym),
            medium=td.Medium(permittivity=1),
        )

    def upload(self, directory, save_name):
        os.makedirs(directory, exist_ok=True)
        local_path = os.path.join(directory, f"{save_name}.json")
        self.sim.to_file(local_path)
        print(f"Simulation saved locally: {local_path}")
        self.task_id = web.upload(self.sim, task_name=save_name, folder_name=directory)
        print(f"Uploaded to server — task_id: {self.task_id}")
        return self.task_id

    def estimate_cost(self):
        if not hasattr(self, 'task_id') or self.task_id is None:
            raise RuntimeError("No task uploaded yet. Call upload() first.")
        cost = web.estimate_cost(self.task_id)
        print(f"Estimated cost: {cost:.3f} FlexCredits")
        return cost

    def run(self, directory, save_name, vgpu=False):
        os.makedirs(directory, exist_ok=True)
        if not hasattr(self, 'task_id') or self.task_id is None:
            self.upload(directory, save_name)

        if vgpu == True:
            self.sim_data = web.run(
                self.sim,
                task_name=f"{directory}/{save_name}",
                folder_name=directory,
                path=os.path.join(directory, f"{save_name}.hdf5"),
                priority=10
            )
        else:
            self.sim_data = web.run(
                self.sim,
                task_name=f"{directory}/{save_name}",
                folder_name=directory,
                path=os.path.join(directory, f"{save_name}.hdf5")
            )           

        self.save_params(directory, save_name)
        print(f"Results saved: {os.path.join(directory, f'{save_name}.hdf5')}")
        return self.sim_data

    # ──────────────────────────────────────────────────────────────────────
    # Data Analysis
    # ──────────────────────────────────────────────────────────────────────

    def find_source_decay(self):
        source = self.sim.sources[0]
        fwidth = source.source_time.fwidth
        time_offset = source.source_time.offset / (2 * np.pi * fwidth)
        pulse_time = 0.44 / fwidth
        return time_offset + pulse_time

    def analyse_resonances(self, start_time=None, freq_window=None):
        if self.sim_data is None:
            raise RuntimeError("No simulation data. Call run() first.")
        if start_time is None:
            start_time = 2 * self.find_source_decay()
        if freq_window is None:
            freq_window = (self.freq0 - self.fwidth / 2, self.freq0 + self.fwidth / 2)

        polarization = self.sim.sources[0].polarization
        combined_signal = 0
        i = 0
        name = f"Point_Monitor_{i}"
        while name in self.sim_data.monitor_data:
            combined_signal += self.sim_data[name].field_components[polarization].squeeze()
            i += 1
            name = f"Point_Monitor_{i}"

        rf = ResonanceFinder(freq_window=freq_window)
        bm = combined_signal.t >= start_time
        data = rf.run_raw_signal(combined_signal[bm], self.sim.dt)

        df = data.to_dataframe()
        df["wl"] = (c0 / df.index) * 1e3

        self.resonance_df = df
        self.combined_signal = combined_signal
        self.resonant_frequency = abs(df.sort_values("Q").index[-1])
        self.Q = df.sort_values("Q").iloc[-1]["Q"]
        return df, combined_signal

    def get_energy_density(self):
        if self.sim_data is None:
            raise RuntimeError("No simulation data. Call run() first.")
        Efield = self.sim_data.monitor_data["Field_Time_Monitor"]
        eps = abs(self.sim.epsilon(box=td.Box(center=(0, 0, 0), size=(td.inf, td.inf, td.inf))))
        eps = eps.interp(coords=dict(x=Efield.Ex.x, y=Efield.Ex.y, z=Efield.Ex.z))
        energy_density = np.abs(Efield.Ex**2 + Efield.Ey**2 + Efield.Ez**2) * eps
        delta_t = energy_density.t[-1] - energy_density.t[0]
        energy_density_mean = energy_density.integrate(coord="t") / delta_t
        self.energy_density = energy_density_mean
        self.eps = eps
        return energy_density_mean, eps

    def mode_volume(self):
        if not hasattr(self, 'energy_density') or self.energy_density is None:
            self.get_energy_density()
        if not hasattr(self, 'resonance_df') or self.resonance_df is None:
            self.analyse_resonances()
        wl_um = abs(self.resonance_df.sort_values("Q").iloc[-1]["wl"]) / 1e3
        n = np.sqrt(abs(self.eps.max()))
        integrated = np.abs(self.energy_density).integrate(coord=("x", "y", "z"))
        Vmode = integrated / np.max(self.energy_density) / (wl_um / n) ** 3
        symmetry_factor = 1
        for s in self.sim.symmetry:
            if s != 0:
                symmetry_factor *= 2
        Vmode *= symmetry_factor
        self.Vmode = Vmode
        print(f"Mode volume: {Vmode:.4f} (λ/n)³")
        return Vmode

    def directional_Q(self):
        if not hasattr(self, 'energy_density') or self.energy_density is None:
            self.get_energy_density()
        if not hasattr(self, 'resonant_frequency') or self.resonant_frequency is None:
            self.analyse_resonances()
        omega = 2 * np.pi * self.resonant_frequency
        Energy = td.EPSILON_0 * np.abs(self.energy_density).integrate(coord=("x", "y", "z"))
        dict_Q = {}
        for coord in ["x", "y", "z"]:
            for direction in ["+", "-"]:
                name = direction + coord
                if name not in self.sim_data.monitor_data:
                    continue
                flux_data = self.sim_data[name].flux
                delta_t = flux_data.t[-1] - flux_data.t[0]
                flux = abs(flux_data.integrate(coord="t")) / delta_t
                dict_Q[name] = (omega * Energy / flux).values
        total_Q = sum(1 / q for q in dict_Q.values()) ** -1
        dict_Q["total"] = total_Q
        self.Q_directional = dict_Q
        for key, val in dict_Q.items():
            print(f"Q_{key} = {val * 1e-6:.2f} M")
        return dict_Q

    def full_analysis(self):
        print("=== Resonance Analysis ===")
        df, _ = self.analyse_resonances()
        print(df.sort_values("Q", ascending=False).head(5))
        print(f"\nBest resonance: f = {self.resonant_frequency:.4e} Hz, Q = {self.Q:.0f}")
        print("\n=== Energy Density ===")
        self.get_energy_density()
        print("Energy density computed.")
        print("\n=== Mode Volume ===")
        self.mode_volume()
        print("\n=== Directional Q ===")
        self.directional_Q()
        print("\n=== Summary ===")
        print(f"  Q (resonance finder): {self.Q:.0f}")
        print(f"  Q (directional total): {self.Q_directional['total'] * 1e-6:.2f} M")
        print(f"  Mode volume: {self.Vmode:.4f} (λ/n)³")
        print(f"  Purcell factor estimate: {(3 / (4 * np.pi**2)) * (self.Q / self.Vmode):.1f}")

    # ──────────────────────────────────────────────────────────────────────
    # Gaussian Fit
    # ──────────────────────────────────────────────────────────────────────

    def plot_mode_along_beam(self):
        if self.sim_data is None:
            raise RuntimeError("No simulation data. Call run() first.")
        if self.resonant_frequency is None:
            self.analyse_resonances()

        pol = self.polarization
        monitor_axis_map = {"Ey": "y", "Ez": "z", "Hy": "y", "Hz": "z"}
        if pol not in monitor_axis_map:
            raise ValueError(f"Polarization '{pol}' not supported.")

        axis = monitor_axis_map[pol]
        mon_data = self.sim_data[f"Field_Profile_Monitor_{axis}"]
        field = getattr(mon_data, pol)
        field_1d = field.isel(t=-1)
        field_line = field_1d.sel(z=0, method="nearest") if axis == "y" else field_1d.sel(y=0, method="nearest")

        x = field_line.x.values
        vals = field_line.values.flatten().real
        vmax = np.max(np.abs(vals))
        if vmax > 0:
            vals = vals / vmax

        def gauss_sinusoid(x, A, sigma, k, phi, offset):
            return A * np.exp(-x**2 / (2 * sigma**2)) * np.sin(k * x + phi) + offset

        A0 = 1.0
        sigma0 = self.taper_length / 2 if hasattr(self, 'taper_length') else self.sim_size[0] / 8
        n_eff = np.sqrt(self.nanobeam_medium.permittivity.real) if hasattr(self, 'nanobeam_medium') else np.sqrt(self.sim.structures[0].medium.permittivity.real)
        k0 = 2 * np.pi * self.resonant_frequency / c0 * n_eff

        try:
            popt, pcov = curve_fit(
                gauss_sinusoid, x, vals,
                p0=[A0, sigma0, k0, 0.0, 0.0],
                bounds=([-np.inf, 0, 0, -np.pi, -np.inf], [np.inf, np.inf, np.inf, np.pi, np.inf]),
                maxfev=20000,
            )
            fit_success = True
        except RuntimeError:
            print("Warning: curve_fit did not converge.")
            popt = [A0, sigma0, k0, 0.0, 0.0]
            pcov = None
            fit_success = False

        A, sigma, k, phi, offset = popt
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(x, vals, color='steelblue', linewidth=0.8, label=f'{pol} along x (normalised)')
        x_fit = np.linspace(x.min(), x.max(), 2000)
        ax.plot(x_fit, gauss_sinusoid(x_fit, *popt), '--', color='red', linewidth=1.5, label='Gaussian × sin fit')
        envelope = A * np.exp(-x_fit**2 / (2 * sigma**2))
        ax.plot(x_fit, envelope + offset, ':', color='orange', linewidth=1.2, label='+envelope')
        ax.plot(x_fit, -envelope + offset, ':', color='orange', linewidth=1.2, label='−envelope')
        if hasattr(self, 'taper_start') and hasattr(self, 'taper_end'):
            ax.axvspan(self.taper_start, self.taper_end, alpha=0.1, color='red', label='Taper region')
        ax.axhline(0, color='gray', ls='-', alpha=0.3)
        ax.set_xlabel('x (µm)')
        ax.set_ylabel(f'{pol} (normalised)')
        ax.set_title(f'Mode profile — f_res = {self.resonant_frequency:.4e} Hz, Q = {self.Q:.0f}')
        ax.legend()
        plt.tight_layout()
        plt.show()

        print(f"Fit: A={A:.4f}, σ={abs(sigma):.4f} µm, k={k:.2f} rad/µm, φ={phi:.3f}")
        return popt, pcov if fit_success else None

    # ──────────────────────────────────────────────────────────────────────
    # Plots
    # ──────────────────────────────────────────────────────────────────────

    def plot_results(self):
        if not hasattr(self, 'sim_data') or self.sim_data is None:
            raise RuntimeError("No simulation data. Call run() first.")

        e_components = ["Ex", "Ey", "Ez"]
        h_components = ["Hx", "Hy", "Hz"]

        #for axis in ("x", "y", "z"):
        for axis in ("z"):
            monitor_name = f"Field_Profile_Monitor_{axis}"
            if monitor_name not in self.sim_data.monitor_data:
                continue
            mon_data = self.sim_data[monitor_name]
            t_last = mon_data.Ex.t[-1]

            fig, axes = plt.subplots(1, 3, figsize=(18, 4))
            fig.suptitle(f"{monitor_name} — |E| at t = {t_last:.2e} s", fontsize=12)
            for i, comp in enumerate(e_components):
                self.sim_data.plot_field(monitor_name, comp, val="abs", t=t_last, ax=axes[i])
                axes[i].set_title(f"|{comp}|")
            plt.tight_layout()
            plt.show()

            fig, axes = plt.subplots(1, 3, figsize=(18, 4))
            fig.suptitle(f"{monitor_name} — |H| at t = {t_last:.2e} s", fontsize=12)
            for i, comp in enumerate(h_components):
                self.sim_data.plot_field(monitor_name, comp, val="abs", t=t_last, ax=axes[i])
                axes[i].set_title(f"|{comp}|")
            plt.tight_layout()
            plt.show()

        '''
        point_monitors = [n for n in self.sim_data.monitor_data if n.startswith("Point_Monitor_")]
        if point_monitors:
            for components, label in [(e_components, "E"), (h_components, "H")]:
                fig, axes = plt.subplots(len(components), 1, figsize=(14, 3 * len(components)), sharex=True)
                fig.suptitle(f"Point monitors — {label} field vs time", fontsize=12)
                for i, comp in enumerate(components):
                    for mon_name in point_monitors:
                        field = getattr(self.sim_data[mon_name], comp)
                        axes[i].plot(field.t * 1e12, field.values.flatten(), linewidth=0.5, label=mon_name)
                    axes[i].set_ylabel(comp)
                    axes[i].legend(fontsize=6, loc='upper right')
                axes[-1].set_xlabel("Time (ps)")
                plt.tight_layout()
                plt.show()
        '''

    def plot_simulation(self):
        indices = np.arange(self.n_holes) - self.n_holes // 2
        n_params = self.params.shape[1]
        n_plots = 2 + n_params

        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots), sharex=True)

        axes[0].plot(indices, self.positions, '.-', color='steelblue', markersize=4, linewidth=0.8)
        axes[0].set_ylabel('x position (µm)')
        axes[0].set_title(f'Hole positions vs index ({len(self.positions)} holes)')
        axes[0].axhline(0, color='red', ls='--', alpha=0.3, label='cavity center')
        axes[0].legend(fontsize=8)

        axes[1].plot(indices, self.lattices, '.-', color='coral', markersize=4, linewidth=0.8)
        axes[1].axhline(self.lattice_mirrors_left, color='blue', ls='--', alpha=0.4, label=f'a_left = {self.lattice_mirrors_left}')
        axes[1].axhline(self.lattice_mirrors_right, color='green', ls='--', alpha=0.4, label=f'a_right = {self.lattice_mirrors_right}')
        axes[1].axhline(self.lattice_taper_center, color='red', ls=':', alpha=0.4, label=f'a_center = {self.lattice_taper_center}')
        axes[1].set_ylabel('Lattice constant (µm)')
        axes[1].set_title('Lattice constant vs hole index')
        axes[1].legend(fontsize=8)

        for j in range(n_params):
            ax = axes[2 + j]
            ax.plot(indices, self.params[:, j], '.-', color='black', markersize=4, linewidth=0.8)
            ax.axhline(self.parameters_mirrors_left[j], color='blue', ls='--', alpha=0.4, label=f'mirror_left = {self.parameters_mirrors_left[j]:.3f}')
            ax.axhline(self.parameters_mirrors_right[j], color='green', ls='--', alpha=0.4, label=f'mirror_right = {self.parameters_mirrors_right[j]:.3f}')
            ax.axhline(self.parameters_taper_center[j], color='red', ls=':', alpha=0.4, label=f'center = {self.parameters_taper_center[j]:.3f}')
            ax.set_ylabel(f'param_{j} (µm)')
            ax.set_title(f'param_{j} vs hole index')
            ax.legend(fontsize=7, loc='upper right')

        axes[-1].set_xlabel('Hole index')
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(14, 4))
        self.sim.plot(z=0, ax=ax)
        rect = Rectangle(
            (self.taper_start, -self.width / 2),
            width=(self.taper_end - self.taper_start),
            height=self.width,
            linewidth=2, edgecolor="red", facecolor="red", alpha=0.15,
            label="Mesh override region",
        )
        ax.add_patch(rect)
        padding = 0.5
        ax.set_xlim(self.taper_start - padding, self.taper_end + padding)
        ax.set_ylim(-self.width * 1.5, self.width * 1.5)
        ax.legend()
        ax.set_title("Override region highlighted — taper zoom")
        plt.show()

        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        self.sim.plot(z=0, ax=axes[0])
        axes[0].set_title('Top view (z = 0)', fontsize=12)
        self.sim.plot(y=0, ax=axes[1])
        axes[1].set_title('Side view (y = 0)', fontsize=12)
        self.sim.plot(x=0, ax=axes[2])
        axes[2].set_title('Cross-section (x = 0)', fontsize=12)
        plt.tight_layout()
        plt.show()




    def animate_field(
        self,
        monitor_name,
        field_component,
        output_path="field_animation.gif",
        fps=15,
        dpi=120,
        cmap="RdBu",
        symmetric_cmap=True,
        vmin=None,
        vmax=None,
        figsize=(12, 4),
        title=None,
        n_frames=None,
    ):
        """Create an animated GIF of a field component from a 2D FieldTimeMonitor.
 
        Parameters
        ----------
        monitor_name : str
            Name of the FieldTimeMonitor to animate. Must be a 2D monitor
            (one spatial dimension has size 0), e.g.
            ``"Field_Profile_Monitor_x"``, ``"Field_Profile_Monitor_y"``,
            or ``"Field_Profile_Monitor_z"``.
        field_component : str
            Which field component to plot: ``"Ex"``, ``"Ey"``, ``"Ez"``,
            ``"Hx"``, ``"Hy"``, or ``"Hz"``.
        output_path : str, optional
            File path for the output GIF (default: ``"field_animation.gif"``).
        fps : int, optional
            Frames per second in the GIF (default: 15).
        dpi : int, optional
            Resolution of each frame (default: 120).
        cmap : str, optional
            Matplotlib colormap name (default: ``"RdBu"``).
        symmetric_cmap : bool, optional
            If True (default), the colour scale is symmetric about zero,
            which works well for oscillating fields with a diverging
            colormap like ``"RdBu"``.
        vmin : float or None, optional
            Minimum value for the colour scale. If None (default), it is
            computed from the data (optionally made symmetric).
        vmax : float or None, optional
            Maximum value for the colour scale. If None (default), it is
            computed from the data.
        figsize : tuple, optional
            Figure size ``(width, height)`` in inches (default: ``(12, 4)``).
        title : str or None, optional
            Custom title template. May contain ``{t}`` for the current time
            in ps, ``{component}`` for the field component name, and
            ``{monitor}`` for the monitor name. If None, a default title is
            used.
        n_frames : int or None, optional
            Number of time steps to include. If None (default), all available
            time steps are used. If set, time steps are uniformly
            sub-sampled.
 
        Returns
        -------
        str
            The path to the saved GIF file.
 
        Raises
        ------
        RuntimeError
            If no simulation data is available.
        KeyError
            If the monitor or field component is not found.
        ValueError
            If the monitor is not a 2D slice (exactly one zero-size axis).
 
        Examples
        --------
        >>> cavity.animate_field("Field_Profile_Monitor_z", "Ey",
        ...                      output_path="Ey_top_view.gif", fps=20)
        """
        if self.sim_data is None:
            raise RuntimeError("No simulation data. Call run() first.")
        if monitor_name not in self.sim_data.monitor_data:
            available = list(self.sim_data.monitor_data.keys())
            raise KeyError(
                f"Monitor '{monitor_name}' not found. "
                f"Available monitors: {available}"
            )
 
        mon_data = self.sim_data[monitor_name]
        if not hasattr(mon_data, field_component):
            raise KeyError(
                f"Field component '{field_component}' not found on monitor "
                f"'{monitor_name}'. Available: Ex, Ey, Ez, Hx, Hy, Hz"
            )
 
        field = getattr(mon_data, field_component)
 
        # ── Determine the 2D slice axes ──────────────────────────────────
        # The monitor has one spatial axis with a single coordinate (size=0).
        # The other two axes form the 2D image.
        spatial_dims = [d for d in ("x", "y", "z") if d in field.dims]
        slice_axis = None
        plot_axes = []
        for d in spatial_dims:
            if field.sizes[d] <= 1:
                slice_axis = d
            else:
                plot_axes.append(d)
 
        if slice_axis is None or len(plot_axes) != 2:
            raise ValueError(
                f"Monitor '{monitor_name}' does not appear to be a 2D slice. "
                f"Spatial dims and sizes: "
                f"{[(d, field.sizes[d]) for d in spatial_dims]}"
            )
 
        # Squeeze the singleton spatial axis
        field_2d = field.squeeze(dim=slice_axis, drop=True)
 
        # ── Sub-sample time steps if requested ───────────────────────────
        t_vals = field_2d.t.values
        n_total = len(t_vals)
 
        if n_frames is not None and n_frames < n_total:
            indices = np.linspace(0, n_total - 1, n_frames, dtype=int)
        else:
            indices = np.arange(n_total)
            n_frames = n_total
 
        # ── Compute colour limits ────────────────────────────────────────
        # Sample a few frames to estimate global min/max efficiently
        sample_idx = np.linspace(0, n_total - 1, min(10, n_total), dtype=int)
        global_max = 0.0
        global_min = 0.0
        for si in sample_idx:
            frame_data = field_2d.isel(t=int(si)).values.real
            global_max = max(global_max, np.nanmax(frame_data))
            global_min = min(global_min, np.nanmin(frame_data))
 
        if vmin is None and vmax is None:
            if symmetric_cmap:
                abs_max = max(abs(global_min), abs(global_max))
                vmin, vmax = -abs_max, abs_max
            else:
                vmin, vmax = global_min, global_max
        elif vmin is None:
            vmin = -vmax if symmetric_cmap else global_min
        elif vmax is None:
            vmax = -vmin if symmetric_cmap else global_max
 
        # ── Coordinate arrays for the two plot axes ──────────────────────
        ax0_vals = field_2d.coords[plot_axes[0]].values
        ax1_vals = field_2d.coords[plot_axes[1]].values
 
        # ── Build the animation ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=figsize)
        first_frame = field_2d.isel(t=int(indices[0])).values.real.T
        im = ax.pcolormesh(
            ax0_vals, ax1_vals, first_frame,
            cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
        )
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(field_component)
        ax.set_xlabel(f"{plot_axes[0]} (µm)")
        ax.set_ylabel(f"{plot_axes[1]} (µm)")
        ax.set_aspect("equal")
 
        if title is None:
            title_template = "{component} | {monitor} | t = {t:.3f} ps"
        else:
            title_template = title
 
        def _update(frame_idx):
            ti = int(indices[frame_idx])
            data = field_2d.isel(t=ti).values.real.T
            im.set_array(data.ravel())
            t_ps = float(t_vals[ti]) * 1e12
            ax.set_title(
                title_template.format(
                    t=t_ps, component=field_component, monitor=monitor_name,
                )
            )
            return [im]
 
        anim = animation.FuncAnimation(
            fig, _update, frames=len(indices), interval=1000 / fps, blit=False,
        )
        anim.save(output_path, writer="pillow", fps=fps, dpi=dpi)
        plt.close(fig)
 
        print(
            f"GIF saved: {output_path} "
            f"({len(indices)} frames, {fps} fps, {dpi} dpi)"
        )
        return output_path