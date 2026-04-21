import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
from scipy.optimize import curve_fit

import tidy3d as td
from tidy3d import web
from tidy3d.plugins.dispersion import AdvancedFastFitterParam, FastDispersionFitter
from tidy3d.plugins.resonance import ResonanceFinder
from tidy3d.plugins.mode import ModeSolver

from scipy.constants import hbar, pi, epsilon_0, c, h

from hole import hole_geometry

C0 = td.constants.C_0           # speed of light in vacuum (µm/s)
C0_m = td.constants.C_0 * 1e-6  # speed of light in vacuum (m/s)

rng = np.random.default_rng(12345)

def _make_serializable(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj

class Cavity_simulation:

    # ──────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────

    def __init__(self, parameters, n_cells, context, beam_layout,
                 t_start=None, run_time=None):
        # Store design inputs
        self.parameters = parameters
        self.n_cells = n_cells
        self.context = context
        self.beam_layout = beam_layout

        # Timing
        self.run_time = run_time if run_time is not None else 5e-12
        self.t_start = t_start if t_start is not None else self.run_time - 1 / self.context["freq0"]

        # Derived wavelengths
        self.wl0 = C0 / self.context["freq0"]
        self.wlM = C0 / (self.context["freq0"] - self.context["fwidth"] / 2)

        # Populated by build()
        self.nanobeam_medium = None
        self.sim_center = None
        self.sim_size = None
        self.sources = None
        self.monitors = None
        self.nanobeam = None
        self.sim = None
        self.defect_start = None
        self.defect_end = None
        self.defect_length = None

        # Mode solver outputs
        self.n_eff = None
        self.k_wg = None
        self.a_last = None

        # Runtime / analysis outputs
        self.sim_data = None
        self.task_id = None
        self._analysis = {}

    def build(self, grid_size_override=(0.01, 0.01, 0.01),
              num_modes=2, plot=False):
        """Heavy construction: medium fitting, geometry, mode solve, simulation.

        Separated from __init__ so that from_file can skip it.
        """
        self.nanobeam_medium = self._resolve_medium(
            self.context["medium"], plot=plot
        )
        self.sim_center, self.sim_size = self._compute_sim_domain()
        self.sources = self._define_source()
        self.monitors = self._define_monitors(n_point_monitors=10)
        self._compute_defect_bounds()
        self.nanobeam = self._build_nanobeam()
        self.sim = self._create_simulation(grid_size_override)

        # Mode solve needs self.sim to exist
        self.n_eff, self.k_wg, self.a_last = self._mode_solve(
            num_modes=num_modes, plot=plot
        )
        return self

    @classmethod
    def from_saved_simulation_file(cls, filepath):
        """Reload from a previously saved HDF5 file without re-running
        medium fitting, mode solving, etc."""
        sim_data = td.SimulationData.from_file(filepath)
        attrs = sim_data.simulation.attrs

        instance = cls(
            n_cells=json.loads(attrs["n_cells"]),
            parameters=json.loads(attrs["parameters"]),
            context=json.loads(attrs["context"]),
            beam_layout=json.loads(attrs["beam_layout"]),
        )

        instance.sim = sim_data.simulation
        instance.sim_data = sim_data
        instance.sim_center = sim_data.simulation.center
        instance.sim_size = sim_data.simulation.size
        instance.run_time = sim_data.simulation.run_time
        instance._compute_defect_bounds()
        instance.nanobeam_medium = instance._resolve_medium(instance.context["medium"], plot=False)
        instance.full_analysis()  # populate all analysis properties

        return instance

    # ──────────────────────────────────────────────────────────────────
    # Analysis properties (lazy caching)
    # ──────────────────────────────────────────────────────────────────

    @property
    def resonance_df(self):
        if "resonance_df" not in self._analysis:
            self.analyse_resonances()
        return self._analysis["resonance_df"]

    @property
    def resonant_frequency(self):  # in Hz  
        if "resonant_frequency" not in self._analysis:
            self.analyse_resonances()
        return self._analysis["resonant_frequency"]

    @property
    def resonant_wavelength(self):  # in µm
        if "resonant_wavelength" not in self._analysis:
            self.analyse_resonances()
        return self._analysis["resonant_wavelength"]

    @property
    def resonant_omega_c(self):  # in rad/s
        if "resonant_omega_c" not in self._analysis:
            self.analyse_resonances()
        return self._analysis["resonant_omega_c"]

    @property
    def Q(self):
        if "Q" not in self._analysis:
            self.analyse_resonances()
        return self._analysis["Q"]

    @property
    def kappa_tot(self):
        if "kappa_tot" not in self._analysis:
            self.decay_rates()
        return self._analysis["kappa_tot"]

    @property
    def energy_density(self):
        if "energy_density" not in self._analysis:
            self.get_energy_density()
        return self._analysis["energy_density"]

    @property
    def eps(self):
        if "eps" not in self._analysis:
            self.get_energy_density()
        return self._analysis["eps"]

    @property
    def Vmode(self):    # in units of (λ[µm]/n)³
        if "Vmode" not in self._analysis:
            self.mode_volume()
        return self._analysis["Vmode"]
    
    @property
    def n_max(self):  
        if "n_max" not in self._analysis:
            self.mode_volume()
        return self._analysis["n_max"]

    @property
    def Q_directional(self):
        if "Q_directional" not in self._analysis:
            self.directional_Q()
        return self._analysis["Q_directional"]

    @property
    def kappa_dir(self):
        if "kappa_dir" not in self._analysis:
            self.decay_rates()
        return self._analysis["kappa_dir"]

    # ──────────────────────────────────────────────────────────────────
    # Guards
    # ──────────────────────────────────────────────────────────────────

    def _require_sim(self):
        if self.sim is None:
            raise RuntimeError("No simulation built. Call build() first.")

    def _require_data(self):
        if self.sim_data is None:
            raise RuntimeError("No simulation data. Call run() first.")

    # ──────────────────────────────────────────────────────────────────
    # Private: domain & geometry helpers
    # ──────────────────────────────────────────────────────────────────

    def _compute_sim_domain(self):
        """Compute simulation center and size from hole positions."""
        padding = 2 * self.wlM
        positions = self.beam_layout["positions"]
        lattices = self.beam_layout["lattice"]

        xmin = positions[0] - lattices[0] - padding
        xmax = positions[-1] + lattices[-1] + padding

        center = ((xmin + xmax) / 2, 0.0, 0.0)
        size = (
            xmax - xmin,
            self.context["width"] + 2 * padding,
            self.context["thickness"] + 2 * padding,
        )
        return center, size

    def _compute_defect_bounds(self):
        n_lt = self.n_cells["N_left_taper"]
        n_lm = self.n_cells["N_left_mirror"]
        n_rt = self.n_cells["N_right_taper"]
        n_rm = self.n_cells["N_right_mirror"]
        pos = self.beam_layout["positions"]
        lat = self.beam_layout["lattice"]

        idx_start = n_lt + n_lm
        idx_end = -(n_rt + n_rm) - 1

        self.defect_start = pos[idx_start] - lat[idx_start] / 2
        self.defect_end = pos[idx_end] + lat[idx_end] / 2
        self.defect_length = self.defect_end - self.defect_start

    def _get_symmetry(self):
        SYMMETRY_MAP = {
            "Ex": [-1,  1,  1],
            "Ey": [ 1, -1,  1],
            "Ez": [ 1,  1, -1],
            "Hx": [ 1, -1, -1],
            "Hy": [-1,  1, -1],
            "Hz": [-1, -1,  1],
        }
        pol = self.context["polarization"]
        if pol not in SYMMETRY_MAP:
            raise ValueError(
                f"Unknown polarization '{pol}'. "
                f"Must be one of {list(SYMMETRY_MAP)}"
            )
        return list(SYMMETRY_MAP[pol])

    # ──────────────────────────────────────────────────────────────────
    # Private: medium
    # ──────────────────────────────────────────────────────────────────

    def _resolve_medium(self, medium_spec, plot=False):
        if isinstance(medium_spec, (int, float)):
            return td.Medium(permittivity=float(medium_spec) ** 2)
        elif isinstance(medium_spec, str):
            return self._medium_from_file(medium_spec, plot=plot)
        else:
            raise TypeError(
                f"context['medium'] must be a number or file path, "
                f"got {type(medium_spec).__name__}"
            )

    @staticmethod
    def _medium_from_file(filename, plot=False):
        advanced_param = AdvancedFastFitterParam(weights=(1, 1))
        
        fitter = FastDispersionFitter.from_file(
            filename, skiprows=1, delimiter="\t", usecols=(0, 1)
        )
        medium, rms = fitter.fit(max_num_poles=3, advanced_param=advanced_param, tolerance_rms=2e-2)
        if plot:
            fitter.plot(medium)
            plt.show()
        return medium

    def _n_at_frequency_from_medium(self, frequency_hz):
        # complex relative permittivity at f
        eps = self.nanobeam_medium.eps_model(frequency_hz)     # complex scalar

        # complex refractive index
        n = np.sqrt(eps)
        return n

    # ──────────────────────────────────────────────────────────────────
    # Private: source & monitors
    # ──────────────────────────────────────────────────────────────────

    def _define_source(self):
        n_defect = self.n_cells["N_defect"]
        mode = self.context["mode"]

        # Determine source offset from origin
        defect_lat = self.parameters["parameters_defect"]["lattice"]
        if n_defect % 2 != 0 and mode == "dielectric":
            source_center = (defect_lat / 2, 0, 0)
        elif n_defect % 2 == 0 and mode == "dielectric":
            source_center = (0, 0, 0)
        elif n_defect % 2 != 0 and mode == "air":
            source_center = (0, 0, 0)
        elif n_defect % 2 == 0 and mode == "air":
            source_center = (defect_lat / 2, 0, 0)
        else:
            source_center = (0, 0, 0)

        return td.PointDipole(
            center=source_center,
            name="Point_Dipole_Source",
            polarization=self.context["polarization"],
            source_time=td.GaussianPulse(
                freq0=self.context["freq0"],
                fwidth=self.context["fwidth"],
                phase=2 * np.pi * np.random.random(),
            ),
        )

    def _define_monitors(self, n_point_monitors=10):
        monitors = []

        # Random point monitors inside the defect region
        deviation = np.array([
            self.context["width"] / 4,
            self.context["width"] / 4,
            self.context["thickness"] / 4,
        ])
        positions = np.random.random((n_point_monitors, 3)) * deviation
        for i in range(n_point_monitors):
            monitors.append(
                td.FieldTimeMonitor(
                    center=tuple(positions[i]),
                    name=f"Point_Monitor_{i}",
                    start=0,
                    size=(0, 0, 0),
                    interval=1,
                )
            )

        # 3D field monitor
        mon_size_3d = tuple(np.array(self.sim_size) - 2 * self.wlM)
        monitors.append(
            td.FieldTimeMonitor(
                center=self.sim_center,
                size=mon_size_3d,
                start=self.t_start,
                name="Field_Time_Monitor",
                interval_space=(5, 5, 5),
                interval=10,
            )
        )

        # 2D field profile monitors
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
                    start=self.t_start,
                    name=f"Field_Profile_Monitor_{axis}",
                )
            )

        # Six planar flux monitors
        for axis_idx, axis_label in enumerate("xyz"):
            for sign in (-1, +1):
                mon_center = list(self.sim_center)
                mon_center[axis_idx] += sign * (self.sim_size[axis_idx] / 2 - self.wlM)
                mon_size = list(self.sim_size)
                mon_size[axis_idx] = 0.0
                prefix = "-" if sign == -1 else "+"
                monitors.append(
                    td.FluxTimeMonitor(
                        center=tuple(mon_center),
                        size=tuple(mon_size),
                        start=self.t_start,
                        name=f"{prefix}{axis_label}",
                    )
                )

        return monitors

    # ──────────────────────────────────────────────────────────────────
    # Private: mode solver
    # ──────────────────────────────────────────────────────────────────

    def _mode_solve(self, num_modes=1, plot=False):
        self._require_sim()

        mode_size = [0, self.sim_size[1] - self.wlM, self.sim_size[2] - self.wlM]

        if self.n_cells["N_defect"] % 2 != 0:
            mode_center = (self.parameters["parameters_defect"]["lattice"] / 2, 0, 0)
        else:
            mode_center = (0, 0, 0)

        n_eff_target = np.real(self._n_at_frequency_from_medium(self.context["freq0"]))


        mode_plane = td.Box(center=mode_center, size=mode_size)
        mode_spec = td.ModeSpec(
            num_modes=num_modes,
            target_neff=n_eff_target,
        )
        solver = ModeSolver(
            simulation=self.sim,
            plane=mode_plane,
            mode_spec=mode_spec,
            freqs=[self.context["freq0"]],
        )
        mode_data = solver.solve()

        if plot:
            solver.plot()
            plt.show()
            print(mode_data.to_dataframe())

            fig, axs = plt.subplots(2*num_modes, 3, figsize=(15, 8), tight_layout=True)
            for mode_ind in range(num_modes):
                rE = 2*mode_ind
                rH = rE + 1

                # Electric (same row)
                solver.plot_field("Ex", "real", f=self.context["freq0"], mode_index=mode_ind, ax=axs[rE, 0])
                solver.plot_field("Ey", "real", f=self.context["freq0"], mode_index=mode_ind, ax=axs[rE, 1])
                solver.plot_field("Ez", "real", f=self.context["freq0"], mode_index=mode_ind, ax=axs[rE, 2])

                # Magnetic (row below)
                solver.plot_field("Hx", "real", f=self.context["freq0"], mode_index=mode_ind, ax=axs[rH, 0])
                solver.plot_field("Hy", "real", f=self.context["freq0"], mode_index=mode_ind, ax=axs[rH, 1])
                solver.plot_field("Hz", "real", f=self.context["freq0"], mode_index=mode_ind, ax=axs[rH, 2])

            plt.show()

        n_eff = mode_data.n_eff.values.flatten()
        k_wg = 2 * np.pi * n_eff * self.context["freq0"] / C0
        a_last = self.wl0 / (2 * n_eff)

        return n_eff, k_wg, a_last

    # ──────────────────────────────────────────────────────────────────
    # Private: beam structure
    # ──────────────────────────────────────────────────────────────────

    def _build_nanobeam(self):
        """Build waveguide slab + air holes as Tidy3D Structures."""
        geometry_wvg = td.PolySlab(
            vertices=[
                [-2 * self.sim_size[0],  self.context["width"] / 2],
                [ 2 * self.sim_size[0],  self.context["width"] / 2],
                [ 2 * self.sim_size[0], -self.context["width"] / 2],
                [-2 * self.sim_size[0], -self.context["width"] / 2],
            ],
            axis=2,
            slab_bounds=(
                -self.context["thickness"] / 2,
                 self.context["thickness"] / 2,
            ),
            sidewall_angle=self.context["sidewall_angle"],
        )
        waveguide = td.Structure(
            geometry=geometry_wvg,
            medium=self.nanobeam_medium,
            name="Nanobeam",
        )

        hole_geometries = [
            hole_geometry(
                self.context["geometry"],
                (x, 0, 0),
                self.beam_layout["hole_params"][i],
                self.context["thickness"],          # pass thickness explicitly
            )
            for i, x in enumerate(self.beam_layout["positions"])
        ]
        holes = td.Structure(
            geometry=td.GeometryGroup(geometries=hole_geometries),
            medium=td.Medium(permittivity=1),
            name="Nanobeam_holes",
        )

        return [waveguide, holes]

    # ──────────────────────────────────────────────────────────────────
    #  Bandstructure
    # ──────────────────────────────────────────────────────────────────

    def bandstructure_tidy(self, directory, save_name, context=None, parameters=None, plot=False):
        if context is None:
            context = self.context
        if parameters is None:
            parameters = self.parameters["parameters_defect"]

        # Frequency range of interest (Hz)
        freq_range_unitless = np.array((0.1, 0.5))  # in units of c/a
        freq_scale = C0 / parameters["lattice"]  # frequency scale determined by the lattice constant
        freq_range = freq_range_unitless * freq_scale

        # Gaussian pulse parameters
        freq0 = np.sum(freq_range) / 2  # central frequency
        freqw = 0.3 * (freq_range[1] - freq_range[0])  # pulse width

        wlM = C0 / (freq0 - freqw / 2)


        slab = td.Structure(
            geometry=td.Box(
                center=(0, 0, 0),
                size=(td.inf, td.inf, context["thickness"]),
            ),
            medium=self.nanobeam_medium,
            name="slab",
        )

        hole = td.Structure(
            geometry=hole_geometry(
                geometry=context["geometry"],
                hole_center=(0, 0, 0),
                params=parameters["hole_params"],
                thickness=context["thickness"],          # pass thickness explicitly
            ),
            medium=td.Medium(permittivity=1),
            name="hole"
        )

        structures = [slab, hole]

        num_dipoles = 7
        num_monitors = 2
        polarization = context["polarization"] # "Hz" for TE-like, "Ez" for TM-like

        dipole_positions = rng.uniform([-parameters["lattice"] / 2, -context["width"] / 2, 0], [parameters["lattice"] / 2, context["width"] / 2, 0], [num_dipoles, 3])
        dipole_phases = rng.uniform(0, 2 * np.pi, num_dipoles)

        pulses = []
        dipoles = []
        for i in range(num_dipoles):
            pulse = td.GaussianPulse(freq0=freq0, fwidth=freqw, phase=dipole_phases[i])
            pulses.append(pulse)
            dipoles.append(
                td.PointDipole(
                    source_time=pulse,
                    center=tuple(dipole_positions[i]),
                    polarization=polarization,
                    name="dipole_" + str(i),
                )
            )

        monitor_positions = rng.uniform([-parameters["lattice"] / 2, -context["width"] / 2, 0], [parameters["lattice"] / 2, context["width"] / 2, 0], [num_monitors, 3])

        t_start = 5/freqw 
        monitors_time = []
        for i in range(num_monitors):
            monitors_time.append(
                td.FieldTimeMonitor(
                    fields=["Ex","Ey","Ez","Hx","Hy","Hz"],
                    center=tuple(monitor_positions[i]),
                    size=(0, 0, 0),
                    start=t_start,
                    name="monitor_time_" + str(i),
                )
            )

        ks = []
        Nk = 8
        for i in range(8):
            k = (1 / 2) * i / Nk
            ks.append(k)

        ks.append(1/2)

        bspecs_gammax = []

        for i in range(Nk+1):
            k = ks[i]
            bspecs_gammax.append(
                td.BoundarySpec(
                    x=td.Boundary.bloch(k),
                    y=td.Boundary.periodic(),
                    z=td.Boundary.pml(),
                )
            )

        bspecs = bspecs_gammax 

        run_time = 200 / freqw
        sims = {}

        spacing = wlM
        sim_size = (parameters["lattice"], context["width"], 2 * spacing + context["thickness"])

        for i in range(Nk+1):
            sims[f"sim_{i}"] = td.Simulation(
                center=(0, 0, 0),
                size=sim_size,
                grid_spec=td.GridSpec.auto(),
                structures=structures,
                sources=dipoles,
                monitors=monitors_time,
                run_time=run_time,
                shutoff=0,
                boundary_spec=bspecs[i],
                normalize_index=None,
                symmetry=(0,0,1),
            )

        if plot:
            fig, ax = plt.subplots(1, 3, tight_layout=True, figsize=(10, 4))
            sims["sim_0"].plot(z=0, ax=ax[0], hlim=[-sim_size[0]/2, sim_size[0]/2], vlim=[-sim_size[0]/2, sim_size[0]/2])
            sims["sim_0"].plot(y=0, ax=ax[1], hlim=[-sim_size[0]/2, sim_size[0]/2], vlim=[-context["thickness"], context["thickness"]])
            sims["sim_0"].plot(x=0, ax=ax[2], hlim=[-sim_size[1]/2, sim_size[1]/2], vlim=[-context["thickness"], context["thickness"]])
            plt.show()

            f, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(8, 4))
            plot_time = 5 / context["fwidth"]
            ax1 = (
                sims["sim_0"]
                .sources[0]
                .source_time.plot(times=np.linspace(0, plot_time, 1001), val="abs", ax=ax1)
            )
            ax1.set_xlim(0, plot_time)
            ax2 = (
                sims["sim_0"]
                .sources[0]
                .source_time.plot_spectrum(
                    times=np.linspace(0, sims["sim_0"].run_time, 10001), val="abs", ax=ax2
                )
            )
            ax2.hlines(1.5e-15, freq_range[0], freq_range[1], linewidth=10, color="g", alpha=0.4)
            ax2.legend(("source spectrum", "measurement"), loc="upper right")
            plt.show()
    
        batch = td.web.Batch(simulations=sims, folder_name=directory, verbose=True)
        batch_data = batch.run(path_dir=directory)
        batch_data.to_file(directory + "/" + save_name + "_batch_metadata.json")

        if plot:
            plt.plot(
                batch_data[f"sim_{Nk}"].monitor_data["monitor_time_0"].Hz.t,
                np.real(batch_data[f"sim_{Nk}"].monitor_data["monitor_time_0"].Hz.squeeze()),
            )
            plt.title("FieldTimeMonitor data")
            plt.xlabel("t(s)")
            plt.ylabel("Hz")
            plt.show()

            plt.plot(
                batch_data[f"sim_{Nk}"].monitor_data["monitor_time_0"].Ez.t,
                np.real(batch_data[f"sim_{Nk}"].monitor_data["monitor_time_0"].Hz.squeeze()),
            )
            plt.title("FieldTimeMonitor data")
            plt.xlabel("t")
            plt.ylabel("Ez")
            plt.show()

        resonance_finder = ResonanceFinder(freq_window=tuple(freq_range))
        resonance_datas = []
        for i in range(Nk+1):
            sim_data = batch_data[f"sim_{i}"]
            resonance_data = resonance_finder.run(signals=sim_data.data)
            resonance_datas.append(resonance_data)

        for i in range(Nk+1):
            resonance_data = resonance_datas[i]
            resonance_data = self.filter_resonances(
                resonance_data=resonance_data, minQ=0, minamp=0.001, maxerr=100
            )
            freqs = resonance_data.freq.to_numpy()
            Qs = resonance_data.Q.to_numpy()
            plt.scatter(np.full(len(freqs), (1 / 2) * i / Nk), freqs / 1e12, color="blue")

        kpar = np.linspace(0, 0.5, 100) # in plane magnitude
        c_m = C0*1e-6
        lattice_m = parameters["lattice"] * 1e-6    # lattice constant in meters (350 nm)
        light_cone = kpar * c_m / lattice_m * 1e-12

        plt.axhline(freq0 / 1e12, color="black")
        plt.plot(kpar, light_cone, color="black", alpha=0.2)
        plt.ylim(0, 0.5*c_m / lattice_m * 1e-12)
        plt.title("Band diagram")
        plt.ylabel("Frequency (THz)")
        plt.xlabel("Wavevector")
        plt.show()

    def filter_resonances(self, resonance_data, minQ, minamp, maxerr):
        resonance_data = resonance_data.where(abs(resonance_data.Q) > minQ, drop=True)
        resonance_data = resonance_data.where(resonance_data.amplitude > minamp, drop=True)
        resonance_data = resonance_data.where(resonance_data.error < maxerr, drop=True)
        return resonance_data

    def freq_nm(freq):
        return (td.C_0 / freq) * 1e3

    # ──────────────────────────────────────────────────────────────────
    # Private: simulation assembly
    # ──────────────────────────────────────────────────────────────────

    def _create_simulation(self, grid_size_override=(0.01, 0.01, 0.01)):
        mesh_override = td.MeshOverrideStructure(
            geometry=td.Box(
                center=(0, 0, 0),
                size=(
                    self.defect_length,
                    self.context["width"],
                    self.context["thickness"],
                ),
            ),
            dl=grid_size_override,
        )
        grid_spec = td.GridSpec.auto(
            min_steps_per_wvl=15,
            override_structures=[mesh_override],
        )
        boundary_spec = td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.pml(),
            z=td.Boundary.pml(),
        )

        sym = self._get_symmetry()

        if not self._is_symmetric():
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
            attrs={
                "n_cells": json.dumps(_make_serializable(self.n_cells)),
                "parameters": json.dumps(_make_serializable(self.parameters)),
                "context": json.dumps(_make_serializable(self.context)),
                "beam_layout": json.dumps(_make_serializable(self.beam_layout)),
            },
        )

    def _is_symmetric(self):
        """Check if left/right cavity halves are mirror-symmetric."""
        nc = self.n_cells
        p = self.parameters

        counts_match = (
            nc["N_left_mirror"] == nc["N_right_mirror"]
            and nc["N_left_taper"] == nc["N_right_taper"]
        )
        if not counts_match:
            return False

        params_match = (
            np.isclose(
                p["parameters_mirrors_left"]["lattice"],
                p["parameters_mirrors_right"]["lattice"],
            )
            and np.allclose(
                p["parameters_mirrors_left"]["hole_params"],
                p["parameters_mirrors_right"]["hole_params"],
            )
            and np.isclose(
                p["parameters_taper_left"]["lattice"],
                p["parameters_taper_right"]["lattice"],
            )
            and np.allclose(
                p["parameters_taper_left"]["hole_params"],
                p["parameters_taper_right"]["hole_params"],
            )
        )
        return params_match

    # ──────────────────────────────────────────────────────────────────
    # Management
    # ──────────────────────────────────────────────────────────────────

    def upload(self, directory, save_name):
        self._require_sim()
        os.makedirs(directory, exist_ok=True)
        local_path = os.path.join(directory, f"{save_name}.json")
        self.sim.to_file(local_path)
        self.task_id = web.upload(self.sim, task_name=save_name, folder_name=directory,)
        return self.task_id

    def estimate_cost(self):
        if self.task_id is None:
            raise RuntimeError("No task uploaded yet. Call upload() first.")
        cost = web.estimate_cost(self.task_id)
        print(f"Estimated cost: {cost:.3f} FlexCredits")
        return cost

    def run(self, directory, save_name):
        self._require_sim()
        os.makedirs(directory, exist_ok=True)

        if self.task_id is None:
            self.upload(directory, save_name)

        self.sim_data = web.run(self.sim, task_name=save_name, folder_name=directory, path=os.path.join(directory, f"{save_name}.hdf5"), verbose=True)
        return self.sim_data

    # ──────────────────────────────────────────────────────────────────
    # Data Analysis
    # ──────────────────────────────────────────────────────────────────

    def _find_source_decay(self):
        # source bandwidth
        src = self.sim.sources[0].source_time

        # time offset to start the source
        time_offset = src.offset / (2 * np.pi * src.fwidth)
        
        # time width for a gaussian pulse
        return time_offset + 0.44 / src.fwidth

    def _count_point_monitors(self):
        i = 0
        while f"Point_Monitor_{i}" in self.sim_data.monitor_data:
            i += 1
        return i

    def analyse_resonances(self, start_time=None, freq_window=None):
        self._require_data()

        if start_time is None:
            start_time = 2 * self._find_source_decay()
        if freq_window is None:
            f0 = self.context["freq0"]
            fw = self.context["fwidth"]
            freq_window = (f0 - fw / 2, f0 + fw / 2)

        pol = self.sim.sources[0].polarization
        combined = sum(self.sim_data[f"Point_Monitor_{i}"].field_components[pol].squeeze() for i in range(self._count_point_monitors()))

        rf = ResonanceFinder(freq_window=freq_window)
        mask = combined.t >= start_time
        data = rf.run_raw_signal(combined[mask], self.sim.dt)

        df = data.to_dataframe()            # the dataframe has columns for: decay, Q, amplitude, phase, error, and index is frequency. So df.index is frequency in Hz 

        df["wl"] = (C0 / df.index)          # I am adding wavelength in um. Because df.index is frequency in Hz and C0 is speed in um/s. 

        best = df.sort_values("Q").iloc[-1]  # when i sort them best.name becomes the frequency in Hz of the best resonance. And best["Q"] is the Q of the best resonance... 
        
        self._analysis["resonance_df"] = df
        self._analysis["combined_signal"] = combined
        self._analysis["resonant_frequency"] = abs(best.name)    # frequency in Hz
        self._analysis["resonant_wavelength"] = abs(best["wl"])  # wavelength in um
        self._analysis["resonant_omega_c"] = 2 * np.pi * abs(best.name)   # angular frequency in rad/s
        self._analysis["Q"] = best["Q"]

        kappa_tot = 2 * np.pi * abs(best.name) / best["Q"]  # total decay rate in rad/s
        self._analysis["kappa_tot"] = kappa_tot
        return df, combined

    def get_energy_density(self):
        self._require_data()

        Efield = self.sim_data.monitor_data["Field_Time_Monitor"]
        
        eps = abs(self.sim.epsilon(box=td.Box(center=(0, 0, 0), size=(td.inf, td.inf, td.inf)), freq=self.resonant_frequency,))     # returns the permittivity within the specified volume.
        eps = eps.interp(coords=dict(x=Efield.Ex.x, y=Efield.Ex.y, z=Efield.Ex.z))

        energy_density = (np.abs(Efield.Ex**2 + Efield.Ey**2 + Efield.Ez**2) * eps)  # it is |E(x, y, z, t)|^2 * eps. This is the energy density at each point in space and time, so it is a 4D array with coordinates (t, x, y, z). it is |E|^2 * eps.

        delta_t = energy_density.t[-1] - energy_density.t[0]
        energy_density_mean = energy_density.integrate(coord="t") / delta_t

        self._analysis["energy_density"] = energy_density_mean
        self._analysis["eps"] = eps
        return energy_density_mean, eps

    def mode_volume(self):
        self._require_data()

        integrated = np.abs(self.energy_density).integrate(coord=("x", "y", "z"))
        
        n_max = np.sqrt(abs(self.eps.max()))

        Vmode = integrated / np.max(self.energy_density) / (self.resonant_wavelength / n_max) ** 3

        symmetry_factor = 1
        for s in self.sim.symmetry:
            if s != 0:
                symmetry_factor *= 2
        Vmode *= symmetry_factor

        self._analysis["Vmode"] = Vmode
        self._analysis["n_max"] = n_max
        print(f"Mode volume: {Vmode:.4f} (λ[um]/n)³")
        return Vmode

    def directional_Q(self):
        self._require_data()

        Energy = td.EPSILON_0 * np.abs(self.energy_density).integrate(coord=("x", "y", "z"))

        dict_Q = {}
        for coord in "xyz":
            for direction in ("+", "-"):
                name = f"{direction}{coord}"
                if name not in self.sim_data.monitor_data:
                    continue
                flux_data = self.sim_data[name].flux
                delta_t = flux_data.t[-1] - flux_data.t[0]
                flux = abs(flux_data.integrate(coord="t")) / delta_t
                dict_Q[name] = (self.resonant_omega_c * Energy / flux).values

        total_Q = sum(1 / q for q in dict_Q.values()) ** -1
        dict_Q["total"] = total_Q

        self._analysis["Q_directional"] = dict_Q
        for key, val in dict_Q.items():
            print(f"Q_{key} = {val * 1e-6:.2f} M")
        return dict_Q

    def decay_rates(self):
        """Compute cavity decay rates κ = ω_c / Q for each channel."""
        self._require_data()
        
        Q_dir = self.Q_directional
        
        kappa = {}
        for channel, Q_val in Q_dir.items():
            kappa[channel] = self.resonant_omega_c / Q_val
            ghz = kappa[channel] / (2 * pi * 1e9)
            print(f"  κ_{channel} / 2π = {ghz:.3f} GHz")
        
        self._analysis["kappa_dir"] = kappa
        return kappa
        
    def full_analysis(self):
        self._require_data()

        print("=== Resonance Analysis ===")
        df, _ = self.analyse_resonances()
        print(df.sort_values("Q", ascending=False).head(5))
        print(f"\nBest resonance: f = {self.resonant_frequency:.4e} Hz, "
              f"Q = {self.Q:.0f}")

        print("\n=== Energy Density ===")
        self.get_energy_density()
        print("Energy density computed.")

        print("\n=== Mode Volume [um³] ===")
        self.mode_volume()

        print("\n=== Directional Q ===")
        self.directional_Q()

        print("\n=== Directional kappa ===")
        self.decay_rates()

        print("\n=== Summary ===")
        Q_dir = self.Q_directional
        print(f"  Q (resonance finder): {self.Q:.0f}")
        print(f"  Kappa (total): {self.kappa_tot / (2 * pi * 1e9):.3f} GHz")
        print(f"  Q (directional total): {Q_dir['total'] * 1e-6:.2f} M")
        print(f"  Mode volume: {self.Vmode:.4f} in units of (λ[um]/n)³")
        purcell = (3 / (4 * np.pi**2)) * (self.Q / self.Vmode)
        print(f"  Purcell factor estimate: {purcell:.1f}")

    # ──────────────────────────────────────────────────────────────────
    # Polarization Profile and g calculation    
    # ──────────────────────────────────────────────────────────────────

    def polarization_profile(self, monitor_name="Field_Profile_Monitor_x", polarization_component="Ey", plot=False):
        """Compute the polarization fraction for a given field component.

        Parameters
        ----------
        monitor_name : str
            Name of a FieldTimeMonitor in sim_data.
        polarization_component : str
            Which component to measure dominance of ('Ex', 'Ey', or 'Ez').
        plot : bool
            If True, plot the polarization map at the last time step.

        Returns
        -------
        polarization_map : xarray.DataArray
            |E_pol| / |E_total| at the final time step.
        """
        self._require_data()

        E_t = self.sim_data[monitor_name]
        Ex = E_t.Ex.isel(t=-1)
        Ey = E_t.Ey.isel(t=-1)
        Ez = E_t.Ez.isel(t=-1)

        E_magnitude = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)

        component_map = {"Ex": Ex, "Ey": Ey, "Ez": Ez}
        if polarization_component not in component_map:
            raise ValueError(
                f"polarization_component must be one of {list(component_map)}, "
                f"got '{polarization_component}'"
            )
        E_pol = np.abs(component_map[polarization_component])

        epsilon = 1e-20 # to prevent division by zero when field is very weak and decayed
        polarization_map = E_pol / (E_magnitude + epsilon)

        if plot:
            fig, ax = plt.subplots(figsize=(6, 6))

            if monitor_name == "Field_Profile_Monitor_x":
                im = polarization_map.plot(
                    ax=ax, cmap="viridis", add_colorbar=False, x="y", y="z",
                    vmin=0.9, vmax=1.0,
                )
            if monitor_name == "Field_Profile_Monitor_y":
                im = polarization_map.plot(
                    ax=ax, cmap="viridis", add_colorbar=False, x="x", y="z",
                    vmin=0.9, vmax=1.0,
                )
            if monitor_name == "Field_Profile_Monitor_z":
                im = polarization_map.plot(
                    ax=ax, cmap="viridis", add_colorbar=False, x="x", y="y",
                    vmin=0.9, vmax=1.0,
                )

            outline = Rectangle(
                (-self.context["width"] / 2, -self.context["thickness"] / 2),
                self.context["width"], self.context["thickness"],
                linewidth=1.5, edgecolor="white", facecolor="none",
                alpha=0.8, linestyle="--",
            )
            ax.add_patch(outline)
            fig.colorbar(im, ax=ax,
                        label=f"Polarization Ratio (|{polarization_component}| / |E|)")
            ax.set_title(
                f"Polarization Ratio — {monitor_name}\n"
                f"|{polarization_component}| / |E| at final time step"
            )
            ax.set_aspect("equal")
            plt.show()

        return polarization_map

    def coupling_g(self, dipole_moment, monitor_name="Field_Profile_Monitor_x", polarization_component="Ey", plot=False, contour_ghz=None):
        """Compute the vacuum Rabi coupling g(r) over the monitor plane.

        Parameters
        ----------
        dipole_moment : float
            Transition dipole moment of the emitter (C·m).
            For example, D2 line of Rb 5^2S_1/2 to 5^2P_3/2 has ~3.58e-29 C·m.
        monitor_name : str
            A FieldTimeMonitor whose last time step gives the cavity mode profile.
        polarization_component : str
            Field component aligned with the dipole ('Ex', 'Ey', or 'Ez').

        Returns
        -------
        g_map : xarray.DataArray
            g(r) in Hz (angular: rad/s → divide by 2π for linear frequency).
        g_max : float
            Maximum coupling strength (Hz).
        """
        self._require_data()

        if dipole_moment is None:
            raise ValueError("dipole_moment must be provided to calculate g.")

        Vm_si = self.Vmode  * (self.resonant_wavelength * 1e-6 / self.n_max) ** 3                 # m³

        # --- peak coupling rate ---
        g_max = dipole_moment * np.sqrt(self.resonant_omega_c / (2 * hbar * epsilon_0 * Vm_si))

        # --- spatial profile ψ(r) = |E_pol(r)| / max|E_pol| ---
        E_t = self.sim_data[monitor_name]
        component_map = {
            "Ex": E_t.Ex.isel(t=-1),
            "Ey": E_t.Ey.isel(t=-1),
            "Ez": E_t.Ez.isel(t=-1),
        }
        E_pol = np.abs(component_map[polarization_component])
        psi = E_pol / (float(E_pol.max()) + 1e-20)

        # --- spatially resolved g ---
        g_map = g_max * psi

        if plot:
            g_plot = g_map / (2 * np.pi * 1e9)  # rad/s → GHz

            fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)

            if monitor_name == "Field_Profile_Monitor_x":
                im = g_plot.plot(ax=ax, cmap="inferno", add_colorbar=False,
                            x="y", y="z")
            if monitor_name == "Field_Profile_Monitor_y":
                im = g_plot.plot(ax=ax, cmap="inferno", add_colorbar=False,
                            x="x", y="z")
            if monitor_name == "Field_Profile_Monitor_z":
                im = g_plot.plot(ax=ax, cmap="inferno", add_colorbar=False,
                            x="x", y="y")

            fig.colorbar(im, ax=ax, label="g / 2π (GHz)")

            # contour lines
            if contour_ghz is not None:
                if isinstance(contour_ghz, (int, float)):
                    contour_ghz = [contour_ghz]
                g_2d = g_plot.squeeze()  # drop any size-1 dimensions like x
                cs = ax.contour(
                    g_2d.y, g_2d.z, g_2d.values.T,
                    levels=contour_ghz,
                    colors="cyan", linewidths=1.2, linestyles="--",
                )
                ax.clabel(cs, fmt="%.2f GHz", fontsize=9, colors="cyan")

            outline = Rectangle(
                (-self.context["width"] / 2, -self.context["thickness"] / 2),
                self.context["width"], self.context["thickness"],
                linewidth=1.5, edgecolor="white", facecolor="none",
                alpha=0.8, linestyle="--",
            )
            ax.add_patch(outline)
            ax.set_xlabel("y (µm)")
            ax.set_ylabel("z (µm)")
            ax.set_title(f"g(r) / 2π (GHz) — {monitor_name}")
            ax.set_aspect("equal")

            plt.show()

        return g_map, float(g_max)

    def cooperativity(self, dipole_moment, monitor_name="Field_Profile_Monitor_x",
                    polarization_component="Ey", channel=None,
                    fiber_efficiency=0.99, plot=False, contour_levels=None,
                    position=None):
        """Compute cooperativity C(r) = 4g(r)² / (κ · γ) over the monitor plane.

        Parameters
        ----------
        dipole_moment : float
            Transition dipole moment (C·m).
        monitor_name : str
            Which field profile monitor to use.
        polarization_component : str
            Field component aligned with the dipole.
        channel : str or None
            If None, use total κ. If e.g. "-x", use κ for that channel only.
        fiber_efficiency : float
            Fiber coupling efficiency (default 0.99).
        plot : bool
            If True, show a 2D colormap of C(r).
        contour_levels : list of float or None
            Draw contour lines at these cooperativity values.
        position : dict or None
            If provided, also print point values at this location.

        Returns
        -------
        C_map : xarray.DataArray
            Cooperativity map C(r).
        C_max : float
            Peak cooperativity.
        """
        self._require_data()

        gamma = self.resonant_omega_c**3 * dipole_moment**2 / (3 * pi * epsilon_0 * hbar * C0_m**3)

        g_map, g_max = self.coupling_g(
            dipole_moment=dipole_moment,
            monitor_name=monitor_name,
            polarization_component=polarization_component)


        C_map = 4 * g_map**2 / (self.kappa_tot * gamma)
        C_max = float(C_map.max())


        kappa_out = self.kappa_dir.get(channel, None)
        eta_map = kappa_out / self.kappa_dir.get('total') * C_map / (C_map + 1) * fiber_efficiency

        # --- optional point query ---
        if position is not None:
            g_at_r = float(g_map.sel(**position, method="nearest"))
            C_at_r = float(C_map.sel(**position, method="nearest"))
            eta_at_r = float(eta_map.sel(**position, method="nearest"))
            print(f"  g(r) / 2π = {g_at_r / (2 * pi * 1e9):.4f} GHz")
            print(f"  γ / 2π    = {gamma / (2 * pi * 1e6):.4f} MHz")
            print(f"  Resonance Calculation κ_tot / 2π    = {self.kappa_tot / (2 * pi * 1e9):.4f} GHz")
            print(f"  C(r)      = {C_at_r:.2f}")
            
            print(f"  Directional Calculation κ_tot / 2π    = {self.kappa_dir.get('total') / (2 * pi * 1e9):.4f} GHz")
            if channel is not None:
                print(f" Directional Calculation κ_{channel} / 2π = {kappa_out / (2 * pi * 1e9):.4f} GHz")
            print(f"  η(r)      = {eta_at_r * 100:.2f}%  (fiber eff. {fiber_efficiency * 100:.1f}%)")

        # --- plot ---
        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), tight_layout=True)

            axis_map = {
                "Field_Profile_Monitor_x": ("y", "z"),
                "Field_Profile_Monitor_y": ("x", "z"),
                "Field_Profile_Monitor_z": ("x", "y"),
            }
            xax, yax = axis_map[monitor_name]

            for ax, data, label, cmap in [
                (axes[0], C_map, "C(r)", "inferno"),
                (axes[1], eta_map * 100, "η(r) (%)", "viridis"),
            ]:
                im = data.plot(ax=ax, cmap=cmap, add_colorbar=False,
                            x=xax, y=yax)
                fig.colorbar(im, ax=ax, label=label)

                if contour_levels is not None and label == "C(r)":
                    if isinstance(contour_levels, (int, float)):
                        contour_levels = [contour_levels]
                    d2d = data.squeeze()
                    coords = (getattr(d2d, xax), getattr(d2d, yax))
                    cs = ax.contour(coords[0], coords[1], d2d.values.T,
                                    levels=contour_levels,
                                    colors="cyan", linewidths=1.2, linestyles="--")
                    ax.clabel(cs, fmt="%.1f", fontsize=9, colors="cyan")

                outline = Rectangle(
                    (-self.context["width"] / 2, -self.context["thickness"] / 2),
                    self.context["width"], self.context["thickness"],
                    linewidth=1.5, edgecolor="white", facecolor="none",
                    alpha=0.8, linestyle="--",
                )
                ax.add_patch(outline)
                ax.set_xlabel(f"{xax} (µm)")
                ax.set_ylabel(f"{yax} (µm)")
                ax.set_aspect("equal")

            axes[0].set_title(f"Cooperativity C(r) — {monitor_name}")
            axes[1].set_title(f"Collection efficiency η(r) — {monitor_name}")
            plt.show()

        return C_map, C_max
    
    # ──────────────────────────────────────────────────────────────────
    # Gaussian Fit
    # ──────────────────────────────────────────────────────────────────

    def plot_mode_along_beam(self):
        self._require_data()

        # Trigger lazy resonance analysis if needed
        _ = self.resonant_frequency

        pol = self.context["polarization"]
        monitor_axis_map = {"Ey": "y", "Ez": "z", "Hy": "y", "Hz": "z"}
        if pol not in monitor_axis_map:
            raise ValueError(f"Polarization '{pol}' not supported for mode plot.")

        axis = monitor_axis_map[pol]
        mon_data = self.sim_data[f"Field_Profile_Monitor_{axis}"]
        field = getattr(mon_data, pol)
        field_1d = field.isel(t=-1)

        if axis == "y":
            field_line = field_1d.sel(z=0, method="nearest")
        else:
            field_line = field_1d.sel(y=0, method="nearest")

        x = field_line.x.values
        vals = field_line.values.flatten().real
        vmax_val = np.max(np.abs(vals))
        if vmax_val > 0:
            vals = vals / vmax_val

        def gauss_sinusoid(x, A, sigma, k, phi, offset):
            return (A * np.exp(-x**2 / (2 * sigma**2))
                    * np.sin(k * x + phi) + offset)

        A0 = 1.0
        sigma0 = self.sim_size[0] / 8 if self.sim_size is not None else 5.0

        if self.nanobeam_medium is not None:
            n_eff_est = np.sqrt(self.nanobeam_medium.permittivity.real)
        else:
            n_eff_est = np.sqrt(
                self.sim.structures[0].medium.permittivity.real
            )
        k0 = 2 * np.pi * self.resonant_frequency / C0 * n_eff_est

        try:
            popt, pcov = curve_fit(
                gauss_sinusoid, x, vals,
                p0=[A0, sigma0, k0, 0.0, 0.0],
                bounds=(
                    [-np.inf, 0, 0, -np.pi, -np.inf],
                    [ np.inf, np.inf, np.inf, np.pi, np.inf],
                ),
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
        ax.plot(x, vals, color="steelblue", linewidth=0.8,
                label=f"{pol} along x (normalised)")

        x_fit = np.linspace(x.min(), x.max(), 2000)
        ax.plot(x_fit, gauss_sinusoid(x_fit, *popt), "--", color="red",
                linewidth=1.5, label="Gaussian × sin fit")

        envelope = A * np.exp(-x_fit**2 / (2 * sigma**2))
        ax.plot(x_fit,  envelope + offset, ":", color="orange", linewidth=1.2,
                label="+envelope")
        ax.plot(x_fit, -envelope + offset, ":", color="orange", linewidth=1.2,
                label="−envelope")

        if self.defect_start is not None and self.defect_end is not None:
            ax.axvspan(self.defect_start, self.defect_end, alpha=0.1,
                       color="red", label="Defect region")

        ax.axhline(0, color="gray", ls="-", alpha=0.3)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel(f"{pol} (normalised)")
        ax.set_title(
            f"Mode profile — f_res = {self.resonant_frequency:.4e} Hz, "
            f"Q = {self.Q:.0f}"
        )
        ax.legend()
        plt.tight_layout()
        plt.show()

        print(f"Fit: A={A:.4f}, σ={abs(sigma):.4f} µm, "
              f"k={k:.2f} rad/µm, φ={phi:.3f}")
        return (popt, pcov) if fit_success else (popt, None)

    # ──────────────────────────────────────────────────────────────────
    # Plots
    # ──────────────────────────────────────────────────────────────────

    def plot_results(self):
        self._require_data()

        e_components = ["Ex", "Ey", "Ez"]
        h_components = ["Hx", "Hy", "Hz"]

        for axis in ("x", "y", "z"):
            monitor_name = f"Field_Profile_Monitor_{axis}"
            if monitor_name not in self.sim_data.monitor_data:
                continue

            mon_data = self.sim_data[monitor_name]
            t_last = mon_data.Ex.t[-1]

            fig, axes = plt.subplots(1, 3, figsize=(18, 4))
            fig.suptitle(
                f"{monitor_name} — |E| at t = {t_last:.2e} s", fontsize=12,
            )
            for i, comp in enumerate(e_components):
                self.sim_data.plot_field(
                    monitor_name, comp, val="abs", t=t_last, ax=axes[i],
                )
                axes[i].set_title(f"|{comp}|")
            plt.tight_layout()
            plt.show()

            fig, axes = plt.subplots(1, 3, figsize=(18, 4))
            fig.suptitle(
                f"{monitor_name} — |H| at t = {t_last:.2e} s", fontsize=12,
            )
            for i, comp in enumerate(h_components):
                self.sim_data.plot_field(
                    monitor_name, comp, val="abs", t=t_last, ax=axes[i],
                )
                axes[i].set_title(f"|{comp}|")
            plt.tight_layout()
            plt.show()

    def plot_simulation(self):
        self._require_sim()

        positions = self.beam_layout["positions"]
        lattices = self.beam_layout["lattice"]
        hole_params = self.beam_layout["hole_params"]

        if isinstance(positions, list):
            positions = np.array(positions)
        if isinstance(lattices, list):
            lattices = np.array(lattices)
        if isinstance(hole_params, list):
            hole_params = np.array(hole_params)

        n_holes = len(positions)
        indices = np.arange(n_holes)
        n_params = hole_params.shape[1] if hole_params.ndim > 1 else 1
        n_plots = 2 + n_params

        n_lt = self.n_cells["N_left_taper"]
        n_lm = self.n_cells["N_left_mirror"]
        n_d  = self.n_cells["N_defect"]
        n_rm = self.n_cells["N_right_mirror"]
        n_rt = self.n_cells["N_right_taper"]

        sections = [
            (0,                        n_lt,                            "Left Taper",   "gold"),
            (n_lt,                     n_lt + n_lm,                     "Left Mirror",  "royalblue"),
            (n_lt + n_lm,              n_lt + n_lm + n_d,               "Defect",       "crimson"),
            (n_lt + n_lm + n_d,        n_lt + n_lm + n_d + n_rm,        "Right Mirror", "royalblue"),
            (n_lt + n_lm + n_d + n_rm, n_lt + n_lm + n_d + n_rm + n_rt, "Right Taper",  "gold"),
        ]

        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots), sharex=True)

        for ax in axes:
            for start, end, label, color in sections:
                ax.axvspan(start - 0.5, end - 0.5, alpha=0.15, color=color)

        axes[0].plot(indices, positions, ".-", color="steelblue",
                     markersize=4, linewidth=0.8)
        axes[0].set_ylabel("x position (µm)")
        axes[0].set_title(f"Hole positions ({n_holes} holes)")
        axes[0].axhline(0, color="red", ls="--", alpha=0.3,
                        label="cavity center")
        axes[0].legend(fontsize=8)

        axes[1].plot(indices, lattices, ".-", color="coral",
                     markersize=4, linewidth=0.8)
        axes[1].set_ylabel("Lattice constant (µm)")
        axes[1].set_title("Lattice constant vs hole index")

        if hole_params.ndim == 1:
            hole_params = hole_params[:, np.newaxis]
        for j in range(n_params):
            ax = axes[2 + j]
            ax.plot(indices, hole_params[:, j], ".-", color="black",
                    markersize=4, linewidth=0.8)
            ax.set_ylabel(f"param_{j} (µm)")
            ax.set_title(f"param_{j} vs hole index")

        axes[-1].set_xlabel("Hole index")
        plt.tight_layout()
        plt.show()

        # Defect zoom
        fig, ax = plt.subplots(figsize=(14, 4))
        self.sim.plot(z=0, ax=ax)
        rect = Rectangle(
            (self.defect_start, -self.context["width"] / 2),
            width=self.defect_length,
            height=self.context["width"],
            linewidth=2, edgecolor="red", facecolor="red", alpha=0.15,
            label="Defect / mesh override region",
        )
        ax.add_patch(rect)
        padding = 0.5
        ax.set_xlim(self.defect_start - padding, self.defect_end + padding)
        ax.set_ylim(-self.context["width"] * 1.5, self.context["width"] * 1.5)
        ax.legend()
        ax.set_title("Defect region — zoom")
        plt.show()

        # Three-view
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        self.sim.plot(z=0, ax=axes[0])
        axes[0].set_title("Top view (z = 0)")
        self.sim.plot(y=0, ax=axes[1])
        axes[1].set_title("Side view (y = 0)")
        self.sim.plot(x=0, ax=axes[2])
        axes[2].set_title("Cross-section (x = 0)")
        plt.tight_layout()
        plt.show()

    # ──────────────────────────────────────────────────────────────────
    # Animation
    # ──────────────────────────────────────────────────────────────────

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
        self._require_data()

        if monitor_name not in self.sim_data.monitor_data:
            available = list(self.sim_data.monitor_data.keys())
            raise KeyError(
                f"Monitor '{monitor_name}' not found. "
                f"Available: {available}"
            )

        mon_data = self.sim_data[monitor_name]
        if not hasattr(mon_data, field_component):
            raise KeyError(
                f"Field component '{field_component}' not found on "
                f"'{monitor_name}'. Available: Ex, Ey, Ez, Hx, Hy, Hz"
            )

        field = getattr(mon_data, field_component)

        # Determine 2D slice axes
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
                f"Monitor '{monitor_name}' is not a 2D slice. "
                f"Dims: {[(d, field.sizes[d]) for d in spatial_dims]}"
            )

        field_2d = field.squeeze(dim=slice_axis, drop=True)

        # Sub-sample time
        t_vals = field_2d.t.values
        n_total = len(t_vals)
        if n_frames is not None and n_frames < n_total:
            indices = np.linspace(0, n_total - 1, n_frames, dtype=int)
        else:
            indices = np.arange(n_total)

        # Colour limits from sampled frames
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

        ax0_vals = field_2d.coords[plot_axes[0]].values
        ax1_vals = field_2d.coords[plot_axes[1]].values

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

        title_template = (
            title or "{component} | {monitor} | t = {t:.3f} ps"
        )

        def _update(frame_idx):
            ti = int(indices[frame_idx])
            data = field_2d.isel(t=ti).values.real.T
            im.set_array(data.ravel())
            t_ps = float(t_vals[ti]) * 1e12
            ax.set_title(title_template.format(
                t=t_ps, component=field_component, monitor=monitor_name,
            ))
            return [im]

        anim = animation.FuncAnimation(
            fig, _update, frames=len(indices),
            interval=1000 / fps, blit=False,
        )
        anim.save(output_path, writer="pillow", fps=fps, dpi=dpi)
        plt.close(fig)

        print(f"GIF saved: {output_path} "
              f"({len(indices)} frames, {fps} fps, {dpi} dpi)")
        return output_path