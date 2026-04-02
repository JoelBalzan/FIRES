# This file defines the dataclass-based schema for the FIRES configuration.

### IMPORTS ###
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


### META ###
@dataclass
class Meta:
    name: str = "run"
    seed: Optional[int] = None
    version: int = 1

### SIMULATION GRID ###
@dataclass
class SimulationGrid:
    f_start_MHz: float
    f_end_MHz: float
    df_MHz: float

    t_start_ms: float
    t_end_ms: float
    dt_ms: float

    reference_freq_MHz: float

### SCATTERING ###
@dataclass
class Scattering:
    index: float

### SCINTILLATION ###
@dataclass
class Scintillation:
    enable: bool = True
    timescale_s: float = 300
    bandwidth_Hz: float = 1.5e6
    derive_from_tau: bool = False

    N_images: int = 5000
    theta_extent: float = 3.0
    return_field: bool = False

@dataclass
class Propagation:
    scattering: Scattering
    scintillation: Scintillation

### AMPLITUDE DISTRIBUTIONS ###
@dataclass
class PowerLawAmp:
    alpha: float
    xmin_scale: float
    xmax_scale: float

@dataclass
class LogNormalAmp:
    sigma: float

@dataclass
class UniformAmp:
    low_scale: float
    high_scale: float

@dataclass
class AmplitudeDistribution:
    type: Literal["normal", "lognormal", "powerlaw", "uniform"]
    powerlaw: Optional[PowerLawAmp] = None
    lognormal: Optional[LogNormalAmp] = None
    uniform: Optional[UniformAmp] = None

### MICROSHOT SCATTER ###
@dataclass
class MicroshotScatter:
    t0_sigma_ms: float = 0.0
    width_sigma_ms: float = 0.0
    amplitude_sigma: float = 0.0
    spectral_index_sigma: float = 0.0
    tau_sigma_ms: float = 0.0
    dm_sigma: float = 0.0
    rm_sigma: float = 0.0
    pa_sigma_deg: float = 0.0
    lfrac_sigma: float = 0.0
    vfrac_sigma: float = 0.0
    dpa_sigma: float = 0.0
    band_centre_sigma: float = 0.0
    band_width_sigma: float = 0.0

### MICROSHOT POPULATION ###
@dataclass
class Microshots:
    N: int
    width_frac_low: float
    width_frac_high: float

### GAUSSIAN COMPONENTS ###
@dataclass
class GaussianComponent:
    t0_ms: float
    width_ms: float
    amplitude_Jy: float
    spectral_index: float
    tau_ms: float
    dm: float
    rm: float
    pa_deg: float
    lfrac: float
    vfrac: float
    dpa_deg_per_ms: float
    band_centre_MHz: float
    band_width_MHz: float

    microshots: Microshots
    microshot_scatter: MicroshotScatter
    amplitude_distribution: AmplitudeDistribution

### EMISSION MODEL ###
@dataclass
class Emission:
    model: Literal["gaussian_microshot"]
    components: List[GaussianComponent]

### SWEEP CONFIG ###
@dataclass
class SweepParameter:
    component: int
    name: str
    start: float
    stop: float
    step: float
    log_steps: Optional[int] = None

@dataclass
class Sweep:
    enable: bool
    mode: Literal["none", "mean", "sd"]
    parameter: SweepParameter

@dataclass
class Analysis:
    sweep: Sweep
    buffer_fraction: float = 1.0


### TOP-LEVEL GROUPS ###
@dataclass
class Simulation:
    grid: SimulationGrid


@dataclass
class Observation:
    sefd: float = 0.0
    target_snr: Optional[float] = None
    baseline_correct: Optional[str] = None


@dataclass
class Numerics:
    n_cpus: int = 1
    nseed: int = 1


@dataclass
class Output:
    write: bool = False
    mode: str = "full"
    directory: str = "simfrbs"


@dataclass
class FiresConfig:
    meta: Meta
    simulation: Simulation
    propagation: Propagation
    emission: Emission
    analysis: Analysis
    observation: Observation = field(default_factory=Observation)
    numerics: Numerics = field(default_factory=Numerics)
    output: Output = field(default_factory=Output)


def _require(section: Dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise ValueError(f"Missing required key '{where}.{key}' in fires.toml")
    return section[key]


def _as_component_list(raw_components: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_components, list):
        return raw_components
    if isinstance(raw_components, dict):
        return [raw_components]
    raise ValueError("'emission.components' must be a table or array of tables")


def _parse_optional_positive_float(value: Any) -> Optional[float]:
    """Return positive float value, or None for disabled/invalid entries."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null", "false", "off", "0"}:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if (v > 0) else None


def _as_bool(value: Any, default: bool = False) -> bool:
    """Parse booleans robustly, including string and numeric inputs."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off", "none", "null", ""}:
            return False
    return bool(value)

def _parse_optional_positive_int(value: Any) -> Optional[int]:
    """Return positive int value, or None for disabled/invalid entries."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null", "false", "off", "0"}:
        return None
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return v if (v > 0) else None

def parse_fires_config(raw: Dict[str, Any]) -> FiresConfig:
    """Parse raw fires.toml dictionary into a strongly typed FiresConfig."""
    if not isinstance(raw, dict):
        raise ValueError("fires.toml root must be a TOML table")

    meta_raw = raw.get("meta", {})
    meta = Meta(
        name=str(meta_raw.get("name", "run")),
        seed=meta_raw.get("seed", None),
        version=int(meta_raw.get("version", 1)),
    )

    sim_raw = _require(raw, "simulation", "root")
    grid_raw = _require(sim_raw, "grid", "simulation")
    sim = Simulation(
        grid=SimulationGrid(
            f_start_MHz=float(_require(grid_raw, "f_start_MHz", "simulation.grid")),
            f_end_MHz=float(_require(grid_raw, "f_end_MHz", "simulation.grid")),
            df_MHz=float(_require(grid_raw, "df_MHz", "simulation.grid")),
            t_start_ms=float(_require(grid_raw, "t_start_ms", "simulation.grid")),
            t_end_ms=float(_require(grid_raw, "t_end_ms", "simulation.grid")),
            dt_ms=float(_require(grid_raw, "dt_ms", "simulation.grid")),
            reference_freq_MHz=float(_require(grid_raw, "reference_freq_MHz", "simulation.grid")),
        )
    )

    prop_raw = _require(raw, "propagation", "root")
    sc_raw = _require(prop_raw, "scattering", "propagation")
    scint_raw = _require(prop_raw, "scintillation", "propagation")
    propagation = Propagation(
        scattering=Scattering(index=float(_require(sc_raw, "index", "propagation.scattering"))),
        scintillation=Scintillation(
            enable=_as_bool(scint_raw.get("enable", True), default=True),
            timescale_s=float(scint_raw.get("timescale_s", 300.0)),
            bandwidth_Hz=float(scint_raw.get("bandwidth_Hz", 1.5e6)),
            derive_from_tau=_as_bool(scint_raw.get("derive_from_tau", False), default=False),
            N_images=int(scint_raw.get("N_images", 5000)),
            theta_extent=float(scint_raw.get("theta_extent", 3.0)),
            return_field=_as_bool(scint_raw.get("return_field", False), default=False),
        ),
    )

    em_raw = _require(raw, "emission", "root")
    raw_components = _as_component_list(_require(em_raw, "components", "emission"))
    components: List[GaussianComponent] = []
    for idx, comp in enumerate(raw_components):
        where = f"emission.components[{idx}]"
        micro_raw = _require(comp, "microshots", where)
        scatter_raw = comp.get("microshot_scatter", {})
        amp_raw = _require(comp, "amplitude_distribution", where)

        amp_dist = AmplitudeDistribution(type=str(_require(amp_raw, "type", f"{where}.amplitude_distribution")).lower())
        if isinstance(amp_raw.get("powerlaw"), dict):
            pw = amp_raw["powerlaw"]
            amp_dist.powerlaw = PowerLawAmp(
                alpha=float(_require(pw, "alpha", f"{where}.amplitude_distribution.powerlaw")),
                xmin_scale=float(_require(pw, "xmin_scale", f"{where}.amplitude_distribution.powerlaw")),
                xmax_scale=float(_require(pw, "xmax_scale", f"{where}.amplitude_distribution.powerlaw")),
            )
        if isinstance(amp_raw.get("lognormal"), dict):
            ln = amp_raw["lognormal"]
            amp_dist.lognormal = LogNormalAmp(sigma=float(_require(ln, "sigma", f"{where}.amplitude_distribution.lognormal")))
        if isinstance(amp_raw.get("uniform"), dict):
            uu = amp_raw["uniform"]
            amp_dist.uniform = UniformAmp(
                low_scale=float(_require(uu, "low_scale", f"{where}.amplitude_distribution.uniform")),
                high_scale=float(_require(uu, "high_scale", f"{where}.amplitude_distribution.uniform")),
            )

        components.append(
            GaussianComponent(
                t0_ms=float(_require(comp, "t0_ms", where)),
                width_ms=float(_require(comp, "width_ms", where)),
                amplitude_Jy=float(_require(comp, "amplitude_Jy", where)),
                spectral_index=float(_require(comp, "spectral_index", where)),
                tau_ms=float(_require(comp, "tau_ms", where)),
                dm=float(_require(comp, "dm", where)),
                rm=float(_require(comp, "rm", where)),
                pa_deg=float(_require(comp, "pa_deg", where)),
                lfrac=float(_require(comp, "lfrac", where)),
                vfrac=float(_require(comp, "vfrac", where)),
                dpa_deg_per_ms=float(_require(comp, "dpa_deg_per_ms", where)),
                band_centre_MHz=float(_require(comp, "band_centre_MHz", where)),
                band_width_MHz=float(_require(comp, "band_width_MHz", where)),
                microshots=Microshots(
                    N=int(_require(micro_raw, "N", f"{where}.microshots")),
                    width_frac_low=float(_require(micro_raw, "width_frac_low", f"{where}.microshots")),
                    width_frac_high=float(_require(micro_raw, "width_frac_high", f"{where}.microshots")),
                ),
                microshot_scatter=MicroshotScatter(
                    t0_sigma_ms=float(scatter_raw.get("t0_sigma_ms", 0.0)),
                    width_sigma_ms=float(scatter_raw.get("width_sigma_ms", 0.0)),
                    amplitude_sigma=float(scatter_raw.get("amplitude_sigma", 0.0)),
                    spectral_index_sigma=float(scatter_raw.get("spectral_index_sigma", 0.0)),
                    tau_sigma_ms=float(scatter_raw.get("tau_sigma_ms", 0.0)),
                    dm_sigma=float(scatter_raw.get("dm_sigma", 0.0)),
                    rm_sigma=float(scatter_raw.get("rm_sigma", 0.0)),
                    pa_sigma_deg=float(scatter_raw.get("pa_sigma_deg", 0.0)),
                    lfrac_sigma=float(scatter_raw.get("lfrac_sigma", 0.0)),
                    vfrac_sigma=float(scatter_raw.get("vfrac_sigma", 0.0)),
                    dpa_sigma=float(scatter_raw.get("dpa_sigma", 0.0)),
                    band_centre_sigma=float(scatter_raw.get("band_centre_sigma", 0.0)),
                    band_width_sigma=float(scatter_raw.get("band_width_sigma", 0.0)),
                ),
                amplitude_distribution=amp_dist,
            )
        )

    emission = Emission(
        model=str(_require(em_raw, "model", "emission")),
        components=components,
    )

    an_raw = _require(raw, "analysis", "root")
    sweep_raw = _require(an_raw, "sweep", "analysis")
    sweep_param_raw = _require(sweep_raw, "parameter", "analysis.sweep")
    # Build Analysis and include optional buffer_fraction (preferred location).
    analysis = Analysis(
        sweep=Sweep(
            enable=_as_bool(_require(sweep_raw, "enable", "analysis.sweep"), default=False),
            mode=str(_require(sweep_raw, "mode", "analysis.sweep")),
            parameter=SweepParameter(
                component=int(_require(sweep_param_raw, "component", "analysis.sweep.parameter")),
                name=str(_require(sweep_param_raw, "name", "analysis.sweep.parameter")),
                start=float(_require(sweep_param_raw, "start", "analysis.sweep.parameter")),
                stop=float(_require(sweep_param_raw, "stop", "analysis.sweep.parameter")),
                step=float(_require(sweep_param_raw, "step", "analysis.sweep.parameter")),
                log_steps=_parse_optional_positive_int(sweep_param_raw.get("log_steps", None)),
            ),
        ),
        buffer_fraction=float(an_raw.get("buffer_fraction", raw.get("observation", {}).get("buffer_fraction", 1.0))),
    )

    obs_raw = raw.get("observation", {})
    observation = Observation(
        sefd=float(obs_raw.get("sefd", 0.0)),
        target_snr=_parse_optional_positive_float(obs_raw.get("target_snr", None)),
        baseline_correct=(
            obs_raw.get("baseline_correct", None)
            if str(obs_raw.get("baseline_correct", None)).strip().lower() not in ("false", "none", "null")
            else None
        ),
    )

    num_raw = raw.get("numerics", {})
    numerics = Numerics(
        n_cpus=int(num_raw.get("n_cpus", 1)),
        nseed=int(num_raw.get("nseed", 1)),
    )

    out_raw = raw.get("output", {})
    output = Output(
        write=_as_bool(out_raw.get("write", False), default=False),
        mode=str(out_raw.get("mode", "full")),
        directory=str(out_raw.get("directory", "simfrbs")),
    )

    return FiresConfig(
        meta=meta,
        simulation=sim,
        propagation=propagation,
        emission=emission,
        analysis=analysis,
        observation=observation,
        numerics=numerics,
        output=output,
    )    