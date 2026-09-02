# FIRES Configuration Reference

This document describes every section and parameter in `fires.toml`, the master configuration file for FIRES.

---

## Table of Contents

- [`[meta]`](#meta)
- [`[simulation.grid]`](#simulationgrid)
- [`[propagation.scattering]`](#propagationscattering)
- [`[propagation.rm]`](#propagationrm)
- [`[propagation.chain]`](#propagationchain)
- [`[propagation.derotate]`](#propagationderotate)
- [`[propagation.scintillation]`](#propagationscintillation)
- [`[emission]`](#emission)
- [`[emission.fold]`](#emissionfold)
- [`[emission.rvm_swing]`](#emissionrvm_swing)
- [`[[emission.components]]`](#emissioncomponents)
- [`[emission.components.microshots]`](#emissioncomponentsmicroshots)
- [`[emission.components.microshot_scatter]`](#emissioncomponentsmicroshot_scatter)
- [`[emission.components.amplitude_distribution]`](#emissioncomponentsamplitude_distribution)
- [`[analysis]`](#analysis)
- [`[analysis.sweep]`](#analysissweep)
- [`[observation]`](#observation)
- [`[numerics]`](#numerics)
- [`[output]`](#output)

---

## `[meta]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | string | `"FIRES"` | Run identifier. Used in output filenames and log headers. |
| `seed` | int / null | `3` | Global RNG seed for reproducibility. Set to an integer for deterministic results across runs. Omit or set to `null` for a random seed drawn from OS entropy. |
| `version` | int | `1` | Config schema version. Reserved for future migration logic; currently unused. |

**Where it flows:** `seed` is passed into `generate_frb()` → `_process_task()` / `psn_dspec()` → `_init_seed()` which calls `np.random.seed(seed)`. `name` appears only in logs and output paths.

---

## `[simulation.grid]`

Defines the time-frequency grid on which the dynamic spectrum is computed.

| Key | Type | Description |
|-----|------|-------------|
| `f_start_MHz` | float | Start of the frequency band (MHz). |
| `f_end_MHz` | float | End of the frequency band (MHz). |
| `df_MHz` | float | Frequency channel width (MHz). Determines `n_freq = floor((f_end - f_start) / df) + 1`. |
| `t_start_ms` | float | Start of the time window relative to grid reference (ms). |
| `t_end_ms` | float | End of the time window (ms). |
| `dt_ms` | float | Time bin width (ms). Determines `n_time = floor((t_end - t_start) / dt) + 1`. |
| `reference_freq_MHz` | float | Reference frequency (MHz) for spectral-index scaling and the scattering law. |

**Where it flows:** These values are extracted in `_master_to_internal()` (genfrb.py:81-88) into `sim_params`, then used to construct `freq_mhz` and `time_ms` arrays via `np.arange()`. They also populate `dspecParams`. The arrays are the fundamental axes of the `(4, n_freq, n_time)` dynamic spectrum.

**Choosing grid parameters:**
- `df_MHz` should be small enough to resolve spectral features but large enough to keep memory manageable. The scattering kernel convolution operates per frequency channel, so finer `df` increases runtime.
- `dt_ms` should be smaller than the narrowest time feature (microshot width, scattering timescale) to avoid discretisation artefacts.
- `t_start_ms` / `t_end_ms` should extend far enough before/after the burst to capture off-pulse noise baselines (the `buffer_fraction` parameter determines how much of the window is guard band).

---

## `[propagation.scattering]`

Controls the temporal scattering (pulse broadening) model.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `index` | float | `-4.85` | Scattering law exponent: `tau(nu) = tau(nu_ref) * (nu / nu_ref)^index`. Typical ISM values are -4 (Kolmogorov) to -4.4 (NE2001). Some FRB fits yield -3.22. |
| `screen` | string | `"thin"` | Scattering screen geometry: `"thin"`, `"thick"`, or `"uniform"`. Affects the shape of the broadening kernel. See Williamson (1972) MNRAS 157, 55. |

**Where it flows:** Stored in `prop_dict["scattering_index"]` and `prop_dict["scattering_screen"]`. Used in `scatter_dspec()` (dspec.py) which convolves each frequency channel with an exponential kernel `h(t) = (1/tau) * exp(-t/tau)`. The per-channel tau is computed as:

```python
tau_cms = tau_ref * (freq_mhz / ref_freq_mhz) ** index
```

The screen type changes the kernel:
- `"thin"`: exponential kernel
- `"thick"`: different functional form implemented in `_build_irf_fft()`
- `"uniform"`: another screen model variant

---

## `[propagation.rm]`

Global Faraday rotation applied to the full dynamic spectrum.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `RM` | float | `0.0` | Global rotation measure (rad m^-2). Rotates the polarisation angle as `chi = chi0 + RM * (lambda^2 - lambda_ref^2)`. |
| `order` | string | `"pre"` | When to apply RM relative to scattering: `"pre"` (before scattering convolution) or `"post"` (after scattering). |

**Where it flows:** Stored in `prop_dict["RM"]` and `prop_dict["order"]`. Applied in `psn_dspec()` (genfns.py:883-897) via `rm_correct_dspec()`.

**Note:** This is *global* RM applied to the summed dspec. There is also a *per-microshot* RM (`rm` in `[[emission.components]]`) applied individually to each microshot before summing, enabling Burn-law depolarisation.

**Note:** If [`[propagation.chain]`](#propagationchain) is configured, it takes precedence over `[propagation.scattering]` and `[propagation.rm]` and this section is ignored.

---

## `[propagation.chain]`

**Test mode** — an ordered list of scattering and RM screens applied sequentially. This gives full manual control over the exact sequence of propagation screens and overrides both [`[propagation.scattering]`](#propagationscattering) and [`[propagation.rm]`](#propagationrm) when present. Use it to interleave *N* scattering screens and *M* RM screens in any order.

| Key | Type | Description |
|-----|------|-------------|
| `steps` | array of inline tables | Ordered list of screen steps, each either `type = "scatter"` or `type = "rm"`. |

### Scatter step

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | — | `"scatter"`. |
| `screen` | string | `"thin"` | Screen geometry: `"thin"`, `"thick"`, or `"uniform"`. |
| `index` | float | `-4.0` | Scattering law exponent: `tau(f) = tau_ms * (f / nu_ref)^index`. |
| `tau_ms` | float | — (required) | Scattering timescale (ms) at the reference frequency. Convolves each channel with the screen kernel. |

### RM step

| Key | Type | Description |
|-----|------|-------------|
| `type` | string | `"rm"`. |
| `RM` | float | — (required) | Rotation measure (rad m^-2). Rotates Q/U via `chi = chi0 - 2*RM*(lambda^2 - lambda_ref^2)`. |

**Example:** scatter → rotate → scatter → rotate:

```toml
[propagation.chain]
steps = [
    { type = "scatter", screen = "thin",  index = -4.0, tau_ms = 1.0 },
    { type = "rm",      RM = 100.0 },
    { type = "scatter", screen = "thick", index = -3.2, tau_ms = 0.5 },
    { type = "rm",      RM = -50.0 },
]
```

**Where it flows:** Parsed into `propagation.chain` (schema.py), converted to a runtime list by `_chain_to_internal()` (genfrb.py) into `prop_dict["chain"]`, and applied to the summed dspec by `apply_chain()` in genfns.py. All steps are linear operations (scattering convolution, RM rotation), so applying the chain to the summed dspec is equivalent to applying it per-component.

---

## `[propagation.derotate]`

Global switch controlling the automatic **RM detection + derotation** performed after noise injection. By default FIRES measures the RM of the simulated burst (via RM synthesis) and de-rotates it to maximize linear polarisation `L/I` — which effectively removes any injected RM from the output.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable` | bool | `true` | If `false`, the derotation rotation is **not** applied to the dspec, so any deliberately injected RM (e.g. from `[propagation.rm]` or a chain `rm` step) is preserved in the raw dynamic spectrum. The RM is **still measured and printed** either way. |

```toml
[propagation.derotate]
enable = false
```

RM is always measured (via RM synthesis) and logged; `enable` only controls whether the measured RM is then de-rotated out to zero position angle. Turning it off is particularly useful for **test mode** workflows where you inject a specific global/chain RM and want to inspect the rotated spectrum as-is, while still seeing the measured value in the log.

**Where it flows:** Parsed into `propagation.derotate.enable`, stored in `prop_dict["derotate"]`, and consulted by the derotation blocks in `psn_dspec()` (genfns.py) and `process_dspec()` (dspec.py).

---

## `[propagation.scintillation]`

Diffractive scintillation applied as a multiplicative gain field across the dynamic spectrum.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable` | bool | `true` | Master switch for scintillation modulation. |
| `timescale_s` | float | `300` | Scintillation timescale (seconds). Controls how fast the gain pattern varies along the time axis. |
| `bandwidth_Hz` | float | `1.5e6` | Scintillation bandwidth (Hz). The frequency coherence scale of the gain pattern. |
| `derive_from_tau` | bool | `false` | If `true`, ignore `bandwidth_Hz` and compute `nu_s = 1 / (2 * pi * tau_ref)` from the component's `tau_ms`. Useful for self-consistent scattering + scintillation. |
| `N_images` | int | `5000` | Number of phasor/image terms used in the screen synthesis. More terms = smoother pattern at higher computational cost. |
| `theta_extent` | float | `3.0` | Angular truncation parameter for the screen's point-source response. |
| `return_field` | bool | `false` | If `true`, return the complex electric field in addition to the intensity gain. Used for specialised polarimetric studies. |

**Where it flows:** Stored in `scint_dict`. The `simulate_scintillation()` C extension (scint/lib_ScintillationMaker.so) generates a complex field `E(t, nu)`, and the gain `g = |E|^2 / <|E|^2>` multiplies all four Stokes parameters in-place (genfns.py:61-106).

---

## `[emission]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | string | `"psn"` | Emission model selector. `"psn"` — standard polarised shot-noise model; `"fold"` — pulsar-fold mode that stacks `nfold` independent PSN realisations. |

**Dispatch logic (genfrb.py):**
- `model = "psn"`: calls `psn_dspec()` directly.
- `model = "fold"`: calls `fold_dspec()` which loops over `nfold` calls to `psn_dspec()` with staggered seeds and averages the output dspecs.

---

## `[emission.fold]`

Only meaningful when `model = "fold"`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `nfold` | int | `10` | Number of independent PSN realisations to stack. Each fold uses an independent RNG seed (`base_seed * nfold + i`). The dspec is the arithmetic mean across all folds; signal sums coherently while noise averages down as `sqrt(nfold)`. |

---

## `[emission.rvm_swing]`

Rotating-Vector-Model (Radhakrishnan & Cooke 1969) PA swing applied to the summed dynamic spectrum.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable` | bool | `false` | Enable RVM PA swing. |
| `alpha_deg` | float | `45.0` | Magnetic inclination angle (deg). Angle between the rotation axis and the magnetic dipole axis. |
| `beta_deg` | float | `5.0` | Impact angle (deg). Minimum angle between the line of sight and the magnetic axis. |
| `period_ms` | float | `1.0` | Rotational period (ms). Drives the phase `phi(t) = 2*pi*(t - phase0) / period`. |
| `phase0_ms` | float | `0.0` | Reference phase offset (ms). |
| `psi0_deg` | float | `0.0` | Constant PA offset (deg). Added to the RVM output. |

**RVM formula (`_rvm_pa_swing_deg`, genfns.py:505-531):**

```
zeta = alpha + beta
phi(t) = 2*pi*(t - phase0) / period
psi(t) = psi0 + arctan2(sin(alpha) * sin(phi),
                        sin(zeta)*cos(alpha) - cos(zeta)*sin(alpha)*cos(phi))
```

The swing is applied to Q/U via rotation by `2*psi(t)` in `_apply_rvm_swing_to_dspec()` (genfns.py:534-548).

---

## `[[emission.components]]`

Each component is a Gaussian envelope containing a population of microshots. Multiple components can be defined as an array of TOML tables.

| Key | Type | Description |
|-----|------|-------------|
| `t0_ms` | float | Component centroid arrival time (ms). Microshot `t0_i` are drawn from `N(t0, width / FWHM_factor)`. |
| `width_ms` | float | Gaussian envelope FWHM (ms). Defines the macro-pulse width that contains the microshots. |
| `amplitude_Jy` | float | Mean microshot amplitude (Jy). The actual per-microshot amplitude is drawn from the configured `amplitude_distribution` centred on this mean. |
| `spectral_index` | float | Power-law spectral index: `I_i(nu) = A_i * (nu / nu_ref)^spectral_index`. Applied per microshot. |
| `tau_ms` | float | Scattering timescale (ms) at `reference_freq_MHz`. Behaviour depends on `tau_sigma_ms` in `microshot_scatter`: if `tau_sigma_ms = 0`, this single tau is convolved with the full dspec after microshot summing; if `tau_sigma_ms > 0`, each microshot gets its own `tau_i` and is scattered individually before summing. |
| `dm` | float | Dispersion measure (pc cm^-3). Produces frequency-dependent time delays: `Delta_t(DM, nu) = 4.15 ms * DM * (nu^-2 - nu_ref^-2)`. |
| `rm` | float | Per-microshot rotation measure (rad m^-2). Applied to individual microshot polarisation angles with a reference-frequency offset: `chi_i = chi0 + RM_i * (lambda^2 - lambda_ref^2)`. When combined with `rm_sigma > 0`, this drives Burn-law depolarisation across the microshot population. |
| `pa_deg` | float | Mean polarisation position angle (deg). Per-microshot `PA_i ~ N(pa_deg, pa_sigma_deg)`. |
| `lfrac` | float | Desired linear polarisation fraction `L/I`. Together with `vfrac`, defines the polarisation state on the Poincaré sphere. The pair is rescaled to ensure `sqrt(L^2 + V^2) <= I^2`. |
| `vfrac` | float | Desired circular polarisation fraction `V/I`. Negative = left-hand circular (IEEE convention). |
| `dpa_deg_per_ms` | float | Intra-microshot PA sweep rate (deg/ms). Adds a linear time-dependent PA ramp across each microshot: `PA_i(t) = PA_i + dPA_i * (t - t0_i)`. |
| `band_centre_MHz` | float | Centre frequency of a Gaussian bandpass filter (MHz). `0` means use the median of `freq_mhz`. |
| `band_width_MHz` | float | FWHM of the Gaussian bandpass (MHz). `0` means no bandpass is applied. |

**Per-component semantics (genfns.py:787-866):**
For each of the `N` microshots in component `g`:
1. Draw random parameters from configured distributions (means, sigmas from `microshot_scatter`).
2. Compute `I_ft = norm_amp * gaussian(time) * spectral_profile(freq)`, where `norm_amp = A_i * (freq / ref_freq) ^ spec_idx_i`.
3. Apply DM delay via `_roll_rows()`.
4. Compute `Q_ft`, `U_ft`, `V_ft` from PA, lfrac, vfrac including Faraday rotation.
5. Optionally scatter per-microshot (if `tau_sigma_ms > 0`).
6. Add to the accumulating `dspec` array: `dspec[0:4] += [I, Q, U, V]`.

---

## `[emission.components.microshots]`

| Key | Type | Description |
|-----|------|-------------|
| `N` | int | Number of microshots per component. |
| `width_frac_low` | float | Lower bound of microshot FWHM as a fraction of the component `width_ms`. The TOML stores this as a fraction (e.g., `0.05` = 5 %), but the code internally converts it to a percentage by multiplying by 100 in `_master_to_internal()` (genfrb.py:125-126). |
| `width_frac_high` | float | Upper bound of microshot FWHM as a fraction of the component `width_ms`. |

Per-microshot width is drawn uniformly: `mg_width_i ~ Uniform(width * frac_low, width * frac_high)`.

---

## `[emission.components.microshot_scatter]`

Per-microshot Gaussian scatter (standard deviation) around each component mean. A non-zero sigma causes the corresponding microshot parameter to be drawn from `N(mean, sigma)`.

| Key | Default | Description |
|-----|---------|-------------|
| `t0_sigma_ms` | `0.0` | Scatter in microshot arrival time (ms). |
| `width_sigma_ms` | `0.0` | Scatter in microshot FWHM (ms). |
| `amplitude_sigma` | `0.0` | Scatter in microshot amplitude (Jy). Only used when `amplitude_distribution.type = "normal"`. |
| `spectral_index_sigma` | `0.0` | Scatter in spectral index. |
| `tau_sigma_ms` | `0.0` | Scatter in scattering timescale (ms). When > 0, each microshot gets its own `tau_i` and per-microshot scattering is applied individually instead of globally. |
| `dm_sigma` | `0.0` | Scatter in dispersion measure (pc cm^-3). |
| `rm_sigma` | `0.0` | Scatter in rotation measure (rad m^-2). Drives Burn-law depolarisation across the microshot ensemble. |
| `pa_sigma_deg` | `30` | Scatter in polarisation position angle (deg). The primary control for PA variance and depolarisation: larger values cause more PA scatter across microshots, reducing `L/I` and increasing `Var(PA)`. |
| `lfrac_sigma` | `0.0` | Scatter in `L/I`. |
| `vfrac_sigma` | `0.0` | Scatter in `V/I`. |
| `dpa_sigma` | `0.0` | Scatter in intra-microshot PA sweep rate (deg/ms). |
| `band_centre_sigma` | `0.0` | Scatter in bandpass centre frequency (MHz). |
| `band_width_sigma` | `0.0` | Scatter in bandpass FWHM (MHz). |

**How micro-scatter interacts with sweeps (genfns.py:124-146):**
When `analysis.sweep.mode = "mean"`, the micro-variance for the swept parameter is automatically zeroed by `_disable_micro_variance_for_swept_base()`. This ensures the sweep isolines the deterministic change of the mean.

---

## `[emission.components.amplitude_distribution]`

Controls the sampling of microshot amplitudes about the component mean.

| Key | Type | Description |
|-----|------|-------------|
| `type` | string | One of `"normal"`, `"lognormal"`, `"powerlaw"`, `"uniform"`. |

### `type = "powerlaw"`

| Key | Description |
|-----|-------------|
| `alpha` | Power-law exponent: `p(A) ~ A^(-alpha)`. |
| `xmin_scale` | Lower truncation bound = `amplitude_Jy * xmin_scale`. |
| `xmax_scale` | Upper truncation bound = `amplitude_Jy * xmax_scale`. |

Sampling formula (`sample_powerlaw`, genfns.py:551-558):
```python
A = (xmin^(1-alpha) + u * (xmax^(1-alpha) - xmin^(1-alpha)))^(1/(1-alpha))
```

### `type = "lognormal"`

| Key | Description |
|-----|-------------|
| `sigma` | Log-space standard deviation. The log-space mean is set to `mu = ln(amplitude_Jy) - sigma^2/2` so that the linear expectation is approximately `amplitude_Jy`. |

### `type = "uniform"`

| Key | Description |
|-----|-------------|
| `low_scale` | Lower bound = `|amplitude_Jy| * low_scale`. |
| `high_scale` | Upper bound = `|amplitude_Jy| * high_scale`. |

### `type = "normal"`

No sub-table. Simply draws from `N(amplitude_Jy, amplitude_sigma)`.

---

## `[analysis]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `buffer_fraction` | float | `0.1` | Fractional guard gap between the on-pulse and off-pulse windows. The gap width is `buffer_fraction * intrinsic_envelope_FWHM`. This prevents pulse leakage from contaminating the noise estimate. |

**Where it flows:** Used everywhere on/off-pulse masks are computed: `on_off_pulse_masks_from_profile()`, `compute_segments()`, `correct_baseline()`, `add_noise()`.

---

## `[analysis.sweep]`

Parameter sweep system for producing analytic plots (PA variance vs. swept parameter, L/I vs. swept parameter, etc.).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable` | bool | `true` | Enable the sweep. Required for multi-FRB analytic plot modes. |
| `mode` | string | `"sd"` | Sweep mode: `"mean"` (vary the component mean of the named parameter), `"sd"` (vary the microshot standard deviation), `"none"` (static). |

### `[analysis.sweep.parameter]`

| Key | Type | Description |
|-----|------|-------------|
| `component` | int | 0-based index of the `[[emission.components]]` entry to sweep. |
| `name` | string | Canonical parameter key. When `mode = "mean"`, use the base name (e.g., `"pa_deg"`, `"tau"`). When `mode = "sd"`, use `"sd_"` + base name (e.g., `"sd_pa_deg"` → becomes `"pa_sigma_deg"` in TOML). The alias system (`canonical_emission_key`) resolves many naming variants. |
| `start` | float | Sweep start value. |
| `stop` | float | Sweep end value (inclusive for linear steps). |
| `step` | float | Linear step size. Ignored if `log_steps` is set. |
| `log_steps` | int / null | If set to a positive integer `N`, override linear stepping with `N` log-spaced points via `np.logspace(log10(start), log10(stop), N)`. |

**Execution flow for sweeps (genfrb.py:452-463):**

1. `_setup_sweep()` generates `xvals` from the start/stop/step (or log_steps) range.
2. Slurm chunking optionally distributes `xvals` across array jobs.
3. Tasks = `product(xvals, range(nseed))` are submitted to a `ProcessPoolExecutor`.
4. Each task calls `_process_task()` → dispatches to `psn_dspec()` or `fold_dspec()`.
5. Results are collected into `measures[xval]`, `snrs[xval]`, `V_params[xval]`, `exp_vars[xval]`.

---

## `[observation]`

Noise model and observation parameters.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sefd` | float | `2` | System Equivalent Flux Density (Jy). Gaussian white noise is injected with standard deviation `sigma = SEFD / sqrt(df * dt)` per polarisation. Set to `0` for a noiseless simulation. |
| `target_snr` | float | `0` | Target integrated signal-to-noise ratio. If > 0, the dynamic spectrum is scaled to achieve this S/N before noise is added. `0` or negative disables target scaling. |
| `baseline_correct` | string / bool | `false` | Per-channel baseline removal method applied after noise injection. Options: `false` (none), `"median"` (subtract per-channel median from off-pulse window), `"mean"` (subtract mean), `"z"` (robust z-score), `"z_i"` (iterative z-score). |

**Where it flows:**
- `sefd` → `add_noise()` (noise.py) which draws `N(0, sigma)` noise for each polarisation channel.
- `target_snr` → `scale_dspec_to_target_snr()` (noise.py) which scales the entire dspec to achieve the desired total S/N.
- `baseline_correct` → `correct_baseline()` (noise.py).

---

## `[numerics]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_cpus` | int | `8` | Number of worker processes for parallel sweep execution. Falls back to the `SLURM_CPUS_PER_TASK` environment variable if set (for HPC array jobs). |
| `nseed` | int | `1` | Number of independent random realisations per sweep point. More realisations improve the statistical precision of estimated quantities (PA variance, L/I, V/I, etc.) at the cost of linear increase in runtime. |

**Seed progression for sweep realisations (genfrb.py:182-183):**
```python
current_seed = (base_seed + realisation) if base_seed is not None else None
```
Each realisation within a sweep point gets an offset from the base seed.

For the `"fold"` model, each internal fold also gets a unique seed:
```python
fold_seed = (current_seed * nfold + i) if current_seed is not None else None
```

---

## `[output]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `directory` | string | `"output"` | Root output directory. Simulation products (FRB pickle files for single runs, sweep dictionaries for analytic runs, Stokes `.npy` cubes, plots) are written here. Can be overridden with `--output-dir` on the CLI. |

---

## Parameter Override System (CLI)

In addition to the TOML config, parameters can be overridden at runtime via `--override-param`:

```bash
fires --config-dir . --override-param tau=0.8 N=5 propagation.scintillation.enable=false
```

- Emission parameters (numeric): matched against canonical keys and written into `gdict` or `sd_dict`.
- Config parameters (dotted paths): applied to the raw TOML dict before schema parsing.

---

## Example: Minimal Valid Config

```toml
[meta]
name = "test"
seed = 1

[simulation.grid]
f_start_MHz = 700
f_end_MHz   = 800
df_MHz      = 1.0
t_start_ms  = -5.0
t_end_ms    = 5.0
dt_ms       = 0.05
reference_freq_MHz = 750

[propagation.scattering]
index = -4.0
screen = "thin"

[propagation.rm]
RM = 0.0
order = "pre"

[propagation.scintillation]
enable = false

[emission]
model = "psn"

[[emission.components]]
t0_ms           = 0.0
width_ms        = 1.0
amplitude_Jy    = 1.0
spectral_index  = 0.0
tau_ms          = 0.0
dm              = 0.0
rm              = 0.0
pa_deg          = 0.0
lfrac           = 0.5
vfrac           = 0.0
dpa_deg_per_ms  = 0.0
band_centre_MHz = 0.0
band_width_MHz  = 0.0

[emission.components.microshots]
N               = 10
width_frac_low  = 0.05
width_frac_high = 0.20

[emission.components.microshot_scatter]
pa_sigma_deg = 10

[emission.components.amplitude_distribution]
type = "normal"

[analysis]
buffer_fraction = 0.1

[analysis.sweep]
enable = false
mode = "none"

[analysis.sweep.parameter]
component = 0
name = "none"
start = 0
stop = 0
step = 0

[observation]
sefd = 0

[numerics]
n_cpus = 8
nseed = 1

[output]
directory = "output"
```
