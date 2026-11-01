# Multi-Neuron Spike-Train Analysis

A Python research project for exploring population activity and statistical
relationships between neurons in multi-unit spike recordings. The analysis
scripts cover burst detection, spike-triggered activity, jitter-based null
models, connectivity scoring, spatial analysis, and stochastic network models.

The repository is organized as a collection of reproducible analysis scripts
rather than a single installable application. Results depend on the input
recording, analysis configuration, and script parameters.

## Analysis overview

- **Population activity and bursts** — bin spike trains, visualize activity,
  detect bursts, and compare activity inside and outside bursts.
- **Spike-triggered analysis** — compute cross-neuron histograms and
  per-configuration \(P_{ij}\) values.
- **Significance analysis** — generate jitter-based null distributions and
  evaluate pairwise responses.
- **Connectivity and spatial structure** — calculate and visualize \(S_{ij}\)
  scores, compare connection strength with unit distance, and produce network
  and clustering figures.
- **Stochastic models** — build and inspect model activity using the \(S_{ij}\)
  and \(P_{ij}\) analysis pipelines.

These are statistical analyses of recorded activity; a significant score by
itself does not establish a biological causal mechanism.

## Repository layout

| Path | Contents |
| --- | --- |
| `Experiment1/` | Earlier spike-triggered, null-model, burst, and significance analyses. |
| `Experiment2/` | Expanded analysis, including configuration-level activity and stochastic models. |
| `Experiment2/Configurations/` | Recording configuration and timing tables used by Experiment 2. |
| `Experiment2/Significance_Analysis/` | Aggregate connectivity, distance, and spatial analyses. |
| `Experiment2/Within_Burst/` | Analyses of activity within detected bursts. |
| `Experiment2/Between_Burst/` | Analyses of activity between bursts. |
| `Experiment2/Spike_Triggered_Average/` | Aggregate spike-triggered analysis and \(P_{ij}\) scripts. |
| `Experiment2/Stochastic_Model_Pij/` | Stochastic models based on \(P_{ij}\). |
| `Experiment2/Stochastic_Model_Sij/` | Stochastic models based on \(S_{ij}\). |
| `data/` | Data-conversion utility and source data currently present in this checkout. |
| `old2/` | Archived earlier analysis work. |

Most scripts write plots and intermediate results next to the script or input
files. Check each script's input paths and parameters before running it.

## Requirements

Use Python 3 and install the libraries used by the analysis scripts:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy matplotlib matplotlib-venn networkx numba plotly
```

On macOS or Linux, create and activate the virtual environment with the
platform's usual `python3 -m venv .venv` and `source .venv/bin/activate`
commands.

## Data and running an analysis

Some scripts require large recording files or intermediate results that are
not included in a fresh clone. In particular, parts of Experiment 2 expect
`Experiment2/neuron_data.json`; the binned-data generator also expects
`Experiment2/Configurations/config_times.csv`. The `.gitignore` excludes
large generated data files, so provide the required inputs separately before
running those stages.

Many scripts use paths relative to their current working directory. For
example, with the Experiment 2 neuron data and `activity.npz` available:

```powershell
cd Experiment2\Raster_Plot
python compute_activity.py
python detect_bursts.py
```

`compute_activity.py` produces the activity file used by `detect_bursts.py`,
which writes burst intervals to `bursts.json`. For other analyses, change to
the relevant script directory and check its input files and top-level
parameters before execution. Some scripts may require substantial memory or
disk space when processing full recordings.

## Reproducibility

Analysis thresholds, time windows, bin sizes, and model settings are defined
in individual scripts. Record any parameter changes alongside reported
results, and retain the input-data version and generated outputs needed to
reproduce a figure or statistic. The repository does not currently provide a
single end-to-end runner or automated test suite.
