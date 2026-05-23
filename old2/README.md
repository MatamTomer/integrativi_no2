# Neuron Spike Analysis Project

This project analyzes spike data from 29 neurons to identify causal relationships, inhibitory connections, and network activity patterns.

## Directory Structure

### `data/`
Contains the raw and processed data files:
- `tomer.txt` - Original spike data
- `neuron_data.json` - Processed spike times for all 29 neurons (429,471 total spikes)
- `load_neuron_data_to_json.py` - Script to convert raw data to JSON format

### `1_spike_triggered_analysis/`
Spike-triggered histogram analysis (±10ms window):
- `spike_triggered_average.py` - Generates spike-triggered histograms for all neuron pairs
- `visualize_with_error_bars.py` - Creates visualizations with error bars showing null distribution
- `spike_triggered_histograms/` - Output histograms for all neuron pairs

### `2_significance_analysis/`
Statistical significance testing with bootstrap null hypothesis:
- `compute_significance_combined.py` - Main script: 1000 iterations with ±20-50ms jitter
- `significance_analysis/` - Results folder containing:
  - `z_scores.json` - Z-scores for all pairs and bins
  - `p_values.json` - Two-sided p-values
  - `null_hypothesis_data.json` - Mean and std from bootstrap
  - Analysis PDFs

#### `split_half_reliability/`
Split-half reliability analysis to validate findings:
- `run_split_half_analysis_automated.py` - Automated pipeline splitting data 50/50
- `significance_analysis_split/` - Results for both halves
  - Split-half correlation: **r = 0.993** (excellent reliability)

### `3_causality_analysis/`
Excitatory causality analysis using S_ij metric:
- `compute_sij_matrix.py` - Computes S_ij = max(Z at 1-5ms) - median(Z at 6-10ms)
- `create_network_graph.py` - Creates directed network graph with threshold S_ij > 50
- Output files:
  - `sij_matrix.json` - 29×29 S_ij matrix
  - `sij_matrix_heatmap.png` - Heatmap visualization
  - `network_graph.png` - Network graph showing excitatory connections
  - Key finding: 51 strong excitatory connections, strongest: Neuron 28→27 (S_ij = 664.1)

### `4_inhibition_analysis/`
Inhibitory activity analysis using D_ij metric:
- `compute_dij_matrix.py` - Computes D_ij = min(Z at 1-5ms) + median(Z at 6-10ms)
- `visualize_dij_distribution.py` - Shows distribution of D_ij values
- `create_inhibitory_network_graph.py` - Creates network graph with threshold D_ij < -10
- Output files:
  - `dij_matrix.json` - 29×29 D_ij matrix
  - `dij_matrix_heatmap.png` - Heatmap visualization
  - `dij_distribution.png` - Distribution analysis (4 panels)
  - `inhibitory_network_graph.png` - Network graph showing inhibitory connections
  - `inhibitory_network_stats.json` - Network statistics
  - Key findings:
    - 61% of connections show inhibitory patterns
    - 30 strong inhibitory connections (D_ij < -10)
    - Strongest: Neuron 3→9 (D_ij = -41.19)

### `5_network_activity_analysis/`
Network activity state analysis (±10ms windows):
- `network_activity_analysis.py` - Calculates network activity level at each spike
- `create_combined_network_activity_pdf.py` - Creates combined PDF with all neurons
- Output files:
  - `network_activity_by_spike.json` - Activity levels for each neuron's spikes
  - `network_activity_combined.pdf` - 58 pages (29 neurons × 2 pages: counts & percentages)
  - `network_activity_histograms/` - Individual PNG histograms
  - Key finding: Mean network activity = 22.48 neurons active (out of 28)

### `old/`
Previous analysis attempts from earlier sessions (archived)

### `unused_scripts/`
Scripts that were developed but superseded by better versions:
- `compute_pvalues.py`
- `compute_significance.py`
- `compute_significance_fast.py`
- `compute_significance_split_halves.py`
- `run_split_half_analysis.py`

## Key Findings Summary

1. **Reliability**: Split-half correlation r = 0.993 confirms highly reproducible results
2. **Excitatory Network**: 51 strong connections (S_ij > 50), with Neuron 28 as main driver
3. **Inhibitory Network**: 30 strong inhibitory connections (D_ij < -10), 61% total inhibitory bias
4. **Network Activity**: Neurons predominantly fire during high synchrony (mean = 22.48/28 active)
5. **Hub Neurons**:
   - Excitatory: Neuron 28 (strongest driver), Neuron 27 (most influenced)
   - Inhibitory: Neuron 14 (strongest inhibitor), Neuron 9 (most inhibited)

## Analysis Parameters

- **Time window**: ±10ms for causal analysis
- **Bootstrap iterations**: 1000 with ±20-50ms jitter
- **Minimum spike threshold**: 300 correlated spikes per pair
- **Bin size**: 1ms
- **Total neurons**: 29
- **Total spikes**: 429,471
- **Valid pairs analyzed**: 408 (out of 841 possible)
