#!/usr/bin/python
import time
from pathlib import Path
from typing import List, Optional
import math
import random
import maxlab as mx

"""
PHASE 1 (INTERLEAVED, BALANCED RANDOM TARGETING) + POISSON GLOBAL TIMING

- Balanced interleaving across neurons (random target each pulse, equal counts)
- Poisson timing for global pulse times: dt ~ Exp(rate), with optional min dt
- Amplitude schedule: 50..700 mV step 25, repeated 27 times per neuron
- Save raw data + event markers for offline analysis (evoked yes/no)
"""

# =========================
# FILL THESE
# =========================
# fmt: off
electrodes: List[int] = [

    # TODO: put recording electrodes here
]
stim_electrodes: List[int] = [
    # TODO: put stimulation electrodes here (one per neuron candidate)
]
# fmt: on

# =========================
# EXPERIMENT SETTINGS
# =========================
BASELINE_SECONDS = 120
SAVE_DIR = "/tmp"
SAVE_FILE = "phase1_interleaved_poisson"

# amplitude schedule: 50..700 by 25 mV
AMP_START_MV = 50
AMP_END_MV = 700
AMP_STEP_MV = 25
REPEATS_PER_NEURON = 27

# Pulse shape: 200 us per phase (at 20 kHz => 4 samples of 50 us)
PHASE_SAMPLES = 4

# Poisson global timing:
# mean dt = 1/POISSON_RATE_HZ seconds
POISSON_RATE_HZ = 10.0        # e.g. 10 Hz global average
MIN_DT_SECONDS = 0.020        # enforce >=20ms between pulses (like the paper)

event_counter = 1  # global event id counter


# ============================================================
# YOUR FUNCTIONS (kept; tiny robustness fix in connect_* )
# ============================================================

def seconds_to_samples(dt_s: float, sample_period_s: float) -> int:
    return max(1, int(round(dt_s / sample_period_s)))

def build_poisson_sweep_sequence(
    rate_hz: float,
    phase_samples: int,
    sample_period_s: float = 50e-6,   # 50 µs per sample
    amplitude_min_mV: int = 10,
    amplitude_max_mV: int = 200,
    amplitude_step_mV: int = 10,
    repeats: int = 10,
) -> mx.Sequence:
    """
    (Not used in this Phase 1 interleaved protocol, but kept.)
    Creates pulses with Exp(rate_hz) inter-pulse intervals.
    """
    seq = mx.Sequence()
    dac_lsb_mV = float(mx.query_DAC_lsb_mV())

    amps_mV = list(range(amplitude_min_mV, amplitude_max_mV + 1, amplitude_step_mV))
    amp_schedule_mV = amps_mV * repeats

    global event_counter
    event_counter += 1
    seq.append(mx.Event(0, 1, event_counter,
                        f"POISSON_SWEEP_START rate={rate_hz}Hz pulses={len(amp_schedule_mV)}"))

    for i, amp_mV in enumerate(amp_schedule_mV, start=1):
        u = random.random()
        dt = -math.log(1.0 - u) / rate_hz
        seq.append(mx.DelaySamples(seconds_to_samples(dt, sample_period_s)))

        amp_bits = int(amp_mV / dac_lsb_mV)

        event_counter += 1
        seq.append(mx.Event(0, 1, event_counter, f"PULSE {i}/{len(amp_schedule_mV)} amp={amp_mV}mV"))

        create_stim_pulse(seq, amp_bits, phase_samples)

    event_counter += 1
    seq.append(mx.Event(0, 1, event_counter, "POISSON_SWEEP_END"))
    return seq

def initialize_system() -> None:
    mx.initialize()
    if mx.send(mx.Core().enable_stimulation_power(True)) != "Ok":
        raise RuntimeError("The system didn't initialize correctly.")

def configure_array(electrodes: List[int], stim_electrodes: List[int]) -> mx.Array:
    array = mx.Array("stimulation")
    array.reset()
    array.clear_selected_electrodes()
    array.select_electrodes(electrodes)
    array.select_stimulation_electrodes(stim_electrodes)
    array.route()
    return array

def load_config(config_file: str) -> mx.Array:
    path = Path(config_file)
    if not path.is_file():
        raise FileNotFoundError(f"Config file '{config_file}' not found.")
    array = mx.Array("stimulation")
    try:
        array.load_config(config_file)
    except Exception as e:
        raise Exception(f"Error loading config file '{config_file}': {str(e)}")
    return array

def connect_stim_units_to_stim_electrodes(
    stim_electrodes: List[int], array: mx.Array
) -> List[int]:
    stim_units: List[int] = []
    for stim_el in stim_electrodes:
        array.connect_electrode_to_stimulation(stim_el)
        stim = array.query_stimulation_at_electrode(stim_el)
        if len(stim) == 0:
            raise RuntimeError(f"No stimulation channel can connect to electrode: {str(stim_el)}")

        if isinstance(stim, (list, tuple)):
            stim_unit_int = int(stim[0])
        else:
            stim_unit_int = int(stim)

        if stim_unit_int in stim_units:
            raise RuntimeError(
                f"Two electrodes connected to the same stim unit. "
                f"This is not allowed. Please Select a neighboring electrode of {stim_el}!"
            )
        stim_units.append(stim_unit_int)
    return stim_units

def powerup_stim_unit(stim_unit: int) -> mx.StimulationUnit:
    return (
        mx.StimulationUnit(stim_unit)
        .power_up(True)
        .connect(True)
        .set_voltage_mode()
        .dac_source(0)
    )

def configure_and_powerup_stim_units(stim_units: List[int]) -> List[mx.StimulationUnit]:
    stim_unit_commands: List[mx.StimulationUnit] = []
    for stim_unit in stim_units:
        stim = powerup_stim_unit(stim_unit)
        stim_unit_commands.append(stim)
        mx.send(stim)
    return stim_unit_commands

def poweroff_all_stim_units() -> None:
    for stimulation_unit in range(0, 32):
        stim = mx.StimulationUnit(stimulation_unit)
        stim.power_up(False)
        stim.connect(False)
        mx.send(stim)

def create_stim_pulse(seq: mx.Sequence, amplitude: int, delay_samples: int) -> mx.Sequence:
    seq.append(mx.DAC(0, 512 - amplitude))
    seq.append(mx.DelaySamples(delay_samples))
    seq.append(mx.DAC(0, 512 + amplitude))
    seq.append(mx.DelaySamples(delay_samples))
    seq.append(mx.DAC(0, 512))
    return seq

def prepare_stim_sequence(
    number_pulses_per_train: int,
    inter_pulse_interval: int,
    phase: int,
    amplitude: int,
    changing_amplitude: Optional[bool] = False,
    max_amplitude: Optional[int] = None,
    amplitude_interval: Optional[int] = None,
) -> mx.Sequence:
    seq = mx.Sequence()
    dac_lsb_mV = float(mx.query_DAC_lsb_mV())
    if changing_amplitude:
        if max_amplitude is None or amplitude_interval is None:
            raise ValueError("Both max_amplitude and amplitude_interval are required for changing_amplitude.")
        for cur_amplitude in range(amplitude, max_amplitude, amplitude_interval):
            for _ in range(number_pulses_per_train):
                seq = create_stim_pulse(seq, int(cur_amplitude/dac_lsb_mV), phase)
                seq.append(mx.DelaySamples(inter_pulse_interval))
            seq.append(mx.DelaySamples(inter_pulse_interval))
    else:
        for _ in range(number_pulses_per_train):
            seq = create_stim_pulse(seq, int(amplitude/dac_lsb_mV), phase)
            seq.append(mx.DelaySamples(inter_pulse_interval))
    return seq

def send_stim_pulses_all_units(seq: mx.Sequence, number_pulse_trains: int) -> None:
    for _ in range(number_pulse_trains):
        print("Send pulse")
        seq.send()
        time.sleep(10)

def send_stim_pulses_units_sequentially(seq: mx.Sequence, stim_units: List[int]) -> None:
    for stim_unit in stim_units:
        print(f"Power up stimulation unit {stim_unit}")
        stim = mx.StimulationUnit(stim_unit)
        stim.power_up(True).connect(True).set_voltage_mode().dac_source(0)
        mx.send(stim)
        print("Send pulse")
        seq.send()
        print(f"Power down stimulation unit {stim_unit}")
        stim = mx.StimulationUnit(stim_unit).power_up(False)
        mx.send(stim)
        time.sleep(2)

# ============================================================
# POISSON dt helper
# ============================================================

def poisson_dt(rate_hz: float, min_dt: float) -> float:
    """Draw dt ~ Exp(rate_hz) with minimum min_dt."""
    dt = random.expovariate(rate_hz)  # mean = 1/rate_hz
    return max(min_dt, dt)

# ============================================================
# PHASE 1 MAIN
# ============================================================

if __name__ == "__main__":

    if not electrodes:
        raise RuntimeError("electrodes list is empty. Fill it.")
    if not stim_electrodes:
        raise RuntimeError("stim_electrodes list is empty. Fill it.")

    # ---------- Initialize ----------
    initialize_system()
    time.sleep(mx.Timing.waitInit)

    wells = [0]
    mx.activate(wells)

    poweroff_all_stim_units()

    # ---------- Routing / Configuration ----------
    array = configure_array(electrodes, stim_electrodes)
    stim_units = connect_stim_units_to_stim_electrodes(stim_electrodes, array)

    array.download()
    time.sleep(mx.Timing.waitAfterDownload)

    mx.offset()
    time.sleep(15)
    mx.clear_events()

    # ---------- Start Recording ----------
    s = mx.Saving()
    s.open_directory(SAVE_DIR)
    s.start_file(SAVE_FILE)

    s.group_define(0, "all_channels", list(range(1024)))
    s.start_recording([0])

    print("Recording started")

    # Baseline marker
    global event_counter
    seq0 = mx.Sequence()
    event_counter += 1
    seq0.append(mx.Event(0, 1, event_counter, "BASELINE_START"))
    seq0.send()

    time.sleep(BASELINE_SECONDS)

    # ---------- Build amplitude schedule ----------
    amps_mV = list(range(AMP_START_MV, AMP_END_MV + 1, AMP_STEP_MV))  # 50..700 step 25
    pulses_per_neuron = len(amps_mV) * REPEATS_PER_NEURON             # 27*27 = 729

    num_neurons = len(stim_units)

    # Balanced target list
    targets: List[int] = []
    for ni in range(num_neurons):
        targets.extend([ni] * pulses_per_neuron)
    random.shuffle(targets)

    # Per-neuron counter: keeps each neuron cycling through amps in order
    per_neuron_count = [0] * num_neurons

    # Query DAC scale once
    dac_lsb_mV = float(mx.query_DAC_lsb_mV())

    print(f"Neurons: {num_neurons}")
    print(f"Amps: {len(amps_mV)} values ({amps_mV[0]}..{amps_mV[-1]} mV step {AMP_STEP_MV})")
    print(f"Pulses per neuron: {pulses_per_neuron}")
    print(f"Total pulses: {len(targets)}")
    print(f"Poisson rate: {POISSON_RATE_HZ} Hz (mean dt={1.0/POISSON_RATE_HZ:.3f}s), min dt={MIN_DT_SECONDS}s")

    # ---------- Interleaved stimulation ----------
    for idx in range(len(targets)):
        ni = targets[idx]
        stim_unit = stim_units[ni]

        # amplitude for THIS neuron (its own sweep counter)
        k = per_neuron_count[ni]
        amp_mV = amps_mV[k % len(amps_mV)]
        per_neuron_count[ni] += 1

        # power up this unit
        mx.send(
            mx.StimulationUnit(stim_unit)
              .power_up(True)
              .connect(True)
              .set_voltage_mode()
              .dac_source(0)
        )

        # one-pulse sequence + event
        seq = mx.Sequence()
        event_counter += 1
        seq.append(mx.Event(0, 1, event_counter, f"STIM neuron={ni} unit={stim_unit} amp_mV={amp_mV}"))

        amp_bits = int(round(amp_mV / dac_lsb_mV))
        create_stim_pulse(seq, amp_bits, PHASE_SAMPLES)
        seq.send()

        # power down
        mx.send(
            mx.StimulationUnit(stim_unit)
              .power_up(False)
              .connect(False)
        )

        # Poisson pacing (global)
        dt = poisson_dt(POISSON_RATE_HZ, MIN_DT_SECONDS)
        time.sleep(dt)

        if (idx + 1) % 200 == 0:
            print(f"Progress: {idx+1}/{len(targets)} pulses")

    # ---------- Stop Recording ----------
    print("Stopping recording")
    s.stop_recording()
    time.sleep(mx.Timing.waitAfterRecording)
    s.stop_file()
    s.group_delete_all()

    print("Done")