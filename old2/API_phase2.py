#!/usr/bin/python
import time
import random
import math
from typing import List, Optional
from pathlib import Path

import maxlab as mx


# =========================
# FILL THESE
# =========================
# Recording electrodes (can be large list)
electrodes: List[int] = [
    # TODO
]

# Exactly TWO stimulation electrodes for phase 2: [pre, post]
stim_electrodes: List[int] = [
    # TODO: [PRE_STIM_ELECTRODE, POST_STIM_ELECTRODE]
]


# =========================
# PHASE 2 PARAMETERS
# =========================
SAVE_DIR = "/tmp"
SAVE_FILE = "phase2_pre_post"

BASELINE_SECONDS = 120

# Pre pulse: strong fixed amplitude
PRE_STRONG_AMP_MV = 600

# Post pulse: sweep (your current idea)
POST_AMP_START_MV = 50
POST_AMP_END_MV = 700
POST_AMP_STEP_MV = 25
POST_REPEATS = 27

# Timing
DT_PRE_TO_POST_S = 0.010   # 10 ms
TRIAL_DT_S = 0.10          # spacing between trials (global pace)

# Pulse shape
PHASE_SAMPLES = 4          # 4*50us = 200us per phase

event_counter = 1


# =========================
# Helper functions (same style as examples)
# =========================
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


def poweroff_all_stim_units() -> None:
    for stimulation_unit in range(0, 32):
        stim = mx.StimulationUnit(stimulation_unit)
        stim.power_up(False)
        stim.connect(False)
        mx.send(stim)


def connect_stim_units_to_stim_electrodes(stim_electrodes: List[int], array: mx.Array) -> List[int]:
    stim_units: List[int] = []
    for stim_el in stim_electrodes:
        array.connect_electrode_to_stimulation(stim_el)
        stim = array.query_stimulation_at_electrode(stim_el)
        if len(stim) == 0:
            raise RuntimeError(f"No stimulation channel can connect to electrode: {str(stim_el)}")

        # robust handling
        if isinstance(stim, (list, tuple)):
            stim_unit_int = int(stim[0])
        else:
            stim_unit_int = int(stim)

        if stim_unit_int in stim_units:
            raise RuntimeError(
                f"Two electrodes connected to the same stim unit. "
                f"Select a neighboring electrode of {stim_el}!"
            )
        stim_units.append(stim_unit_int)
    return stim_units


def create_stim_pulse(seq: mx.Sequence, amplitude_bits: int, phase_samples: int) -> mx.Sequence:
    # Biphasic pulse on DAC0: + then - then back to baseline
    seq.append(mx.DAC(0, 512 - amplitude_bits))
    seq.append(mx.DelaySamples(phase_samples))
    seq.append(mx.DAC(0, 512 + amplitude_bits))
    seq.append(mx.DelaySamples(phase_samples))
    seq.append(mx.DAC(0, 512))
    return seq


def power_up_unit(unit: int) -> None:
    mx.send(mx.StimulationUnit(unit).power_up(True).connect(True).set_voltage_mode().dac_source(0))


def power_down_unit(unit: int) -> None:
    mx.send(mx.StimulationUnit(unit).power_up(False).connect(False))


if __name__ == "__main__":
    if len(stim_electrodes) != 2:
        raise RuntimeError("For Phase 2, set stim_electrodes = [PRE, POST] exactly two electrodes.")
    if not electrodes:
        raise RuntimeError("electrodes is empty.")

    initialize_system()
    time.sleep(mx.Timing.waitInit)

    wells = [0]
    mx.activate(wells)

    poweroff_all_stim_units()

    array = configure_array(electrodes, stim_electrodes)
    stim_units = connect_stim_units_to_stim_electrodes(stim_electrodes, array)
    pre_unit, post_unit = stim_units[0], stim_units[1]

    array.download()
    time.sleep(mx.Timing.waitAfterDownload)

    mx.offset()
    time.sleep(15)
    mx.clear_events()

    # Start recording
    s = mx.Saving()
    s.open_directory(SAVE_DIR)
    s.start_file(SAVE_FILE)
    s.group_define(0, "all_channels", list(range(1024)))
    s.start_recording(wells)

    print("Recording started")

    # Baseline marker
    global event_counter
    seq0 = mx.Sequence()
    event_counter += 1
    seq0.append(mx.Event(0, 1, event_counter, "BASELINE_START"))
    seq0.send()

    time.sleep(BASELINE_SECONDS)

    # Build post schedule
    post_amps = list(range(POST_AMP_START_MV, POST_AMP_END_MV + 1, POST_AMP_STEP_MV))
    total_trials = len(post_amps) * POST_REPEATS

    # Equal representation of amplitudes; random order across all trials
    post_amp_sequence: List[int] = []
    for _ in range(POST_REPEATS):
        post_amp_sequence.extend(post_amps)
    random.shuffle(post_amp_sequence)

    # 50/50 pre-on/off across trials
    pre_conditions = [1] * (total_trials // 2) + [0] * (total_trials - total_trials // 2)
    random.shuffle(pre_conditions)

    dac_lsb_mV = float(mx.query_DAC_lsb_mV())
    pre_bits = int(PRE_STRONG_AMP_MV / dac_lsb_mV)

    print(f"Trials: {total_trials}")
    print(f"PRE strong: {PRE_STRONG_AMP_MV} mV (unit {pre_unit})")
    print(f"POST sweep: {POST_AMP_START_MV}..{POST_AMP_END_MV} step {POST_AMP_STEP_MV} (unit {post_unit})")
    print(f"Δt pre→post: {DT_PRE_TO_POST_S*1000:.1f} ms")

    for t in range(total_trials):
        pre_on = pre_conditions[t]
        post_amp_mV = post_amp_sequence[t]
        post_bits = int(post_amp_mV / dac_lsb_mV)

        # trial header event
        event_counter += 1
        header = (
            f"TRIAL={t+1}/{total_trials} "
            f"PRE={'ON' if pre_on else 'OFF'} "
            f"PREamp_mV={PRE_STRONG_AMP_MV} "
            f"POSTamp_mV={post_amp_mV} "
            f"DTms={DT_PRE_TO_POST_S*1000:.1f} "
            f"preUnit={pre_unit} postUnit={post_unit}"
        )
        seqH = mx.Sequence()
        seqH.append(mx.Event(0, 1, event_counter, header))
        seqH.send()

        # PRE pulse (optional)
        if pre_on:
            power_up_unit(pre_unit)
            seqP = mx.Sequence()
            event_counter += 1
            seqP.append(mx.Event(0, 1, event_counter, "PRE_PULSE"))
            create_stim_pulse(seqP, pre_bits, PHASE_SAMPLES)
            seqP.send()
            power_down_unit(pre_unit)

        time.sleep(DT_PRE_TO_POST_S)

        # POST pulse (always)
        power_up_unit(post_unit)
        seqQ = mx.Sequence()
        event_counter += 1
        seqQ.append(mx.Event(0, 1, event_counter, "POST_PULSE"))
        create_stim_pulse(seqQ, post_bits, PHASE_SAMPLES)
        seqQ.send()
        power_down_unit(post_unit)

        time.sleep(TRIAL_DT_S)

        if (t + 1) % 50 == 0:
            print(f"Progress: {t+1}/{total_trials}")

    print("Stopping recording")
    s.stop_recording()
    time.sleep(mx.Timing.waitAfterRecording)
    s.stop_file()
    s.group_delete_all()
    print("Done")