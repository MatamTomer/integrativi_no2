#!/usr/bin/python

"""
STIMULATION AND RECORDING COMBINED

This script stimulates electrodes sequentially and records the neural response.
It combines functionality from API_mx.py (stimulation) and recordings.py.

Steps:
1. Initialize the system
2. Configure recording and stimulation electrodes
3. Start recording
4. Send stimulation pulses (sequentially to each electrode)
5. Stop recording and save file
"""

import time
import maxlab as mx
from typing import List

# Recording electrodes
# fmt: off
electrodes = [4885, 4666, 4886, 4022, 5327, 5328, 5106, 5326, 3138, 3140, 2919,
              5105, 4667, 4448, 5109, 4669, 4665, 3798, 4021, 3141, 4668, 4240,
              3363, 3803, 3580, 3801, 2921, 3799, 4239, 3359, 3142, 3797, 3361,
              4020, 4241, 4018, 4889, 4447, 3357, 5108, 4888, 5107, 4446, 3583,
              3360, 3802, 3358, 3578, 2920, 4019, 3582, 3362, 3577, 4887, 3139,
              3800, 3579, 3581,]
# fmt: on

# Stimulation electrodes
stim_electrodes = [3580, 4887]

# Global event counter for tracking stimulation events
event_counter = 1


# ============== STIMULATION FUNCTIONS ==============

def create_stim_pulse(seq: mx.Sequence, amplitude: int, delay_samples: int) -> mx.Sequence:
    """Create a biphasic stimulation pulse."""
    global event_counter
    event_counter += 1
    seq.append(mx.Event(0, 1, event_counter, f"amplitude {amplitude} event_id {event_counter}"))
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
    changing_amplitude: bool = False,
    max_amplitude: int = None,
    amplitude_interval: int = None,
) -> mx.Sequence:
    """Prepare a stimulation sequence (pulse train).

    If changing_amplitude=True, pulses increase from amplitude to max_amplitude
    in steps of amplitude_interval.
    """
    seq = mx.Sequence()
    dac_lsb_mV = float(mx.query_DAC_lsb_mV())

    if changing_amplitude:
        if max_amplitude is None or amplitude_interval is None:
            raise ValueError("max_amplitude and amplitude_interval required when changing_amplitude=True")
        for cur_amplitude in range(amplitude, max_amplitude, amplitude_interval):
            for _ in range(number_pulses_per_train):
                seq = create_stim_pulse(seq, int(cur_amplitude / dac_lsb_mV), phase)
                seq.append(mx.DelaySamples(inter_pulse_interval))
            seq.append(mx.DelaySamples(inter_pulse_interval))  # Extra gap between amplitude levels
    else:
        for _ in range(number_pulses_per_train):
            seq = create_stim_pulse(seq, int(amplitude / dac_lsb_mV), phase)
            seq.append(mx.DelaySamples(inter_pulse_interval))
    return seq


def connect_stim_units_to_electrodes(stim_electrodes: List[int], array: mx.Array) -> List[int]:
    """Connect stimulation units to the stimulation electrodes."""
    stim_units: List[int] = []
    for stim_el in stim_electrodes:
        array.connect_electrode_to_stimulation(stim_el)
        stim = array.query_stimulation_at_electrode(stim_el)
        if len(stim) == 0:
            raise RuntimeError(f"No stimulation channel can connect to electrode: {stim_el}")
        stim_unit_int = int(stim)
        if stim_unit_int in stim_units:
            raise RuntimeError(f"Two electrodes connected to same stim unit. Select different electrode for {stim_el}")
        stim_units.append(stim_unit_int)
    return stim_units


def send_stim_pulses_sequentially(seq: mx.Sequence, stim_units: List[int]) -> None:
    """Send stimulation pulses to each unit one at a time."""
    for stim_unit in stim_units:
        print(f"  Stimulating unit {stim_unit}...")

        # Power up this unit
        stim = mx.StimulationUnit(stim_unit)
        stim.power_up(True).connect(True).set_voltage_mode().dac_source(0)
        mx.send(stim)

        # Send pulse train
        seq.send()

        # Power down this unit
        stim = mx.StimulationUnit(stim_unit).power_up(False)
        mx.send(stim)

        time.sleep(2)  # Wait between electrodes


def poweroff_all_stim_units() -> None:
    """Power off all stimulation units."""
    for stimulation_unit in range(0, 32):
        stim = mx.StimulationUnit(stimulation_unit)
        stim.power_up(False)
        stim.connect(False)
        mx.send(stim)


# ============== MAIN SCRIPT ==============

if __name__ == "__main__":
    # === CONFIGURATION ===
    wells = [0]  # Which well(s) to use (MaxTwo)
    dir_name = "/tmp"
    file_name = "stim_recording"
    recording_duration = 30  # seconds (includes stimulation time)

    # Stimulation parameters
    number_pulses = 10
    inter_pulse_interval = 2000  # samples (100ms)
    phase = 4  # samples (200us)
    amplitude = 200  # mV (starting amplitude)
    changing_amplitude = True  # Increase amplitude over time
    max_amplitude = 400  # mV (ending amplitude)
    amplitude_interval = 50  # mV (step size: 200, 250, 300, 350)

    # === INITIALIZE ===
    print("Initializing system...")
    mx.initialize()
    if mx.send(mx.Core().enable_stimulation_power(True)) != "Ok":
        raise RuntimeError("System didn't initialize correctly.")
    time.sleep(mx.Timing.waitInit)

    # === CONFIGURE ARRAY ===
    print("Configuring array...")
    mx.activate(wells)

    array = mx.Array("stim_and_record")
    array.reset()
    array.clear_selected_electrodes()
    array.select_electrodes(electrodes)
    array.select_stimulation_electrodes(stim_electrodes)
    array.route()

    # Connect stimulation units
    stim_units = connect_stim_units_to_electrodes(stim_electrodes, array)
    print(f"  Stim electrodes {stim_electrodes} -> units {stim_units}")

    array.download(wells)
    time.sleep(mx.Timing.waitAfterDownload)

    # === OFFSET COMPENSATION ===
    print("Running offset compensation...")
    mx.offset()
    time.sleep(15)
    mx.clear_events()

    # === SETUP RECORDING ===
    print("Setting up recording...")
    s = mx.Saving()
    s.open_directory(dir_name)
    s.start_file(file_name)
    s.group_define(0, "all_channels", list(range(1024)))

    # === PREPARE STIMULATION ===
    print("Preparing stimulation sequence...")
    poweroff_all_stim_units()  # Ensure all off before sequential stim
    seq = prepare_stim_sequence(
        number_pulses_per_train=number_pulses,
        inter_pulse_interval=inter_pulse_interval,
        phase=phase,
        amplitude=amplitude,
        changing_amplitude=changing_amplitude,
        max_amplitude=max_amplitude,
        amplitude_interval=amplitude_interval,
    )

    # === START RECORDING ===
    print(f"\n{'='*50}")
    print("STARTING RECORDING")
    print(f"{'='*50}")
    s.start_recording(wells)

    # Wait a moment before stimulation (baseline recording)
    print("Recording baseline (2 seconds)...")
    time.sleep(2)

    # === STIMULATE ===
    print("\nSending stimulation pulses sequentially...")
    send_stim_pulses_sequentially(seq, stim_units)

    # Continue recording after stimulation
    remaining_time = recording_duration - 2 - (len(stim_units) * 4)  # Rough estimate
    if remaining_time > 0:
        print(f"\nRecording post-stimulation ({remaining_time} seconds)...")
        time.sleep(remaining_time)

    # === STOP RECORDING ===
    print(f"\n{'='*50}")
    print("STOPPING RECORDING")
    print(f"{'='*50}")
    s.stop_recording()
    time.sleep(mx.Timing.waitAfterRecording)
    s.stop_file()
    s.group_delete_all()

    print(f"\nRecording saved to: {dir_name}/{file_name}")
    print("Done!")
