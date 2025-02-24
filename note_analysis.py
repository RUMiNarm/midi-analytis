import os
from collections import Counter

import matplotlib.pyplot as plt
from mido import MidiFile


def analyze_midi_notes(midi_path, track_index=1):
    mid = MidiFile(midi_path)
    if track_index >= len(mid.tracks):
        raise ValueError("Invalid track index")

    track = mid.tracks[track_index]
    notes = []
    time = 0
    for msg in track:
        time += msg.time
        if msg.type == "note_on" or msg.type == "note_off":
            notes.append(
                {
                    "note": msg.note,
                    "time": time,
                    "type": msg.type,
                    "velocity": msg.velocity if msg.type == "note_on" else 0,
                }
            )

    return notes


def print_note_analysis(notes):
    for note in notes:
        note_type = "ON" if note["type"] == "note_on" else "OFF"
        print(
            f"Note {note['note']} {note_type} at time {note['time']} with velocity {note['velocity']}"
        )


def plot_note_analysis(notes, output_path):
    note_on_times = [note["time"] for note in notes if note["type"] == "note_on"]
    note_off_times = [note["time"] for note in notes if note["type"] == "note_off"]
    note_on_velocities = [
        note["velocity"] for note in notes if note["type"] == "note_on"
    ]
    note_off_velocities = [
        note["velocity"] for note in notes if note["type"] == "note_off"
    ]

    plt.figure(figsize=(10, 8))

    # Note ON times and velocities
    plt.scatter(note_on_times, note_on_velocities, color="blue", label="Note ON")
    # Note OFF times and velocities
    plt.scatter(note_off_times, note_off_velocities, color="red", label="Note OFF")

    plt.title("Note Analysis")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def analyze_note_usage(notes):
    note_counts = Counter(note["note"] for note in notes if note["type"] == "note_on")
    return note_counts


def plot_note_usage(note_counts, output_path):
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    counts = [note_counts.get(i, 0) for i in range(12)]

    plt.figure(figsize=(10, 8))
    plt.bar(notes, counts, color="skyblue")
    plt.title("Note Usage")
    plt.xlabel("Notes")
    plt.ylabel("Counts")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def analyze_note_lengths(notes):
    note_lengths = []
    note_on_times = {}

    for note in notes:
        if note["type"] == "note_on":
            note_on_times[note["note"]] = note["time"]
        elif note["type"] == "note_off" and note["note"] in note_on_times:
            length = note["time"] - note_on_times[note["note"]]
            note_lengths.append(length)
            del note_on_times[note["note"]]

    return note_lengths


def plot_note_lengths(note_lengths, output_path):
    plt.figure(figsize=(10, 8))
    plt.hist(note_lengths, bins=50, color="skyblue")
    plt.title("Note Lengths")
    plt.xlabel("Length")
    plt.ylabel("Counts")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    midi_path = "input/d-aki.mid"  # 解析したいMIDIファイルのパスを指定
    track_index = 1  # 解析したいトラック番号

    notes = analyze_midi_notes(midi_path, track_index)
    print_note_analysis(notes)
    output_path = "output/note_analysis.png"  # グラフの出力先を指定
    plot_note_analysis(notes, output_path)

    note_counts = analyze_note_usage(notes)
    output_path = "output/note_usage.png"  # グラフの出力先を指定
    plot_note_usage(note_counts, output_path)

    note_lengths = analyze_note_lengths(notes)
    output_path = "output/note_lengths.png"  # グラフの出力先を指定
    plot_note_lengths(note_lengths, output_path)
