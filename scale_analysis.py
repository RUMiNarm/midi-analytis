import json
import os
from collections import Counter

import matplotlib.pyplot as plt
from mido import MidiFile


def analyze_midi_scales(midi_path, track_index):
    mid = MidiFile(midi_path)
    notes = []

    if track_index >= len(mid.tracks):
        raise ValueError("Invalid track index")

    track = mid.tracks[track_index]
    for msg in track:
        if msg.type == "note_on" and msg.velocity > 0:
            notes.append(
                msg.note % 12
            )  # Normalize to pitch class (C=0, C#=1, ... B=11)

    note_counts = Counter(notes)

    return note_counts


def plot_note_counts(note_counts, title, output_path):
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    counts = [note_counts.get(i, 0) for i in range(12)]
    total_notes = sum(counts)
    frequencies = [count / total_notes for count in counts]

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # 出現回数のグラフ
    axs[0].bar(notes, counts, color="skyblue")
    axs[0].set_title("Note Counts")
    axs[0].set_xlabel("Notes")
    axs[0].set_ylabel("Counts")

    # 頻度のグラフ
    axs[1].bar(notes, frequencies, color="lightgreen")
    axs[1].set_title("Note Frequencies")
    axs[1].set_xlabel("Notes")
    axs[1].set_ylabel("Frequency")

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close()


def save_analysis_to_json(note_counts, title, output_path):
    analysis_data = {
        "title": title,
        "note_counts": {str(note): count for note, count in note_counts.items()},
    }
    with open(output_path, "w") as json_file:
        json.dump(analysis_data, json_file, indent=4)


def save_all_analyses_to_json(all_analyses, output_path):
    with open(output_path, "w") as json_file:
        json.dump(all_analyses, json_file, indent=4)


def analyze_folder(folder_path, output_folder, track_index):
    os.makedirs(output_folder, exist_ok=True)
    all_analyses = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".mid") or filename.endswith(".midi"):
            midi_path = os.path.join(folder_path, filename)
            try:
                scale_counts = analyze_midi_scales(midi_path, track_index)
                print(f"Used notes in {filename}:")
                for note, count in sorted(scale_counts.items()):
                    print(f"Note {note}: {count} times")
                output_image_path = os.path.join(
                    output_folder, f"{os.path.splitext(filename)[0]}_analysis.png"
                )
                plot_note_counts(scale_counts, filename, output_image_path)
                all_analyses[filename] = {
                    "note_counts": {
                        str(note): count for note, count in scale_counts.items()
                    }
                }
            except ValueError as e:
                print(f"Error processing {filename}: {e}")

    output_json_path = os.path.join(output_folder, "all_analyses.json")
    save_all_analyses_to_json(all_analyses, output_json_path)


if __name__ == "__main__":
    folder_path = "input/okinawa/"  # MIDIファイルが含まれるフォルダのパスを指定
    output_folder = "output/okinawa"  # 画像ファイルとJSONファイルの出力先フォルダを指定
    track_index = 0  # 解析したいトラック番号

    analyze_folder(folder_path, output_folder, track_index)
