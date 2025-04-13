import argparse
import os

from mido import MidiFile, MidiTrack


def extract_midi_track(input_path, track_index, output_path):
    mid = MidiFile(input_path)
    if track_index >= len(mid.tracks):
        raise ValueError("Invalid track index")

    new_mid = MidiFile()
    new_track = MidiTrack()
    new_mid.tracks.append(new_track)

    for msg in mid.tracks[track_index]:
        new_track.append(msg)

    new_mid.save(output_path)
    print(f"✅ Extracted track {track_index} from {input_path} to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract a specific track from a MIDI file."
    )
    parser.add_argument("input_path", help="Path to the input MIDI file")
    parser.add_argument("track_index", type=int, help="Index of the track to extract")
    parser.add_argument(
        "-o", "--output", help="Path to the output MIDI file", default=None
    )

    args = parser.parse_args()

    output_path = (
        args.output
        if args.output
        else os.path.join(
            os.path.dirname(args.input_path), f"extracted_track_{args.track_index}.mid"
        )
    )

    extract_midi_track(args.input_path, args.track_index, output_path)
