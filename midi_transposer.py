import argparse
import os
from collections import defaultdict

from mido import Message, MidiFile, MidiTrack


def transpose_midi(input_file, output_folder, semitone_shift):
    """
    MIDIファイルを指定された半音シフトで移調し、各チャンネルごとに専用のトラックを作成して保存する
    """
    try:
        midi = MidiFile(input_file)

        # 出力ファイル名を作成 (元のファイル名 + _transposed±X)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        new_file_name = f"{base_name}_transposed{semitone_shift:+}.mid"
        output_path = os.path.join(output_folder, new_file_name)

        # 新しいMIDIファイル作成
        new_midi = MidiFile()
        new_midi.ticks_per_beat = midi.ticks_per_beat

        # 各チャンネルごとに専用のトラックを作成
        channel_tracks = defaultdict(MidiTrack)
        new_midi.tracks.append(MidiTrack())  # 最初のトラック（MIDIヘッダー情報用）

        for track in midi.tracks:
            for msg in track:
                if msg.type == "track_name":  # トラック名を保持
                    new_midi.tracks[0].append(msg)

                elif msg.type == "note_on" or msg.type == "note_off":
                    # 各チャンネルごとに専用のトラックを確保
                    if msg.channel not in channel_tracks:
                        channel_tracks[msg.channel] = MidiTrack()
                        new_midi.tracks.append(channel_tracks[msg.channel])

                    new_note = max(
                        0, min(127, msg.note + semitone_shift)
                    )  # ノート範囲制限
                    new_msg = Message(
                        msg.type,
                        note=new_note,
                        velocity=msg.velocity,
                        time=msg.time,
                        channel=msg.channel,
                    )
                    channel_tracks[msg.channel].append(new_msg)

                elif msg.type == "program_change":
                    if msg.channel not in channel_tracks:
                        channel_tracks[msg.channel] = MidiTrack()
                        new_midi.tracks.append(channel_tracks[msg.channel])

                    channel_tracks[msg.channel].append(msg)

                else:
                    # その他のメッセージ（テンポ、コントロールチェンジなど）を最初のトラックに追加
                    new_midi.tracks[0].append(msg)

        # 出力先フォルダが存在しない場合は作成
        os.makedirs(output_folder, exist_ok=True)

        # 変換後のMIDIを保存
        new_midi.save(output_path)
        print(f"✅ {input_file} → {output_path}")

    except Exception as e:
        print(f"❌ Error processing {input_file}: {e}")


def process_midi_files(input_path: str, output_folder: str, semitone_shift: int):
    """
    入力がファイルかフォルダかを判定
    """
    if os.path.isfile(input_path):
        transpose_midi(input_path, output_folder, semitone_shift)
    elif os.path.isdir(input_path):
        # フォルダ内のすべてのMIDIファイルを処理
        midi_files = [f for f in os.listdir(input_path) if f.endswith(".mid")]
        if not midi_files:
            print(f"❌ No MIDI files found in {input_path}")
            return
        for midi_file in midi_files:
            transpose_midi(
                os.path.join(input_path, midi_file), output_folder, semitone_shift
            )
    else:
        print(f"❌ Invalid path: {input_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transpose MIDI files by a given number of semitones."
    )
    parser.add_argument("input_path", help="Path to the input MIDI file or folder")
    parser.add_argument(
        "semitone_shift",
        type=int,
        help="Number of semitones to shift (e.g., 2 for C→D)",
    )
    parser.add_argument(
        "-d",
        "--destination",
        help="Output folder (default: same as input)",
        default=None,
    )

    args = parser.parse_args()

    # 出力フォルダが指定されていない場合は入力と同じフォルダを使用
    output_folder = (
        args.destination
        if args.destination
        else os.path.dirname(os.path.abspath(args.input_path))
    )

    process_midi_files(args.input_path, output_folder, args.semitone_shift)
