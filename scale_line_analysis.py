import os
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from mido import MidiFile

# 日本語フォントを設定
matplotlib.rcParams["font.family"] = "MS Gothic"  # Windowsの場合


# 音符の長さをテキスト表現に変換（ticks → note name）
def tick_to_note_length(tick, ticks_per_beat):
    quarter = ticks_per_beat
    eighth = quarter // 2
    sixteenth = quarter // 4
    dotted_quarter = int(quarter * 1.5)
    dotted_eighth = int(eighth * 1.5)
    triplet_quarter = quarter // 3
    triplet_eighth = eighth // 3

    def is_close(value, target, tolerance=0.1):
        """値が目標値に近いかを判定（許容範囲は±10%）"""
        return abs(value - target) <= target * tolerance

    if is_close(tick, quarter):
        return "四分"
    elif is_close(tick, eighth):
        return "八分"
    elif is_close(tick, sixteenth):
        return "十六分"
    elif is_close(tick, quarter * 2):
        return "二分"
    elif is_close(tick, quarter * 4):
        return "全"
    elif is_close(tick, dotted_quarter):
        return "付点四分"
    elif is_close(tick, dotted_eighth):
        return "付点八分"
    elif is_close(tick, triplet_quarter):
        return "三連符（四分）"
    elif is_close(tick, triplet_eighth):
        return "三連符（八分）"
    elif is_close(tick, quarter * 2 + dotted_quarter):
        return "付点二分"
    else:
        return f"その他({tick})"


# MIDIノート番号を音階名に変換
def note_number_to_pitch(note_number, ignore_octave=False):
    """MIDIノート番号を音階名に変換"""
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    pitch_class = pitch_classes[note_number % 12]
    if ignore_octave:
        return pitch_class  # オクターブを無視
    octave = (note_number // 12) - 1
    return f"{pitch_class}{octave}"


# 指定トラックをパースして音符の長さのリストを作成
def extract_note_lengths(track, ticks_per_beat):
    note_lengths = []
    note_on_times = {}

    current_time = 0
    for msg in track:
        current_time += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            note_on_times[msg.note] = current_time
        elif msg.type in ["note_off", "note_on"] and msg.note in note_on_times:
            start_time = note_on_times.pop(msg.note)
            duration = current_time - start_time
            note_lengths.append(duration)

    # 長さを記号に変換
    return [tick_to_note_length(t, ticks_per_beat) for t in note_lengths]


# 指定トラックをパースして音階のリストを作成
def extract_note_pitches(track, ignore_octave=False):
    """トラック内の音符のピッチ（音階）を抽出"""
    note_pitches = []

    for msg in track:
        if msg.type == "note_on" and msg.velocity > 0:
            note_pitches.append(note_number_to_pitch(msg.note, ignore_octave))

    return note_pitches


# n-gramをカウント
def count_ngrams(note_items, n=3):
    """音符または音階のn-gramをカウント"""
    ngrams = [tuple(note_items[i : i + n]) for i in range(len(note_items) - n + 1)]
    return Counter(ngrams)


# メイン処理
def analyze_midi_file(
    filename,
    track_index=1,
    n=3,
    top_k=10,
    output_folder="output",
    mode="length",
    ignore_octave=False,
):
    if not os.path.exists(filename):
        print(f"❌ ファイルが見つかりません: {filename}")
        return

    try:
        mid = MidiFile(filename)
        if track_index >= len(mid.tracks):
            print(f"❌ 無効なトラックインデックス: {track_index}")
            return

        ticks_per_beat = mid.ticks_per_beat
        track = mid.tracks[track_index]

        if mode == "length":
            note_items = extract_note_lengths(track, ticks_per_beat)
            title_suffix = "音符並び"
            output_suffix = "lengths"
        elif mode == "pitch":
            note_items = extract_note_pitches(track, ignore_octave)
            title_suffix = "音階並び"
            output_suffix = "pitches"
            if ignore_octave:
                title_suffix += "（オクターブ無視）"
                output_suffix += "_ignore_octave"
        else:
            print(f"❌ 無効なモード: {mode}")
            return

        ngram_counts = count_ngrams(note_items, n)
        most_common = ngram_counts.most_common(top_k)

        # 表で表示
        df = pd.DataFrame(most_common, columns=[f"{n}-gram", "出現回数"])
        print(df)

        # グラフで表示
        labels = ["-".join(ng) for ng, _ in most_common]
        values = [count for _, count in most_common]

        plt.figure(figsize=(10, 5))
        plt.barh(labels, values)
        plt.xlabel("出現回数")
        plt.title(f"上位 {top_k} の{title_suffix} ({n}-gram)")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # 出力フォルダを作成
        os.makedirs(output_folder, exist_ok=True)

        # MIDIファイル名を利用してグラフを保存
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_path = os.path.join(
            output_folder, f"{base_name}_{n}-grams_{output_suffix}.png"
        )
        plt.savefig(output_path)
        print(f"✅ グラフを保存しました: {output_path}")

        plt.show()

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")


# 使用例
if __name__ == "__main__":
    filename = "output_midi/output_05_22-02.mid"
    analyze_midi_file(
        filename,
        track_index=1,
        n=2,
        top_k=10,
        mode="pitch",
        ignore_octave=True,  # オクターブを無視する場合はTrue
    )
    analyze_midi_file(
        filename,
        track_index=1,
        n=3,
        top_k=10,
        mode="pitch",
        ignore_octave=True,  # オクターブを無視する場合はTrue
    )
    analyze_midi_file(
        filename,
        track_index=1,
        n=4,
        top_k=10,
        mode="pitch",
        ignore_octave=True,  # オクターブを無視する場合はTrue
    )
