#!/usr/bin/env python3
"""markov_composer.py

二階マルコフ連鎖（直前2音 → 次音）を利用した自動メロディ生成スクリプト。
ピッチ列と音価列を独立に学習し、統計的に類似した新メロディを生成します。

◤ 変更点（★）
────────────────────────────────
* **ワイルドカード対応**：学習用 MIDI に `input/okinawa/*.mid` のような
  パターンを直接渡せます（Windows CMD／`uv run` でも OK）。
* glob で展開し、マッチしない場合はそのままファイル名として扱います。

参考論文：Zheng, X. et al. “An automatic composition model of Chinese folk music”.
"""

from __future__ import annotations

import argparse
import glob
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple

from music21 import converter, meter, note, stream, tempo

# 型エイリアス -------------------------------------------------------------
PitchState = Tuple[int, int]  # (prev2, prev1) → 次のピッチを予測
DurationState = Tuple[float, float]  # (prev2, prev1) → 次の音価を予測


class SecondOrderMarkov:
    """二階マルコフモデル（汎用）。"""

    def __init__(self):
        self.counts: defaultdict[Tuple, Counter] = defaultdict(Counter)
        self.prob: dict[Tuple, List[Tuple]] = {}

    # ---------------- 学習 ----------------
    def add_sequence(self, seq: List):
        if len(seq) < 3:
            return
        for i in range(2, len(seq)):
            prev = (seq[i - 2], seq[i - 1])
            curr = seq[i]
            self.counts[prev][curr] += 1

    def finalize(self):
        for prev, counter in self.counts.items():
            total = sum(counter.values())
            cumulative = []
            acc = 0
            for k, c in counter.items():
                acc += c
                cumulative.append((k, acc / total))
            self.prob[prev] = cumulative

    # ---------------- 推論 ----------------
    def next(self, state):
        if state not in self.prob:
            state = random.choice(list(self.prob.keys()))
        r = random.random()
        for value, threshold in self.prob[state]:
            if r <= threshold:
                return value
        return self.prob[state][-1][0]


# -------------------------------------------------------------------------
# データ抽出
# -------------------------------------------------------------------------


def extract_melody_parts(midi_path: Path) -> Tuple[List[int], List[float]]:
    s = converter.parse(midi_path)
    try:
        part = s.parts[0].flat.notes
    except Exception:
        part = s.flat.notes

    pitches, durations = [], []
    for n in part:
        if isinstance(n, note.Note):
            pitches.append(n.pitch.midi)
            durations.append(float(n.quarterLength))
    return pitches, durations


# -------------------------------------------------------------------------
# モデル構築
# -------------------------------------------------------------------------


def build_models(midi_files: List[Path]):
    pitch_model, dur_model = SecondOrderMarkov(), SecondOrderMarkov()
    for f in midi_files:
        p, d = extract_melody_parts(f)
        pitch_model.add_sequence(p)
        dur_model.add_sequence(d)
    pitch_model.finalize()
    dur_model.finalize()
    return pitch_model, dur_model


# -------------------------------------------------------------------------
# メロディ生成
# -------------------------------------------------------------------------


def generate_melody(
    pitch_model: SecondOrderMarkov,
    dur_model: SecondOrderMarkov,
    measures: int = 16,
    time_sig: str = "4/4",
    tempo_bpm: int = 120,
    seed_pitches: PitchState | None = None,
    seed_durations: DurationState | None = None,
):
    if seed_pitches is None:
        seed_pitches = random.choice(list(pitch_model.prob.keys()))
    if seed_durations is None:
        seed_durations = random.choice(list(dur_model.prob.keys()))

    pitches, durations = list(seed_pitches), list(seed_durations)
    beats_per_measure = int(time_sig.split("/")[0])
    target_quarters = measures * beats_per_measure

    total_q = sum(durations)
    while total_q < target_quarters:
        pitches.append(pitch_model.next((pitches[-2], pitches[-1])))
        dur_next = dur_model.next((durations[-2], durations[-1]))
        durations.append(dur_next)
        total_q += dur_next

    melody = stream.Stream()
    melody.append(tempo.MetronomeMark(number=tempo_bpm))
    melody.append(meter.TimeSignature(time_sig))
    for p, d in zip(pitches[2:], durations[2:]):
        melody.append(note.Note(pitch=p, quarterLength=d))
    return melody


# -------------------------------------------------------------------------
# MIDI 出力
# -------------------------------------------------------------------------


def save_midi(s: stream.Stream, path: Path):
    s.write("midi", fp=path)
    print(f"MIDI ファイルを保存しました → {path}")


# -------------------------------------------------------------------------
# CLI エントリポイント
# -------------------------------------------------------------------------


def cli():
    parser = argparse.ArgumentParser(description="二階マルコフ連鎖によるメロディ生成器")
    parser.add_argument(
        "midi_patterns",
        nargs="+",
        help="学習用 MIDI ファイルまたはワイルドカードパターン",
    )
    parser.add_argument("output", type=Path, help="出力先 MIDI ファイル")
    parser.add_argument("--measures", type=int, default=16, help="生成する小節数")
    parser.add_argument("--tempo", type=int, default=120, help="テンポ (BPM)")
    args = parser.parse_args()

    # ★ ワイルドカード展開 --------------------------------------------------
    expanded: List[Path] = []
    for pat in args.midi_patterns:
        matches = [Path(m) for m in glob.glob(pat)]
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(Path(pat))  # マッチしない → そのまま扱う

    if not expanded:
        parser.error("指定パターンに一致する MIDI ファイルがありません。")

    expanded = sorted(expanded)
    print("学習ファイル数:", len(expanded))

    pitch_model, dur_model = build_models(expanded)
    comp = generate_melody(
        pitch_model, dur_model, measures=args.measures, tempo_bpm=args.tempo
    )
    save_midi(comp, args.output)


if __name__ == "__main__":
    cli()
