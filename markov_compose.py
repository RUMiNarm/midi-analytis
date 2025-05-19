#!/usr/bin/env python3
"""markov_composer.py

二階マルコフ連鎖（直前2音 → 次音）を利用した自動メロディ生成スクリプトです。
ピッチ列と音価列をそれぞれ独立に学習することで、元楽曲の統計的特徴を真似た
新しい旋律を作り出します。本実装は以下の論文を参考にしています。

  Zheng, X. ほか「An automatic composition model of Chinese folk music」
  AIP Conference Proceedings 1820, 080003 (2017).

───────────────────────────────────────────
■ 必要環境
  * Python 3.9 以上
  * music21 8.0 以上     →  pip install music21

■ 使い方（最小例）
  jasmine.mid を学習し，18 小節のメロディを output.mid に出力:

      python markov_composer.py jasmine.mid output.mid --measures 18

  複数ファイルをまとめて渡すと連結して学習します。

■ 主なオプション
  --measures N   生成する小節数（既定 16）
  --tempo BPM    生成MIDI のテンポ（既定 120）
  --seed P,D     初期 2 音のピッチ (MIDI 値または音名) と音価 (quarterLength)
                 をカンマ区切りで指定（例: 60,62,1.0,0.5）
"""

from __future__ import annotations

import argparse
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
        # counts[(a,b)][c] = 出現回数
        self.counts: defaultdict[Tuple, Counter] = defaultdict(Counter)
        # prob[(a,b)] = [(c1, 0.5), (c2, 0.8), ...]  累積確率リスト
        self.prob: dict[Tuple, List[Tuple]] = {}

    # ---------------- 学習 ----------------
    def add_sequence(self, seq: List):
        """シーケンスを取り込み、(前々項,前項)→次項 の頻度を更新。"""
        if len(seq) < 3:
            return  # 学習には最低 3 要素必要
        for i in range(2, len(seq)):
            prev = (seq[i - 2], seq[i - 1])
            curr = seq[i]
            self.counts[prev][curr] += 1

    def finalize(self):
        """学習完了後に呼び出し、累積確率テーブルを構築。"""
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
        """与えられた状態から次値をサンプリング（未学習状態ならランダム）。"""
        if state not in self.prob:
            state = random.choice(list(self.prob.keys()))
        r = random.random()
        for value, threshold in self.prob[state]:
            if r <= threshold:
                return value
        # ここに到達することはほぼない
        return self.prob[state][-1][0]


# -------------------------------------------------------------------------
# データ抽出
# -------------------------------------------------------------------------


def extract_melody_parts(midi_path: Path) -> Tuple[List[int], List[float]]:
    """メロディライン（ピッチ列と音価列）を取り出して返す。"""
    s = converter.parse(midi_path)
    # メロディと仮定できる最上パート、無ければ全パート平坦化
    try:
        part = s.parts[0].flat.notes
    except Exception:
        part = s.flat.notes

    melody_notes = []
    for n in part:
        if isinstance(n, note.Note):
            melody_notes.append(n)
        elif isinstance(n, note.Rest):
            # 休符は無視
            continue

    pitches = [n.pitch.midi for n in melody_notes]
    durations = [float(n.quarterLength) for n in melody_notes]
    return pitches, durations


# -------------------------------------------------------------------------
# モデル構築
# -------------------------------------------------------------------------


def build_models(midi_files: List[Path]) -> Tuple[SecondOrderMarkov, SecondOrderMarkov]:
    """複数MIDIからピッチモデルと音価モデルを学習して返す。"""
    pitch_model = SecondOrderMarkov()
    dur_model = SecondOrderMarkov()
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
) -> stream.Stream:
    """学習済みモデルから指定小節数のメロディを生成し、music21 Stream を返す。"""

    # シードが未指定ならランダムに取得
    if seed_pitches is None:
        seed_pitches = random.choice(list(pitch_model.prob.keys()))
    if seed_durations is None:
        seed_durations = random.choice(list(dur_model.prob.keys()))

    pitches = [*seed_pitches]
    durations = [*seed_durations]

    # 目標の総四分音符数（ざっくり）
    beats_per_measure = int(time_sig.split("/")[0])
    target_quarters = measures * beats_per_measure

    total_quarters = sum(durations)
    while total_quarters < target_quarters:
        next_pitch = pitch_model.next((pitches[-2], pitches[-1]))
        next_dur = dur_model.next((durations[-2], durations[-1]))
        pitches.append(next_pitch)
        durations.append(next_dur)
        total_quarters += next_dur

    # music21 Stream を構築
    melody = stream.Stream()
    melody.append(tempo.MetronomeMark(number=tempo_bpm))
    melody.append(meter.TimeSignature(time_sig))
    for p, d in zip(pitches[2:], durations[2:]):  # シード2音はスキップ
        n = note.Note(pitch=p, quarterLength=d)
        melody.append(n)
    return melody


# -------------------------------------------------------------------------
# MIDI 出力
# -------------------------------------------------------------------------


def save_midi(s: stream.Stream, path: Path):
    """生成した Stream を MIDI ファイルとして保存。"""
    s.write("midi", fp=path)
    print(f"MIDI ファイルを保存しました → {path}")


# -------------------------------------------------------------------------
# CLI エントリポイント
# -------------------------------------------------------------------------


def cli():
    ap = argparse.ArgumentParser(description="二階マルコフ連鎖によるメロディ生成器")
    ap.add_argument("input", nargs="+", type=Path, help="学習用 MIDI ファイル群")
    ap.add_argument("output", type=Path, help="出力先 MIDI ファイル")
    ap.add_argument("--measures", type=int, default=16, help="生成する小節数")
    ap.add_argument("--tempo", type=int, default=120, help="テンポ (BPM)")
    args = ap.parse_args()

    pitch_model, dur_model = build_models(args.midi_files)
    comp = generate_melody(
        pitch_model, dur_model, measures=args.measures, tempo_bpm=args.tempo
    )
    save_midi(comp, args.output)


if __name__ == "__main__":
    cli()
