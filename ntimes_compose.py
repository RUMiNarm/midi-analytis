#!/usr/bin/env python
# -*- coding: utf‑8 -*-
"""
main.py
========
マルコフ連鎖によるメロディ自動生成ツール
------------------------------------------

◆ 概要
    * 指定フォルダ直下・配下にある **すべての MIDI ファイルの Track 0** を読み込み，
      「音高列」「音価列」の 2 系列を個別に N‑gram Markov 連鎖へ学習します。
    * 学習済みモデルから長さ *L* の音高列／音価列を生成し，MIDI (Format 0) として書き出します。
    * CLI から `--data_dir`, `--order`, `--length` などを指定可能です。

◆ 必要環境
    * Python ≥ 3.10
    * pip で取得可能な以下ライブラリ  
      ‑ pretty_midi, mido, numpy, argparse

◆ インストール（uv を使用）
    ```bash
    uv pip install -r requirements.txt
    ```

◆ 使い方
    ```bash
    uv python main.py \
        --data_dir  ./input_midis \
        --length    128 \
        --out       generated.mid \
        --order     3          # 省略可 (既定 3)
        --program   0          # 省略可 (既定 0 = Acoustic Grand Piano)
        --seed      42         # 省略可
    ```

◆ 生成ファイル
    * `generated.mid` : 80 BPM／単一トラックの GM 音源 MIDI

"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import mido
import numpy as np
import pretty_midi as pm


# --------------------------------------------------------------------------- #
#                               Public API 関数                               #
# --------------------------------------------------------------------------- #
def load_midis(path: str | Path) -> Tuple[List[int], List[float]]:
    """
    ディレクトリ配下のすべての *.mid / *.midi を再帰的に走査し，
    **Track 0** の Note イベントから音高列・音価列を抽出して返します。

    Parameters
    ----------
    path : str | pathlib.Path
        走査対象のフォルダ

    Returns
    -------
    pitches : List[int]
        0–127 の MIDI ノート番号列
    durations : List[float]
        4 分音符 = 1.0 とした拍単位の音価列
    """
    path = Path(path).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"ディレクトリが見つかりません: {path}")

    pitches: List[int] = []
    durations: List[float] = []

    for midi_path in sorted(path.rglob("*.mid*")):
        try:
            _extract_track0(midi_path, pitches, durations)
        except Exception as exc:
            print(f"[WARN] 解析失敗: {midi_path.name} ({exc})", file=sys.stderr)

    if len(pitches) == 0:
        raise RuntimeError("Track 0 から音符を抽出できる MIDI が見つかりませんでした。")

    return pitches, durations


def build_chain(notes: Sequence[int | float], n: int) -> Dict[Tuple, List]:
    """
    与えられた系列から N‑gram Markov 連鎖 (多重遷移辞書) を構築します。

    Parameters
    ----------
    notes : Sequence[int | float]
        学習データ列（音高 or 音価）
    n : int
        N‑gram の N （状態を構成する要素数）

    Returns
    -------
    chain : dict[tuple, list]
        key = 直近 N 個のタプル，value = 次状態候補のリスト（多重集合）
    """
    if n < 1:
        raise ValueError("`n` は 1 以上を指定してください。")
    if len(notes) <= n:
        raise ValueError("データ長が n より短く，モデルを構築できません。")

    chain: Dict[Tuple, List] = defaultdict(list)
    for i in range(len(notes) - n):
        state = tuple(notes[i : i + n])
        nxt = notes[i + n]
        chain[state].append(nxt)
    return chain


def generate_notes(
    model: Dict[Tuple, List],
    length: int,
    seed: int | None = None,
) -> List[int | float]:
    """
    N‑gram Markov 連鎖を用いて系列を生成します。

    Parameters
    ----------
    model : dict
        ``build_chain`` が返す遷移辞書
    length : int
        生成する要素数
    seed : int, optional
        乱数シード（再現性用）

    Returns
    -------
    seq : List[int | float]
        生成系列
    """
    if length < 1:
        raise ValueError("`length` は 1 以上を指定してください。")

    rng = random.Random(seed)

    # ランダムに初期 state を選択
    state = rng.choice(list(model.keys()))
    seq: List[int | float] = list(state)

    # length 個になるまで生成
    while len(seq) < length:
        candidates = model.get(state)
        if not candidates:  # デッドエンド → 再抽選
            state = rng.choice(list(model.keys()))
            seq.extend(list(state))
            continue
        nxt = rng.choice(candidates)
        seq.append(nxt)
        state = tuple(seq[-len(state) :])

    return seq[:length]


def save_midi(
    notes: Sequence[Tuple[int, float]],
    out_file: str | Path,
    program: int = 0,
    tempo_bpm: int = 80,
    velocity: int = 100,
) -> None:
    """
    音高＋音価列を PrettyMIDI で MIDI ファイルとして保存します。

    Parameters
    ----------
    notes : Sequence[(pitch, dur_beat)]
        (音高, 4 分音符 = 1.0 の拍数) のタプル列
    out_file : str | pathlib.Path
        書き出し先 *.mid
    program : int, default 0
        GM プログラム番号 (0–127)
    tempo_bpm : int, default 80
        テンポ (BPM)
    velocity : int, default 100
        ノートオン速度
    """
    midi = pm.PrettyMIDI(initial_tempo=tempo_bpm)
    inst = pm.Instrument(program=program)
    time = 0.0
    sec_per_beat = 60.0 / tempo_bpm

    for pitch, dur in notes:
        if dur <= 0:
            continue
        start = time
        end = start + dur * sec_per_beat
        note = pm.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=end,
        )
        inst.notes.append(note)
        time = end

    midi.instruments.append(inst)
    out_file = str(out_file)
    midi.write(out_file)
    print(f"[INFO] 書き出しました → {out_file}")


def main() -> None:
    """
    CLI エントリポイント。`argparse` でオプションを解釈して処理を実行します。
    """
    parser = argparse.ArgumentParser(
        description="N‑gram Markov 連鎖による MIDI メロディ生成ツール"
    )
    parser.add_argument(
        "--data_dir", required=True, help="学習用 MIDI が入ったフォルダ"
    )
    parser.add_argument(
        "--order",
        type=int,
        default=3,
        help="Markov 連鎖の次数 (デフォルト: 3)",
    )
    parser.add_argument(
        "--length",
        type=int,
        required=True,
        help="生成する音符数",
    )
    parser.add_argument(
        "--out",
        default="output.mid",
        help="生成 MIDI の出力ファイル名 (デフォルト: output.mid)",
    )
    parser.add_argument(
        "--program",
        type=int,
        default=0,
        help="GM プログラム番号 0–127 (デフォルト: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="乱数シード (省略可)",
    )
    args = parser.parse_args()

    try:
        # 1) データ読み込み
        pitches, durations = load_midis(args.data_dir)

        # 2) モデル構築
        pitch_chain = build_chain(pitches, args.order)
        dur_chain = build_chain(durations, args.order)

        # 3) 生成
        rng_seed = args.seed
        gen_pitches = generate_notes(pitch_chain, args.length, rng_seed)
        gen_durs = generate_notes(
            dur_chain, args.length, None if rng_seed is None else rng_seed + 1
        )

        # 4) 書き出し
        save_midi(
            list(zip(gen_pitches, gen_durs)),
            args.out,
            program=args.program,
            tempo_bpm=80,
        )

        # 5) requirements.txt 自動生成（存在しない場合のみ）
        _write_requirements()

    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


# --------------------------------------------------------------------------- #
#                              Internal Helper 関数                           #
# --------------------------------------------------------------------------- #
def _extract_track0(
    midi_path: Path,
    pitches: List[int],
    durations: List[float],
) -> None:
    """Track 0 から Note 列を抽出してリストに追記する。"""
    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat
    time = 0
    active: Dict[int, int] = {}

    for msg in mid.tracks[0]:
        time += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = time
        elif msg.type in ("note_off", "note_on") and (
            msg.type == "note_off" or msg.velocity == 0
        ):
            start = active.pop(msg.note, None)
            if start is None:
                continue
            dur_ticks = time - start
            dur_beats = dur_ticks / ticks_per_beat
            if dur_beats <= 0:
                continue
            pitches.append(msg.note)
            durations.append(dur_beats)


def _write_requirements(req_path: str | Path = "requirements.txt") -> None:
    """pretty_midi / mido / numpy が未記載なら requirements.txt を生成・追記する。"""
    req_path = Path(req_path)
    pkgs = ["pretty_midi>=0.2", "mido>=1.2", "numpy>=1.22"]
    if req_path.exists():
        existing = {
            line.strip().split("==")[0].split(">=")[0]
            for line in req_path.read_text().splitlines()
        }
        pkgs = [p for p in pkgs if p.split(">=")[0] not in existing]
        if not pkgs:
            return
        with req_path.open("a", encoding="utf-8") as fp:
            fp.write("\n" + "\n".join(pkgs) + "\n")
    else:
        req_path.write_text("\n".join(pkgs) + "\n", encoding="utf-8")
    print(f"[INFO] requirements.txt を更新しました → {req_path}")


# --------------------------------------------------------------------------- #
#                                    Main                                     #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
