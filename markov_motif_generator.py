#!/usr/bin/env python3
"""
markov_motif_generator.py
---------------------------------
Re‑implementation of the Markov‑chain based **tri‑note motif** generator
presented in the paper

    Zhang, C. (2025) *The analysis of Chinese National ballad composition
    education based on artificial intelligence and deep learning*,
    *Scientific Reports*, 15:9215.

This version **fixes an earlier bug** where two state labels had four notes
instead of three, causing a runtime error.

Changes v1.1 – 2025‑06‑13
~~~~~~~~~~~~~~~~~~~~~~~~
*   Replaced invalid labels ``"3H5H6H2"`` and ``"6H5H3H2"`` with proper
    tri‑note motifs ``"5H6H1H"`` (ascending) and ``"6H3H2H"`` (descending).
*   Added a validation assert so the script will fail fast at startup if any
    state decodes to other than 3 notes.
*   Minor docstring cleanup.

Usage example
~~~~~~~~~~~~
    # 32 motifs trained from folk.mid → out.mid
    python markov_motif_generator.py --corpus folk.mid --motifs 32 --out out.mid

Dependencies
~~~~~~~~~~~~
    pip install numpy mido python-rtmidi

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import mido
import numpy as np

###############################################################################
# 0.  Pentatonic‑motif state space                                          #
###############################################################################

_DEGREE_TO_SEMITONE = {"1": 0, "2": 2, "3": 4, "5": 7, "6": 9}


def _degree_to_midi(deg: str, oct_shift: int = 0, tonic: int = 60) -> int:
    return tonic + _DEGREE_TO_SEMITONE[deg] + 12 * oct_shift


# ------------------------------------------------------------------------- #
# Valid 22 tri‑note states (11 ascending + 11 descending)
# ------------------------------------------------------------------------- #
ASC_STATES: Tuple[str, ...] = (
    "123",
    "235",
    "216L",
    "6L1L5L",
    "561H",
    "356",
    "3H5H6H",
    "2H1H6",
    "2H3H5H",
    "1H2H3H",
    "5H6H1H",  # fixed
)
DESC_STATES: Tuple[str, ...] = (
    "321",
    "532",
    "6L12",
    "1L6L5L",
    "1H6H5",
    "653",
    "6H5H3H",
    "6H1H2H",
    "5H3H2H",
    "3H2H1H",
    "6H3H2H",  # fixed
)
ALL_STATES: Tuple[str, ...] = ASC_STATES + DESC_STATES
assert len(ALL_STATES) == 22, "State list must have exactly 22 entries"

###############################################################################
# 1.  Helpers                                                               #
###############################################################################


def _state_to_pitches(state: str, tonic: int = 60) -> Tuple[int, int, int]:
    notes: List[int] = []
    i = 0
    while i < len(state):
        ch = state[i]
        if ch not in "12356":
            raise ValueError(f"Malformed state token: {state!r}")
        deg = ch
        oct_shift = 0
        j = i + 1
        while j < len(state) and state[j] in "LH":
            oct_shift += -1 if state[j] == "L" else 1
            j += 1
        notes.append(_degree_to_midi(deg, oct_shift, tonic))
        i = j
    if len(notes) != 3:
        raise ValueError(f"State {state!r} decoded to {len(notes)} notes; expected 3")
    return tuple(notes)


# Validate all states at import time
for _lbl in ALL_STATES:
    _state_to_pitches(_lbl)  # will raise if malformed

###############################################################################
# 2.  Markov model (unchanged from v1.0)                                    #
###############################################################################


class MotifMarkov:
    def __init__(self):
        self.counts = np.zeros((22, 22), dtype=np.int64)
        self.trans = np.zeros((22, 22), dtype=np.float64)

    def train(self, seq: Sequence[int]):
        for a, b in zip(seq, seq[1:]):
            self.counts[a, b] += 1

    def normalise(self, smoothing: float = 1.0):
        mat = self.counts + smoothing
        self.trans = mat / mat.sum(axis=1, keepdims=True)

    def generate(self, n: int, seed: int | None = None) -> List[int]:
        if seed is None:
            seed = int(np.random.choice(22))
        out = [seed]
        for _ in range(n - 1):
            out.append(int(np.random.choice(22, p=self.trans[out[-1]])))
        return out


###############################################################################
# 3.  Corpus utilities (unchanged)                                          #
###############################################################################


def _parse_text(path: Path) -> List[int]:
    toks = path.read_text().split()
    return [ALL_STATES.index(t) for t in toks]


def _extract_from_midi(path: Path, tonic: int = 60) -> List[int]:
    mid = mido.MidiFile(str(path))
    melody = [
        m.note for tr in mid.tracks for m in tr if m.type == "note_on" and m.velocity
    ]
    seq: List[int] = []
    for i in range(len(melody) - 2):
        tri = tuple(melody[i : i + 3])
        for idx, lbl in enumerate(ALL_STATES):
            if tri == _state_to_pitches(lbl, tonic):
                seq.append(idx)
                break
    return seq


def corpus_sequence(paths: List[Path]) -> List[int]:
    seq: List[int] = []
    for p in paths:
        seq.extend(
            _extract_from_midi(p)
            if p.suffix.lower() in {".mid", ".midi"}
            else _parse_text(p)
        )
    if not seq:
        raise RuntimeError("No motif states found in corpus")
    return seq


###############################################################################
# 4.  MIDI export (unchanged)                                               #
###############################################################################


def states_to_midi(
    ids: Sequence[int], tonic: int = 60, tempo: int = 500000, ticks: int = 480
):
    mid = mido.MidiFile(ticks_per_beat=ticks)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    cursor = 0
    step = ticks // 3
    for sid in ids:
        notes = _state_to_pitches(ALL_STATES[sid], tonic)
        for i, n in enumerate(notes):
            tr.append(
                mido.Message("note_on", note=n, velocity=80, time=(0 if i else cursor))
            )
            tr.append(mido.Message("note_off", note=n, velocity=0, time=step))
            cursor = 0
        cursor += ticks - step * 3
    return mid


###############################################################################
# 5.  CLI (minor cosmetic tweaks)                                           #
###############################################################################


def main(argv: List[str] | None = None):
    ap = argparse.ArgumentParser(
        description="Tri‑note motif generator via Markov chain (v1.1)"
    )
    ap.add_argument("--corpus", nargs="*", type=Path, help="MIDI or .txt corpus files")
    ap.add_argument("--motifs", type=int, default=16)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args(argv)

    model = MotifMarkov()
    if args.corpus:
        model.train(corpus_sequence(args.corpus))
        model.normalise()
    else:
        model.trans[:] = 1 / 22.0

    seq = model.generate(args.motifs, seed=args.seed)
    print("Generated motif sequence (state labels):")
    print(" ".join(ALL_STATES[i] for i in seq))

    if args.out or args.play:
        mid = states_to_midi(seq)
        if args.out:
            mid.save(args.out)
            print(f"MIDI saved to {args.out}")
        if args.play:
            try:
                port = mido.get_output_names()[0]
                with mido.open_output(port) as p:
                    for m in mid.play():
                        p.send(m)
            except IndexError:
                print("--play requested but no MIDI OUT ports found", file=sys.stderr)


if __name__ == "__main__":
    main()
