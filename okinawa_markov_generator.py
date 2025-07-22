#!/usr/bin/env python3
"""
okinawa_markov_generator.py
===========================
Crafts Okinawan‑style melodies by learning tri‑note *motifs* and their time‑values
from a MIDI corpus (monophonic, C‑centred) and sampling them with a first‑order
Markov chain – mirroring the two‑stage approach in **Zhang (2025)** but tuned for
the **Ryukyu (琉球) pentatonic** scale:

    ♩ C  E  F  G  B   (degrees 1‑3‑4‑5‑7)

Highlights
~~~~~~~~~~
* **Automatic motif dictionary** – every unique consecutive 3‑note slice that is
  *strictly inside* the Ryukyu scale becomes a state; no hard‑coded list.
* **Rhythmic modelling** – records the triplet of *delta‑times* alongside each
  motif, then samples a matching rhythm pattern when the motif is re‑used.
* **Flexible learning** – any number of .mid files; the script picks the most
  note‑dense track if multi‑track.
* **Zero external data** – fully learns states & probabilities from the input.

Quick‑start
~~~~~~~~~~~
```
# 1. Install deps
pip install numpy mido python-rtmidi

# 2. Train on Okinawan corpus & export 64 motifs
python okinawa_markov_generator.py \
       --corpus asadoyaYunta_CMajor.mid jinjin.mid karafune_C.mid \
       --motifs 64 --out okinawa.mid
```
Optional `--play` can live‑stream the output if a MIDI port is available.

Implementation notes
~~~~~~~~~~~~~~~~~~~~
* Ryukyu mapping is encoded in `_RYUKYU_PC_TO_DEGREE`.
* Motif state key = `(deg1,deg2,deg3)` *tuple*; rhythm key = matching tuple of
  *ticks* differences.  A defaultdict[Counter] stores counts → transition matrix.
* Tempo defaults to 120 BPM; adjust with `--tempo`.

Author: ChatGPT o3, 2025‑06‑13
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import mido
import numpy as np

###############################################################################
# Scale helpers                                                               #
###############################################################################

_RYUKYU_PC_TO_DEGREE = {0: "1", 4: "3", 5: "4", 7: "5", 11: "7"}
_DEGREE_TO_PC = {v: k for k, v in _RYUKYU_PC_TO_DEGREE.items()}


def pc_to_degree(pc: int) -> str | None:
    """Return scale degree label ('1','3','4','5','7') for pitch‑class or None."""
    return _RYUKYU_PC_TO_DEGREE.get(pc)


###############################################################################
# Corpus extraction                                                           #
###############################################################################


def extract_leading_track(mid: mido.MidiFile) -> List[mido.Message]:
    """Return messages of the track with most note_on events."""
    best = max(
        mid.tracks,
        key=lambda t: sum(1 for m in t if m.type == "note_on" and m.velocity),
    )
    return best


def track_to_melody(track: Sequence[mido.Message]) -> Tuple[List[int], List[int]]:
    """Convert a monophonic track to (notes, durations) in ticks.
    Assumes note_on/note_off pairs & ignores overlap errors.
    """
    notes: List[int] = []
    durs: List[int] = []
    time_acc = 0
    on_note = None
    on_time = 0
    for msg in track:
        time_acc += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            if on_note is not None:
                # handle legato: end previous note here
                notes.append(on_note)
                durs.append(time_acc - on_time)
            on_note = msg.note
            on_time = time_acc
        elif (msg.type == "note_off") or (msg.type == "note_on" and msg.velocity == 0):
            if on_note is not None:
                notes.append(on_note)
                durs.append(time_acc - on_time)
                on_note = None
    return notes, durs


def melody_to_motifs(
    notes: List[int], durs: List[int]
) -> List[Tuple[Tuple[str, str, str], Tuple[int, int, int]]]:
    motifs = []
    for i in range(len(notes) - 2):
        pcs = [n % 12 for n in notes[i : i + 3]]
        degs = [pc_to_degree(pc) for pc in pcs]
        if None in degs:
            continue  # skip motif containing out‑of‑scale note
        motif_key = tuple(degs)  # ('1','3','4')
        rhythm_key = tuple(durs[i : i + 3])
        motifs.append((motif_key, rhythm_key))
    return motifs


def load_corpus(paths: Sequence[Path]):
    trans_counts: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)
    rhythm_dict: Dict[Tuple[str, str, str], List[Tuple[int, int, int]]] = defaultdict(
        list
    )

    for p in paths:
        mid = mido.MidiFile(str(p))
        track = extract_leading_track(mid)
        notes, durs = track_to_melody(track)
        motifs = melody_to_motifs(notes, durs)
        for (motif, rhythm), (next_motif, _) in zip(motifs, motifs[1:]):
            trans_counts[motif][next_motif] += 1
            rhythm_dict[motif].append(rhythm)
        if motifs:  # add rhythm for last motif
            rhythm_dict[motifs[-1][0]].append(motifs[-1][1])

    states = list(trans_counts.keys())
    idx = {m: i for i, m in enumerate(states)}
    n = len(states)
    mat = np.zeros((n, n), dtype=float)
    for m_from, counter in trans_counts.items():
        row = mat[idx[m_from]]
        total = sum(counter.values())
        for m_to, c in counter.items():
            row[idx[m_to]] = c / total

    return states, mat, rhythm_dict


###############################################################################
# Generation                                                                  #
###############################################################################


def sample_next(probs: np.ndarray) -> int:
    return int(np.random.choice(len(probs), p=probs))


def generate_sequence(
    states: List[Tuple[str, str, str]],
    trans: np.ndarray,
    rhythm_dict: Dict[Tuple[str, str, str], List[Tuple[int, int, int]]],
    length: int,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[int, int, int]]]:
    rng = np.random.default_rng()
    idx = rng.integers(len(states))
    seq_states = [states[idx]]
    seq_rhythm = [rng.choice(rhythm_dict[states[idx]])]

    for _ in range(length - 1):
        idx = sample_next(trans[idx])
        state = states[idx]
        seq_states.append(state)
        seq_rhythm.append(rng.choice(rhythm_dict[state]))
    return seq_states, seq_rhythm


###############################################################################
# MIDI export                                                                 #
###############################################################################

_DEGREE_TO_MIDI_C4 = {"1": 60, "3": 64, "4": 65, "5": 67, "7": 71}  # Ryukyu in C4


def degrees_to_pitches(
    degs: Tuple[str, str, str], octave_shift: int = 0
) -> Tuple[int, int, int]:
    return tuple(_DEGREE_TO_MIDI_C4[d] + octave_shift * 12 for d in degs)


def sequence_to_midi(
    states: List[Tuple[str, str, str]],
    rhythms: List[Tuple[int, int, int]],
    tempo: int = 500000,
    ticks_per_beat: int = 480,
) -> mido.MidiFile:
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    cursor = 0
    for motif, rhythm in zip(states, rhythms):
        notes = degrees_to_pitches(motif)
        for note, dur in zip(notes, rhythm):
            tr.append(mido.Message("note_on", note=note, velocity=80, time=cursor))
            tr.append(mido.Message("note_off", note=note, velocity=0, time=dur))
            cursor = 0
        # ensure last note's off time gap not duplicated in next note_on
        cursor += 0
    return mid


###############################################################################
# CLI                                                                         #
###############################################################################


def main(argv=None):
    ap = argparse.ArgumentParser(description="Okinawan motif Markov generator")
    ap.add_argument(
        "--corpus",
        nargs="+",
        type=Path,
        required=True,
        help="MIDI corpus files (Ryukyu) in C key",
    )
    ap.add_argument(
        "--motifs", type=int, default=32, help="Number of motifs to generate"
    )
    ap.add_argument("--out", type=Path, help="Output MIDI file")
    ap.add_argument("--play", action="store_true", help="Send to first MIDI OUT port")
    ap.add_argument(
        "--tempo", type=int, default=500000, help="µs per quarter note, default 120 BPM"
    )
    args = ap.parse_args(argv)

    print("Loading corpus & building model…")
    states, trans, rhythms = load_corpus(args.corpus)
    print(f"Learned {len(states)} unique motifs.")

    motifs, rtm = generate_sequence(states, trans, rhythms, args.motifs)
    print("Generated motif sequence (degrees):")
    print(" ".join(["".join(m) for m in motifs]))

    mid = sequence_to_midi(motifs, rtm, tempo=args.tempo)

    if args.out:
        mid.save(args.out)
        print(f"MIDI saved → {args.out}")
    if args.play:
        try:
            port_name = mido.get_output_names()[0]
            with mido.open_output(port_name) as port:
                for m in mid.play():
                    port.send(m)
        except IndexError:
            print("No MIDI OUT port available. Skipping playback.", file=sys.stderr)


if __name__ == "__main__":
    main()
