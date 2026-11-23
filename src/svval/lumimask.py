"""
Golden-JSON lumi mask, vectorized. Same semantics as KUCMSAodSkimmer::isValidLumisection:
a (run, lumi) is good if the run is in the JSON and lumi is inside one of its [start, end]
ranges (both ends inclusive); a run not in the JSON is bad.

The ntuples are NOT lumi-masked at CRAB level, so this mask is required.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Iterable, Optional

import numpy as np

DEFAULT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'golden_json')


class LumiMask:
    def __init__(self, json_files: Iterable[str]):
        starts, ends = [], []
        self.files = []
        for path in json_files:
            with open(path) as f:
                data = json.load(f)
            self.files.append(os.path.basename(path))
            for run, ranges in data.items():
                r = np.uint64(int(run)) << np.uint64(32)
                for a, b in ranges:
                    if b < a:
                        raise ValueError(f"{path}: run {run} has a reversed range [{a}, {b}]")
                    starts.append(r | np.uint64(a))
                    ends.append(r | np.uint64(b))
        if not starts:
            raise ValueError("no lumi ranges loaded")
        starts = np.array(starts, dtype=np.uint64)
        ends = np.array(ends, dtype=np.uint64)
        order = np.argsort(starts)
        self._starts, self._ends = starts[order], ends[order]
        if np.any(self._starts[1:] <= self._ends[:-1]):
            raise ValueError("overlapping lumi ranges in the golden JSONs")

    @classmethod
    def from_dir(cls, directory: Optional[str] = None) -> "LumiMask":
        d = os.path.abspath(directory or DEFAULT_DIR)
        files = sorted(glob.glob(os.path.join(d, '*.json')) + glob.glob(os.path.join(d, '*.txt')))
        if not files:
            raise FileNotFoundError(f"no golden JSONs in {d}")
        return cls(files)

    def __call__(self, run, lumi) -> np.ndarray:
        run = np.asarray(run).astype(np.uint64)
        lumi = np.asarray(lumi).astype(np.uint64)
        key = (run << np.uint64(32)) | lumi
        idx = np.searchsorted(self._starts, key, side='right') - 1
        ok = idx >= 0
        out = np.zeros(key.shape, dtype=bool)
        out[ok] = key[ok] <= self._ends[idx[ok]]
        return out
