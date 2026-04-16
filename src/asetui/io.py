from __future__ import annotations

from pathlib import Path

from ase.atoms import Atoms
from ase.io import read


def read_atoms(path: str | Path) -> Atoms:
    """Read an ASE structure file into a single Atoms object."""
    atoms = read(Path(path))
    if isinstance(atoms, list):
        if not atoms:
            raise ValueError("No frames found in input file.")
        return atoms[0]
    return atoms


def read_all_frames(path: str | Path) -> list:
    """Read frames from an ASE-supported file. Returns a list of Atoms.

    Supports ASE slice notation: path@index, e.g. traj.xyz@:10 or traj.xyz@-1.
    Without a slice, all frames are read.
    """
    path_str = str(path)
    if "@" in path_str:
        filepath, index_str = path_str.rsplit("@", 1)
    else:
        filepath, index_str = path_str, ":"
    result = read(filepath, index=index_str)
    if isinstance(result, Atoms):
        return [result]
    if not result:
        raise ValueError("No frames found in input file.")
    return list(result)
