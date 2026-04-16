from pathlib import Path

from ase import Atoms
from ase.io import write

from asetui.io import read_all_frames


def test_read_all_frames_flattens_multiple_inputs(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write(first, [Atoms("H", positions=[(0.0, 0.0, 0.0)]), Atoms("He", positions=[(0.0, 0.0, 0.0)])])
    write(second, Atoms("Li", positions=[(0.0, 0.0, 0.0)]))

    frames = read_all_frames([first, second])

    assert [atoms.get_chemical_formula() for atoms in frames] == ["H", "He", "Li"]


def test_read_all_frames_applies_slice_per_input_before_flattening(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write(first, [Atoms("H", positions=[(0.0, 0.0, 0.0)]), Atoms("He", positions=[(0.0, 0.0, 0.0)])])
    write(second, [Atoms("Li", positions=[(0.0, 0.0, 0.0)]), Atoms("Be", positions=[(0.0, 0.0, 0.0)])])

    frames = read_all_frames([f"{first}@-1", f"{second}@:1"])

    assert [atoms.get_chemical_formula() for atoms in frames] == ["He", "Li"]
