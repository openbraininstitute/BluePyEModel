"""
Copyright 2023-2024 Blue Brain Project / EPFL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pickle
from unittest.mock import MagicMock

import pytest

from bluepyemodel.emodel_pipeline.emodel_metadata import EModelMetadata
from bluepyemodel.tools.utils import deduplicate_checkpoint_paths
from bluepyemodel.tools.utils import existing_checkpoint_paths
from bluepyemodel.tools.utils import read_checkpoint


def test_deduplicate_checkpoint_paths_prefers_pickle():
    paths = [
        "checkpoints/L5PC/test/emodel=L5PC__seed=0.h5",
        "checkpoints/L5PC/test/emodel=L5PC__seed=0.pkl",
    ]

    deduplicated = deduplicate_checkpoint_paths(paths)

    assert deduplicated == ["checkpoints/L5PC/test/emodel=L5PC__seed=0.pkl"]


def test_deduplicate_checkpoint_paths_keeps_single_format():
    paths = ["checkpoints/L5PC/test/emodel=L5PC__seed=1.h5"]

    assert deduplicate_checkpoint_paths(paths) == paths


def test_existing_checkpoint_paths_deduplicates_h5_and_pkl(workspace):
    metadata = EModelMetadata(
        emodel="L5PC",
        mtype="L5TPC:A",
        ttype="t type",
        iteration_tag="test",
        brain_region="somatosensory cortex",
        allen_notation="SSCX",
    )
    checkpoint_dir = workspace / "checkpoints" / "L5PC" / "test"
    checkpoint_dir.mkdir(parents=True)
    stem = "emodel=L5PC__seed=0"
    (checkpoint_dir / f"{stem}.h5").touch()
    (checkpoint_dir / f"{stem}.pkl").touch()

    paths = existing_checkpoint_paths(
        metadata,
        checkpoint_paths=[
            str(checkpoint_dir / f"{stem}.h5"),
            str(checkpoint_dir / f"{stem}.pkl"),
        ],
    )

    assert paths == [str(checkpoint_dir / f"{stem}.pkl")]


def test_existing_checkpoint_paths_finds_h5_and_pkl_in_checkpoints_dir(workspace):
    metadata = EModelMetadata(
        emodel="L5PC",
        mtype="L5TPC:A",
        ttype="t type",
        iteration_tag="test",
        brain_region="somatosensory cortex",
        allen_notation="SSCX",
    )
    checkpoint_dir = workspace / "checkpoints" / "L5PC" / "test"
    checkpoint_dir.mkdir(parents=True)
    stem = "emodel=L5PC__seed=1"
    (checkpoint_dir / f"{stem}.h5").touch()
    (checkpoint_dir / f"{stem}.pkl").touch()

    paths = existing_checkpoint_paths(metadata)

    assert paths == [f"./checkpoints/L5PC/test/{stem}.pkl"]


def test_read_checkpoint_uses_hdf5_reader(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.h5"
    checkpoint_path.touch()
    expected = ({"generation": 1}, 7)
    read_h5 = MagicMock(return_value=expected)
    monkeypatch.setattr("bluepyemodel.tools.utils.read_checkpoint_h5", read_h5)

    result = read_checkpoint(checkpoint_path)

    read_h5.assert_called_once_with(str(checkpoint_path))
    assert result == expected


def test_read_checkpoint_reads_pickle(tmp_path):
    checkpoint_path = tmp_path / "emodel=L5PC__seed=3.pkl"
    payload = {"generation": 2}
    with open(checkpoint_path, "wb") as checkpoint_file:
        pickle.dump(payload, checkpoint_file)

    run, seed = read_checkpoint(checkpoint_path)

    assert run == payload
    assert seed == 3
