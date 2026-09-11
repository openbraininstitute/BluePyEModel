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

import os

import pytest

from bluepyemodel.tools.mechanisms import compile_mechs
from bluepyemodel.tools.mechanisms import compile_mechs_in_emodel_dir
from bluepyemodel.tools.mechanisms import copy_and_compile_mechanisms
from bluepyemodel.tools.mechanisms import copy_mechs
from bluepyemodel.tools.mechanisms import delete_compiled_mechanisms
from bluepyemodel.tools.mechanisms import discriminate_by_temp
from bluepyemodel.tools.mechanisms import get_mechanism_currents
from bluepyemodel.tools.mechanisms import get_mechanism_name
from bluepyemodel.tools.utils import are_same_protocol
from bluepyemodel.tools.utils import format_protocol_name_to_list
from bluepyemodel.tools.utils import get_curr_name
from bluepyemodel.tools.utils import get_loc_name
from bluepyemodel.tools.utils import get_protocol_name
from bluepyemodel.tools.utils import select_rec_for_thumbnail
from tests.utils import DATA


def test_get_mechanism_currents():
    # writes ion current
    ion_currs, nonspec_currs, ionic_concentrations = get_mechanism_currents(
        DATA / "mechanisms" / "Ca_HVA2.mod"
    )
    assert ion_currs == ["ica"]
    assert nonspec_currs == []
    assert ionic_concentrations == ["cai"]
    # writes ionic concentration
    ion_currs, nonspec_currs, ionic_concentrations = get_mechanism_currents(
        DATA / "mechanisms" / "CaDynamics_DC0.mod"
    )
    assert ion_currs == []
    assert nonspec_currs == []
    assert ionic_concentrations == ["cai"]
    # writes non-specific current
    ion_currs, nonspec_currs, ionic_concentrations = get_mechanism_currents(
        DATA / "mechanisms" / "Ih.mod"
    )
    assert ion_currs == []
    assert nonspec_currs == ["ihcn"]
    assert ionic_concentrations == []


def test_format_protocol_name_to_list():
    # str case
    name, amp = format_protocol_name_to_list("APWaveform_140")
    assert name == "APWaveform"
    assert amp == 140.0

    name, amp = format_protocol_name_to_list("APWaveform_140.0")
    assert name == "APWaveform"
    assert amp == 140.0

    name, amp = format_protocol_name_to_list("APWaveform")
    assert name == "APWaveform"
    assert amp is None

    # list case
    name, amp = format_protocol_name_to_list(["APWaveform", 140])
    assert name == "APWaveform"
    assert amp == 140.0

    # error case
    with pytest.raises(TypeError, match="protocol_name should be a string or a list."):
        format_protocol_name_to_list(None)


def test_are_same_protocol():
    # None case
    assert not are_same_protocol("APWaveform_140", None)
    assert not are_same_protocol(None, "APWaveform_140")

    # str case
    assert not are_same_protocol("APWaveform_140", "IDRest_100")
    assert not are_same_protocol("APWaveform_140", "APWaveform_120")
    assert not are_same_protocol("APWaveform_140", "IDRest_140")
    assert are_same_protocol("APWaveform_140", "APWaveform_140")
    assert are_same_protocol("APWaveform_140", "APWaveform_140.0")

    # list case
    assert not are_same_protocol(["APWaveform", 140], ["APWaveform", 120])
    assert not are_same_protocol(["APWaveform", 140], ["IDRest", 140])
    assert are_same_protocol(["APWaveform", 140], ["APWaveform", 140])
    assert are_same_protocol(["APWaveform", 140], ["APWaveform", 140.0])

    # mixed case
    assert not are_same_protocol("APWaveform_140", ["APWaveform", 120])
    assert are_same_protocol("APWaveform_140", ["APWaveform", 140])
    assert are_same_protocol("APWaveform_140", ["APWaveform", 140.0])
    assert are_same_protocol("APWaveform_140.0", ["APWaveform", 140])


def test_select_rec_for_thumbnail():
    """Test for select_rec_for_thumbnail function."""
    # normal case: select step protocol with lowest positive amplitude
    rec_names = [
        "IDrest_200.soma.v",
        "IDrest_-40.soma.v",
        "IDrest_130.soma.v",
        "sAHP_40.soma.v",
    ]
    assert select_rec_for_thumbnail(rec_names) == "IDrest_130.soma.v"

    # empty recordings case
    with pytest.raises(
        ValueError, match="No recording in recording_names. Can not plot thumbnail."
    ):
        select_rec_for_thumbnail([])

    # no step protocols case: return 1st protocol
    assert select_rec_for_thumbnail(["sAHP_40.soma.v", "sAHP_20.soma.v"]) == "sAHP_40.soma.v"

    # step protocol present but not positive case: return 1st protocol
    assert select_rec_for_thumbnail(["sAHP_40.soma.v", "IDrest_-20.soma.v"]) == "sAHP_40.soma.v"

    # additional step protocol names case
    rec_names = ["IDrest_200.soma.v", "MyStep_100.soma.v"]
    other_step_prot_names = ["MyStep", "Test"]
    assert (
        select_rec_for_thumbnail(rec_names, additional_step_prots=other_step_prot_names)
        == "MyStep_100.soma.v"
    )

    # thumbnail_rec case
    rec_names = ["IDrest_130.soma.v", "sAHP_40.soma.v"]
    assert select_rec_for_thumbnail(rec_names, thumbnail_rec="sAHP_40.soma.v") == "sAHP_40.soma.v"

    # thumbnail_rec not in rec_names:
    rec_names = ["sAHP_40.soma.v", "IDrest_130.soma.v"]
    assert (
        select_rec_for_thumbnail(rec_names, thumbnail_rec="sAHP_20.soma.v") == "IDrest_130.soma.v"
    )

    # absolute amplitude with float amp case
    rec_names = [
        "IDrest_0.2.soma.v",
        "IDrest_-0.04.soma.v",
        "IDrest_0.13.soma.v",
        "sAHP_0.04.soma.v",
    ]
    assert select_rec_for_thumbnail(rec_names) == "IDrest_0.13.soma.v"


def test_get_protocol_name():
    # feature keys
    feature_name = "IV_40.0.soma.v.voltage_base"
    assert get_protocol_name(feature_name) == "IV_40.0"

    feature_name = "IV_40.soma.v.voltage_base"
    assert get_protocol_name(feature_name) == "IV_40"

    feature_name = "ProtocolA.1.soma.v.some_feature"
    assert get_protocol_name(feature_name) == "ProtocolA.1"

    # response keys
    feature_name = "IV_40.0.soma.v"
    assert get_protocol_name(feature_name) == "IV_40.0"

    feature_name = "IV_40.soma.v"
    assert get_protocol_name(feature_name) == "IV_40"

    feature_name = "ProtocolA.1.soma.v"
    assert get_protocol_name(feature_name) == "ProtocolA.1"


def test_get_loc_name():
    # feature keys
    feature_name = "IV_40.0.soma.v.voltage_base"
    assert get_loc_name(feature_name) == "soma"

    feature_name = "IV_40.soma.v.voltage_base"
    assert get_loc_name(feature_name) == "soma"

    feature_name = "IV_40.0"
    with pytest.raises(IndexError, match="Location name not found in the feature name."):
        get_loc_name(feature_name)

    # response keys
    feature_name = "IV_40.0.soma.v"
    assert get_loc_name(feature_name) == "soma"

    feature_name = "IV_40.soma.v"
    assert get_loc_name(feature_name) == "soma"

    feature_name = "ProtocolA.1.soma.v"
    assert get_loc_name(feature_name) == "soma"


def test_get_curr_name():
    # feature keys
    feature_name = "IV_40.0.soma.v.voltage_base"
    assert get_curr_name(feature_name) == "v"

    feature_name = "IV_40.soma.v.voltage_base"
    assert get_curr_name(feature_name) == "v"

    feature_name = "IV_40.0.soma"
    with pytest.raises(IndexError, match="Current name not found in the feature name."):
        get_curr_name(feature_name)

    # response keys
    feature_name = "IV_40.0.soma.v"
    assert get_curr_name(feature_name) == "v"

    feature_name = "IV_40.soma.v"
    assert get_curr_name(feature_name) == "v"

    feature_name = "ProtocolA.1.soma.v"
    assert get_curr_name(feature_name) == "v"


# ---------------------------------------------------------------------------
# bluepyemodel.tools.mechanisms
# ---------------------------------------------------------------------------


def test_copy_mechs(tmp_path):
    src_mech = DATA / "mechanisms" / "Ih.mod"
    out_dir = tmp_path / "mechanisms"

    copy_mechs([{"path": str(src_mech)}], out_dir)

    assert (out_dir / "Ih.mod").is_file()


def test_copy_mechs_empty_list_is_noop(tmp_path):
    out_dir = tmp_path / "mechanisms"

    copy_mechs([], out_dir)

    assert not out_dir.exists()


def test_copy_mechs_raises_on_missing_file(tmp_path):
    out_dir = tmp_path / "mechanisms"

    with pytest.raises(FileNotFoundError):
        copy_mechs([{"path": str(tmp_path / "does_not_exist.mod")}], out_dir)


def test_delete_compiled_mechanisms(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "x86_64").mkdir()

    delete_compiled_mechanisms()

    assert not (tmp_path / "x86_64").is_dir()


def test_delete_compiled_mechanisms_noop_when_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Should not raise even if x86_64 does not exist.
    delete_compiled_mechanisms()


def test_compile_mechs_raises_on_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        compile_mechs(tmp_path / "does_not_exist")


def test_compile_mechs_in_emodel_dir_logs_exception_and_restores_cwd(tmp_path, caplog):
    original_cwd = os.getcwd()
    # The parent directory exists, but it contains no "mechanisms" subdirectory,
    # so compile_mechs (called with "./mechanisms") raises FileNotFoundError.
    mechanisms_directory = tmp_path / "mechanisms"

    compile_mechs_in_emodel_dir(mechanisms_directory)

    # cwd must be restored even though compile_mechs raised internally.
    assert os.getcwd() == original_cwd
    assert "Cannot compile the mechanisms" in caplog.text


def test_copy_and_compile_mechanisms_noop_for_local_access_point():
    class FakeLocalAccessPoint:
        """Minimal stand-in with the class name LocalAccessPoint."""

    access_point = FakeLocalAccessPoint()

    # Should not raise, and should not attempt to compile anything since
    # the class name is not NexusAccessPoint.
    copy_and_compile_mechanisms(access_point)


def test_get_mechanism_name_suffix():
    mech_file = DATA / "mechanisms" / "Ih.mod"
    assert get_mechanism_name(mech_file) == "Ih"


def test_get_mechanism_name_point_process(tmp_path):
    mech_file = tmp_path / "MyPointProcess.mod"
    mech_file.write_text(
        "NEURON {\n    POINT_PROCESS MyPointProcess\n}\n",
    )
    assert get_mechanism_name(mech_file) == "MyPointProcess"


def test_get_mechanism_name_raises_when_not_found(tmp_path):
    mech_file = tmp_path / "empty.mod"
    mech_file.write_text("NEURON {\n}\n")

    with pytest.raises(RuntimeError, match="Could not find SUFFIX nor POINT_PROCESS"):
        get_mechanism_name(mech_file)


class _FakeTemperature:
    def __init__(self, value):
        self.value = value


class _FakeResource:
    def __init__(self, temperature=None):
        if temperature is not None:
            self.temperature = _FakeTemperature(temperature)


def test_discriminate_by_temp_no_temperatures_returns_all():
    resources = [_FakeResource(34), _FakeResource(37)]
    assert discriminate_by_temp(resources, None) == resources
    assert discriminate_by_temp(resources, []) == resources


def test_discriminate_by_temp_filters_matching_resources():
    r34 = _FakeResource(34)
    r37 = _FakeResource(37)
    resources = [r34, r37]

    result = discriminate_by_temp(resources, [34])

    assert result == [r34]


def test_discriminate_by_temp_falls_back_to_next_temperature():
    r37 = _FakeResource(37)
    resources = [r37]

    # No resource matches 34, so it should recurse and match 37.
    result = discriminate_by_temp(resources, [34, 37])

    assert result == [r37]


def test_discriminate_by_temp_returns_all_when_none_match_any_temperature():
    resources = [_FakeResource(20)]

    result = discriminate_by_temp(resources, [34, 37])

    assert result == resources
