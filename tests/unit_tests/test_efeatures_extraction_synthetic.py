"""Synthetic coverage for pure efeature-extraction helpers."""

import pickle
from types import SimpleNamespace

import pytest

from bluepyemodel.efeatures_extraction import efeatures_extraction as extraction


def feature(name, protocol, mean, original_std=0.1, settings=None):
    return SimpleNamespace(
        name=f"{protocol}.{name}",
        efel_feature_name=name,
        protocol_name=protocol,
        mean=mean,
        original_std=original_std,
        efel_settings={} if settings is None else settings,
    )


def test_extraction_output_directory_and_pickle(tmp_path):
    output = extraction.get_extraction_output_directory("model")
    assert str(output) == "figures/model/efeatures_extraction"
    missing = extraction.read_extraction_output(tmp_path / "missing.pkl")
    assert missing is None
    filepath = tmp_path / "cells.pkl"
    filepath.write_bytes(pickle.dumps({"cells": 1}))
    assert extraction.read_extraction_output(filepath) == {"cells": 1}


def test_interpolate_rmp_updates_rmp_feature():
    config = SimpleNamespace(
        efeatures=[
            feature("ohmic_input_resistance_vb_ssse", "RinProtocol", 100),
            feature("voltage_base", "IV_-40", -70),
            feature("voltage_base", "IDrest_100", -68),
            feature("bpo_holding_current", "RMPProtocol", 0.1),
            feature("steady_state_voltage_stimend", "RMPProtocol", 0),
        ]
    )

    extraction.interpolate_RMP(config)

    assert config.efeatures[-1].mean == pytest.approx(-79.0)


@pytest.mark.parametrize(
    "efeatures, error",
    [
        ([feature("bpo_holding_current", "RMP", 0.1)], TypeError),
        ([feature("ohmic_input_resistance_vb_ssse", "RinProtocol", 100)], TypeError),
        (
            [
                feature("ohmic_input_resistance_vb_ssse", "RinProtocol", 100),
                feature("bpo_holding_current", "RMP", 0.1),
            ],
            ValueError,
        ),
    ],
)
def test_interpolate_rmp_reports_missing_inputs(efeatures, error):
    with pytest.raises(error):
        extraction.interpolate_RMP(SimpleNamespace(efeatures=efeatures))


def test_threshold_efeatures_std_only_changes_nondefault_large_std():
    changed = feature("a", "Step", 2.0, original_std=4.0)
    exempt = feature("b", "Step", 2.0, original_std=5.0)
    small = feature("c", "Step", 2.0, original_std=1.0)
    extraction.threshold_efeatures_std(
        SimpleNamespace(efeatures=[changed, exempt, small]), 5.0
    )

    assert changed.original_std == 2.0
    assert exempt.original_std == 5.0
    assert small.original_std == 1.0


def test_define_extraction_reader_function_variants(tmp_path):
    access_point = SimpleNamespace(
        pipeline_settings=SimpleNamespace(extraction_reader=None)
    )
    assert extraction.define_extraction_reader_function(access_point) is None
    reader = lambda _: None
    access_point.pipeline_settings.extraction_reader = reader
    assert extraction.define_extraction_reader_function(access_point) is reader
    module_path = tmp_path / "reader_module.py"
    module_path.write_text("def read(value):\n    return value\n")
    access_point.pipeline_settings.extraction_reader = [str(module_path), "read"]
    assert extraction.define_extraction_reader_function(access_point)(3) == 3
    access_point.pipeline_settings.extraction_reader = "invalid"
    with pytest.raises(TypeError):
        extraction.define_extraction_reader_function(access_point)


def test_attach_pdfs_and_update_minimum_delay(monkeypatch):
    efeatures = {"Step": {"soma": [{"feature": "Spikecount"}]}}
    monkeypatch.setattr(
        extraction,
        "search_figure_efeatures",
        lambda *args: ("absolute.pdf", "relative.pdf"),
    )
    extraction.attach_efeatures_pdf("model", efeatures)
    assert efeatures["Step"]["soma"][0]["pdfs"] == {
        "amp": "absolute.pdf",
        "amp_rel": "relative.pdf",
    }

    protocol = SimpleNamespace(
        name="Step",
        stimuli=[{"delay": 1.0, "totduration": 10.0}],
    )
    target = feature(
        "Spikecount", "Step", 1, settings={"stim_start": 1.0, "stim_end": 5.0}
    )
    config = SimpleNamespace(protocols=[protocol], efeatures=[target])
    access_point = SimpleNamespace(
        pipeline_settings=SimpleNamespace(minimum_protocol_delay=3.0)
    )
    extraction.update_minimum_protocols_delay(access_point, config)

    assert protocol.stimuli[0] == {"delay": 3.0, "totduration": 12.0}
    assert target.efel_settings == {"stim_start": 3.0, "stim_end": 7.0}


def test_update_minimum_delay_ignores_later_delay_and_unrelated_features():
    protocol = SimpleNamespace(
        name="Step", stimuli=[{"delay": 4.0, "totduration": 10.0}]
    )
    other = SimpleNamespace(name="Other", stimuli=[{"amp": 1}])
    feature_other = feature("Spikecount", "Other", 1, settings={})
    config = SimpleNamespace(protocols=[protocol, other], efeatures=[feature_other])
    access_point = SimpleNamespace(
        pipeline_settings=SimpleNamespace(minimum_protocol_delay=3.0)
    )

    assert extraction.update_minimum_protocols_delay(access_point, config) is config
    assert protocol.stimuli[0]["delay"] == 4.0
    assert other.stimuli[0] == {"amp": 1}
    assert feature_other.efel_settings == {}
