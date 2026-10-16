"""Deterministic tests for plotting utility helpers."""

import matplotlib
import numpy
import pytest

matplotlib.use("Agg")

from types import SimpleNamespace

from bluepyemodel.emodel_pipeline import plotting_utils as pu


def test_get_traces_ylabel_variants():
    assert pu.get_traces_ylabel("v") == "Voltage (mV)"
    assert pu.get_traces_ylabel("ina") == "Current (pA)"
    assert pu.get_traces_ylabel("cai") == "Ionic concentration (mM)"
    assert pu.get_traces_ylabel("xyz") == ""


def test_get_recording_names_adds_pre_protocol_recordings():
    protocol_config = [
        SimpleNamespace(
            recordings_from_config=[
                {"name": "Step_150.soma.v"},
                {"name": "Step_150.dend.v"},
            ]
        )
    ]
    stimuli = {
        "Step_150": SimpleNamespace(
            name="Step_150", recordings=[SimpleNamespace(name="Step_150.soma.v")]
        ),
        "RMPProtocol": SimpleNamespace(
            name="RMPProtocol", recordings=[SimpleNamespace(name="RMPProtocol.soma.v")]
        ),
        "NoRecording": SimpleNamespace(name="NoRecording", recordings=[]),
    }

    names = pu.get_recording_names(protocol_config, stimuli)

    assert names == {"Step_150.soma.v", "Step_150.dend.v", "RMPProtocol.soma.v"}


def test_get_traces_names_and_float_responses_splits_values():
    responses = {
        "Step_150.soma.v": SimpleNamespace(),
        "Ignored.soma.v": SimpleNamespace(),
        "bpo_threshold_current": 0.2,
        "bpo_holding_current": -0.05,
        "bpo_rmp": -75.0,
        "bpo_rin": 150.0,
        "bpo_other": 1.0,
    }

    traces, threshold, holding, rmp, rin = pu.get_traces_names_and_float_responses(
        responses, {"Step_150.soma.v"}
    )

    assert traces == ["Step_150.soma.v"]
    assert threshold == pytest.approx(0.2)
    assert holding == pytest.approx(-0.05)
    assert rmp == pytest.approx(-75.0)
    assert rin == pytest.approx(150.0)


def test_get_title_variants():
    assert pu.get_title("L5PC", None, None) == "L5PC"
    assert pu.get_title("L5PC", "iter1", 3) == "L5PC ; iteration = iter1 ; seed = 3"
    assert pu.get_title("L5PC", None, 3) == "L5PC ; seed = 3"


def test_rel_to_abs_amplitude_and_missing_responses():
    responses = {"bpo_threshold_current": 0.2, "bpo_holding_current": -0.05}

    assert pu.rel_to_abs_amplitude(200.0, responses) == pytest.approx(0.35)
    assert numpy.isnan(pu.rel_to_abs_amplitude(200.0, {}))


def test_binning_returns_input_when_below_bin_count():
    x, y, err = pu.binning([1.0, 2.0], [3.0, 4.0], n_bin=5)

    assert x == [1.0, 2.0]
    assert y == [3.0, 4.0]
    assert err == [0.0, 0.0]


def test_binning_averages_within_bins():
    x = list(range(10))
    y = [float(v) for v in range(10)]

    new_x, new_y, y_err = pu.binning(x, y, n_bin=2)

    assert len(new_x) == 2
    assert new_x[0] == pytest.approx(2.25)
    assert new_y[0] == pytest.approx(2.0)
    assert all(not numpy.isnan(v) for v in y_err)


def test_get_ordered_currentscape_keys_groups_and_skips():
    keys = [
        "Step_150.soma.v",
        "Step_150.soma.ina",
        "Step_150.soma.cai",
        "Step_150.dend.v",
        "RMPProtocol.soma.v",
        "SearchHoldingCurrent.soma.v",
        "bpo_threshold_current",
    ]

    ordered = pu.get_ordered_currentscape_keys(keys)

    assert set(ordered) == {"Step_150"}
    soma = ordered["Step_150"]["soma"]
    assert soma["voltage_key"] == "Step_150.soma.v"
    assert soma["current_keys"] == ["Step_150.soma.ina"]
    assert soma["current_names"] == ["ina"]
    assert soma["ion_conc_keys"] == ["Step_150.soma.cai"]
    assert soma["ion_conc_names"] == ["cai"]
    assert ordered["Step_150"]["dend"]["voltage_key"] == "Step_150.dend.v"


def test_get_voltage_currents_from_files(tmp_path):
    numpy.savetxt(tmp_path / "Step.soma.v.dat", [[0.0, -80.0], [1.0, -70.0]])
    numpy.savetxt(tmp_path / "Step.soma.ina.dat", [[0.0, -0.1], [1.0, -0.2]])
    numpy.savetxt(tmp_path / "Step.soma.cai.dat", [[0.0, 0.001], [1.0, 0.002]])
    key_dict = {
        "voltage_key": "Step.soma.v",
        "current_keys": ["Step.soma.ina"],
        "ion_conc_keys": ["Step.soma.cai"],
    }

    time, voltage, currents, ion_concentrations = pu.get_voltage_currents_from_files(
        key_dict, tmp_path
    )

    numpy.testing.assert_allclose(time, [0.0, 1.0])
    numpy.testing.assert_allclose(voltage, [-80.0, -70.0])
    numpy.testing.assert_allclose(currents[0], [-0.1, -0.2])
    numpy.testing.assert_allclose(ion_concentrations[0], [0.001, 0.002])


def test_get_original_protocol_name_matches_case_insensitively():
    evaluator = SimpleNamespace(
        fitness_protocols={
            "main_protocol": SimpleNamespace(protocols={"IDrest_100": object()})
        }
    )

    assert pu.get_original_protocol_name("idrest_100", evaluator) == "IDrest_100"
    assert pu.get_original_protocol_name("absent", evaluator) == "absent"


def test_find_matching_feature_by_recording_name():
    target = SimpleNamespace(
        recording_names={"": "IV_-40.soma.v"},
        efel_feature_name="voltage_deflection_vb_ssse",
    )
    other = SimpleNamespace(
        recording_names={"": "Step_150.soma.v"}, efel_feature_name="mean_frequency"
    )
    evaluator = SimpleNamespace(
        fitness_calculator=SimpleNamespace(
            objectives=[
                SimpleNamespace(features=[other]),
                SimpleNamespace(features=[target]),
            ]
        )
    )

    feature, exact_match = pu.find_matching_feature(evaluator, "IV_-40")

    assert feature is target
    assert exact_match is True


def test_find_matching_feature_falls_back_to_relaxed_condition():
    relaxed = SimpleNamespace(
        recording_names={"": "IV_-40.soma.v"},
        efel_feature_name="voltage_deflection_vb_ssse",
        stimulus_current=lambda: 0.1,
    )
    evaluator = SimpleNamespace(
        fitness_calculator=SimpleNamespace(
            objectives=[SimpleNamespace(features=[relaxed])]
        )
    )

    feature, exact_match = pu.find_matching_feature(evaluator, "IV_-40.0")

    assert feature is relaxed
    assert exact_match is False


def test_find_matching_feature_returns_none_when_absent():
    evaluator = SimpleNamespace(fitness_calculator=SimpleNamespace(objectives=[]))

    assert pu.find_matching_feature(evaluator, "IV_-40") == (None, False)


def test_get_experimental_FI_curve_for_plotting_bins_data():
    recordings = [
        SimpleNamespace(
            protocol_name="idrest",
            amp_rel=100.0,
            amp=0.2,
            efeatures={"mean_frequency": 10.0},
        ),
        SimpleNamespace(
            protocol_name="idrest",
            amp_rel=200.0,
            amp=0.4,
            efeatures={"mean_frequency": 20.0},
        ),
        SimpleNamespace(
            protocol_name="other",
            amp_rel=100.0,
            amp=0.2,
            efeatures={"mean_frequency": 99.0},
        ),
    ]
    cells = [SimpleNamespace(recordings=recordings)]

    result = pu.get_experimental_FI_curve_for_plotting(cells, "idrest")

    amp_rel, freq_rel, _, amp, freq_abs, _ = result
    assert list(amp_rel) == [100.0, 200.0]
    assert list(freq_rel) == [10.0, 20.0]
    assert list(amp) == [0.2, 0.4]
    assert list(freq_abs) == [10.0, 20.0]


def test_get_simulated_FI_curve_for_plotting_uses_relative_amplitudes():
    values = {"IDrest_100.soma.v.mean_frequency": [12.0]}
    evaluator = SimpleNamespace(
        fitness_calculator=SimpleNamespace(calculate_values=lambda responses: values)
    )
    responses = {"bpo_threshold_current": 0.2, "bpo_holding_current": -0.05}

    amp_rel, amp, freq = pu.get_simulated_FI_curve_for_plotting(
        evaluator, responses, "IDrest"
    )

    assert amp_rel == [100.0]
    assert amp == [pytest.approx(0.15)]
    numpy.testing.assert_allclose(freq, [12.0])


def test_get_simulated_FI_curve_for_plotting_without_threshold_current():
    values = {
        "IDrest_100.soma.v.mean_frequency": None,
        "IDrest_200.soma.v.mean_frequency": [20.0],
    }
    evaluator = SimpleNamespace(
        fitness_calculator=SimpleNamespace(calculate_values=lambda responses: values)
    )

    amp_rel, amp, freq = pu.get_simulated_FI_curve_for_plotting(evaluator, {}, "IDrest")

    assert all(numpy.isnan(v) for v in amp_rel)
    assert amp == [100.0, 200.0]
    assert numpy.isnan(freq[0])
    assert freq[1] == pytest.approx(20.0)


def test_save_fig_writes_file(tmp_path):
    import matplotlib.pyplot as plt

    plt.plot([0, 1], [0, 1])

    pu.save_fig(tmp_path, "figure.pdf")

    assert (tmp_path / "figure.pdf").is_file()


def test_plot_fi_curves_writes_comparison(tmp_path):
    expt_data = ([100.0], [10.0], [1.0], [0.2], [10.0], [1.0])
    sim_data = ([100.0], [0.2], [11.0])
    emodel = SimpleNamespace(
        seed=1, emodel_metadata=SimpleNamespace(as_string=lambda seed: f"model__{seed}")
    )

    pu.plot_fi_curves(expt_data, sim_data, tmp_path, emodel, write_fig=True)

    assert (tmp_path / "model__1__FI_curve_comparison.pdf").is_file()


def test_extract_experimental_data_for_IV_curve_computes_missing_features():
    computed = []

    class Recording:
        def __init__(self, amp_rel, amp, efeatures):
            self.protocol_name = "iv"
            self.amp_rel = amp_rel
            self.amp = amp
            self.efeatures = efeatures

        def compute_efeatures(self, names, efel_settings=None):
            computed.append(names[0])
            self.efeatures[names[0]] = 1.0

    positive = Recording(50.0, 0.1, {})
    negative = Recording(-50.0, -0.1, {})
    ignored = Recording(150.0, 0.3, {})
    cells = [SimpleNamespace(recordings=[positive, negative, ignored])]

    peak_data, deflection_data = pu.extract_experimental_data_for_IV_curve(
        cells, {}, prot_name="iv"
    )

    assert "maximum_voltage_from_voltagebase" in computed
    assert "voltage_deflection_vb_ssse" in computed
    assert peak_data["amp_rel"] == [50.0]
    assert peak_data["amp"] == [pytest.approx(0.1)]
    assert deflection_data["amp_rel"] == [-50.0]
    assert deflection_data["amp"] == [pytest.approx(-0.1)]


def test_get_impedance_uses_efel_fallback_features(monkeypatch):
    monkeypatch.setattr(pu.efel, "reset", lambda: None)
    monkeypatch.setattr(pu.efel, "set_setting", lambda name, value: None)
    monkeypatch.setattr(
        pu.efel,
        "get_feature_values",
        lambda traces, features: [
            {
                "voltage_base": None,
                "steady_state_voltage_stimend": [-70.0],
                "current_base": None,
                "steady_state_current_stimend": [0.0],
            }
        ],
    )

    freq, impedance = pu.get_impedance(
        numpy.arange(0.0, 100.0, 0.1),
        numpy.sin(numpy.arange(0.0, 100.0, 0.1)),
        numpy.cos(numpy.arange(0.0, 100.0, 0.1)),
        10.0,
        90.0,
        {"interp_step": 0.1},
    )

    assert len(freq) == len(impedance)


def test_get_impedance_returns_none_when_efel_features_missing(monkeypatch):
    monkeypatch.setattr(pu.efel, "reset", lambda: None)
    monkeypatch.setattr(pu.efel, "set_setting", lambda name, value: None)
    monkeypatch.setattr(
        pu.efel,
        "get_feature_values",
        lambda traces, features: [
            {
                "voltage_base": None,
                "steady_state_voltage_stimend": None,
                "current_base": [0.0],
                "steady_state_current_stimend": None,
            }
        ],
    )

    assert pu.get_impedance([0.0, 1.0], [1.0, 1.0], [0.0, 0.0], 0.0, 1.0, {}) == (
        None,
        None,
    )


def test_get_sinespec_evaluator_supports_absolute_amplitude():
    main_protocol = SimpleNamespace(
        protocols={"IDrest_100": object(), "RMPProtocol": object()},
        compute_execution_order=lambda: ["SineSpec_0.1"],
    )
    evaluator = SimpleNamespace(
        fitness_protocols={"main_protocol": main_protocol},
        fitness_calculator=SimpleNamespace(objectives=[]),
    )

    updated = pu.get_sinespec_evaluator(
        evaluator, {"amp": 0.1, "threshold_based": False}, {"Threshold": -20.0}
    )

    assert set(updated.fitness_protocols["main_protocol"].protocols) == {"SineSpec_0.1"}
    assert (
        updated.fitness_calculator.objectives[0].features[0].efel_feature_name
        == "impedance"
    )


def test_update_evaluator_adds_missing_fi_protocol_and_feature():
    protocol = SimpleNamespace(
        stimuli=[
            SimpleNamespace(
                holding_current=-0.05,
                threshold_current=0.2,
                delay=250.0,
                duration=1350.0,
                total_duration=1850.0,
            )
        ],
        recordings=[],
    )
    feature = SimpleNamespace(
        name="IDrest_100.soma.v.mean_frequency",
        recording_names={"": "IDrest_100.soma.v"},
        efel_feature_name="mean_frequency",
        stim_start=250.0,
        stim_end=1600.0,
        threshold=-20.0,
        stimulus_current=lambda: 0.2,
    )
    objective = SimpleNamespace(features=[feature])
    evaluator = SimpleNamespace(
        fitness_protocols={
            "main_protocol": SimpleNamespace(protocols={"IDrest_100": protocol})
        },
        fitness_calculator=SimpleNamespace(objectives=[objective]),
    )

    updated = pu.update_evaluator([200.0], "IDrest_100", evaluator)

    assert "IDrest_200" in updated.fitness_protocols["main_protocol"].protocols
    assert any(
        "IDrest_200" in obj.name
        for obj in updated.fitness_calculator.objectives
        if hasattr(obj, "name")
    )


def test_create_protocol_scales_amplitude_and_preserves_timing():
    source = SimpleNamespace(
        stimuli=[
            SimpleNamespace(
                holding_current=-0.05,
                threshold_current=0.2,
                delay=250.0,
                duration=1350.0,
                total_duration=1850.0,
            )
        ],
        recordings=[
            SimpleNamespace(
                name="IDrest_100.soma.v", location=pu.soma_loc, variable="v"
            )
        ],
        cvode_active=True,
        stochasticity=False,
    )
    feature = SimpleNamespace(stimulus_current=lambda: 0.2)

    absolute = pu.create_protocol(150, 0.1, feature, source, "IDrest_150")
    relative = pu.create_protocol(150, None, None, source, "IDrest_150_relative")

    assert absolute.stimulus.amplitude == pytest.approx(300.0)
    assert relative.stimulus.amplitude == pytest.approx(150.0)
    assert absolute.stimulus.stim_start == pytest.approx(250.0)
    assert absolute.recordings[0].name == "IDrest_150.soma.v"


def test_extract_experimental_data_for_iv_curve_covers_both_feature_signs():
    """Positive amplitudes use the peak feature, negative ones the deflection."""
    positive = SimpleNamespace(
        protocol_name="IV",
        amp_rel=50,
        amp="0.1",
        efeatures={"maximum_voltage_from_voltagebase": -65.0},
    )
    negative = SimpleNamespace(
        protocol_name="iv", amp_rel=-50, amp="-0.1", efeatures={}
    )

    def compute_efeatures(names, efel_settings):
        assert names == ["voltage_deflection_vb_ssse"]
        assert efel_settings == {}
        negative.efeatures[names[0]] = -15.0

    negative.compute_efeatures = compute_efeatures
    ignored = SimpleNamespace(
        protocol_name="Other",
        amp_rel=50,
        amp="0.1",
        efeatures={"maximum_voltage_from_voltagebase": 1},
    )

    peak, deflection = pu.extract_experimental_data_for_IV_curve(
        [SimpleNamespace(recordings=[positive, negative, ignored])], {}, n_bin=5
    )

    assert peak == {
        "amp_rel": [50.0],
        "amp": [0.1],
        "feat_rel": [-65.0],
        "feat_rel_err": [0.0],
        "feat_abs": [-65.0],
        "feat_abs_err": [0.0],
    }
    assert deflection == {
        "amp_rel": [-50.0],
        "amp": [-0.1],
        "feat_rel": [-15.0],
        "feat_rel_err": [0.0],
        "feat_abs": [-15.0],
        "feat_abs_err": [0.0],
    }
