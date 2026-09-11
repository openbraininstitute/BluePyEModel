"""Tests for morphology modifier helpers."""

import numpy

from bluepyemodel.evaluation.modifiers import (
    ZERO,
    get_synth_axon_hoc,
    isolate_axon,
    isolate_soma,
    remove_axon,
    taper_function,
)


def test_taper_function():
    numpy.testing.assert_allclose(
        taper_function(10.0, strength=2.0, taper_scale=10.0, terminal_diameter=0.5),
        2.0 * numpy.exp(-1.0) + 0.5,
    )


def test_get_synth_axon_hoc_includes_parameters():
    hoc = get_synth_axon_hoc([60.0, 2.0, 10.0, 0.5])

    assert "L_target = 60.0" in hoc
    assert "strength = 2.0" in hoc
    assert "taper_scale = 10.0" in hoc
    assert "terminal_diameter = 0.5" in hoc


class Section:
    def __init__(self):
        self.diam = 1.0


class H:
    def __init__(self):
        self.deleted = []

    def delete_section(self, sec):
        self.deleted.append(sec)


def test_remove_axon_sets_diameter_and_deletes_myelin():
    h = H()
    myelin = Section()
    axon = Section()
    icell = type("Cell", (), {"myelin": [myelin], "axonal": [axon]})()
    sim = type("Sim", (), {"neuron": type("Neuron", (), {"h": h})()})()

    remove_axon(sim=sim, icell=icell)

    assert h.deleted == [myelin]
    assert axon.diam == ZERO


def test_isolate_soma_deletes_non_somatic_sections():
    h = H()
    axon, basal, apical = Section(), Section(), Section()
    icell = type(
        "Cell",
        (),
        {"axonal": [axon], "basal": [basal], "apical": [apical], "soma": []},
    )()
    sim = type("Sim", (), {"neuron": type("Neuron", (), {"h": h})()})()

    isolate_soma(sim=sim, icell=icell)

    assert h.deleted == [axon, basal, apical]


def test_isolate_axon_deletes_non_axonal_sections():
    h = H()
    soma, basal, apical = Section(), Section(), Section()
    icell = type(
        "Cell",
        (),
        {"axonal": [], "basal": [basal], "apical": [apical], "soma": [soma]},
    )()
    sim = type("Sim", (), {"neuron": type("Neuron", (), {"h": h})()})()

    isolate_axon(sim=sim, icell=icell)

    assert h.deleted == [basal, apical, soma]
