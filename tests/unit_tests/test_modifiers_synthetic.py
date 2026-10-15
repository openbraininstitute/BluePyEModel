"""Deterministic tests for morphology modifiers using neuron-like fakes."""

import re

import pytest

from bluepyemodel.evaluation.modifiers import (
    get_synth_axon_hoc,
    replace_axon_legacy,
    replace_axon_olfactory_bulb,
    replace_axon_with_taper,
    synth_axon,
    taper_function,
)


class Segment:
    def __init__(self, diam=1.0):
        self.diam = diam


class Section:
    """Minimal stand-in for a NEURON section."""

    def __init__(self, length=10.0, nseg=1, diam=1.0, name="section"):
        self.L = length
        self._nseg = nseg
        self.name = name
        self.segments = [Segment(diam) for _ in range(nseg)]
        self.connections = []

    @property
    def nseg(self):
        return self._nseg

    @nseg.setter
    def nseg(self, value):
        value = int(value)
        current = self.segments
        if value > len(current):
            current = current + [Segment(current[-1].diam if current else 1.0)] * 0
            while len(current) < value:
                current.append(Segment(current[-1].diam if current else 1.0))
        else:
            current = current[:value]
        self.segments = current
        self._nseg = value

    def __iter__(self):
        return iter(self.segments)

    def connect(self, parent, parentx=None, childx=None):
        self.connections.append((parent, parentx, childx))


class SectionList(list):
    """List that mimics NEURON's ``append(sec=...)`` API."""

    def append(self, sec=None):  # pylint: disable=arguments-differ
        super().append(sec)


class FakeH:
    def __init__(self, cell):
        self.cell = cell
        self.deleted = []
        self.executed = []

    def delete_section(self, sec):
        self.deleted.append(sec)

    def execute(self, command, _cell=None):
        self.executed.append(command)
        match = re.match(r"create (\w+)(?:\[(\d+)\])?$", command)
        name, count = match.group(1), match.group(2)
        if count is None:
            setattr(self.cell, name, Section(name=name))
        else:
            setattr(
                self.cell,
                name,
                [Section(name=f"{name}[{i}]") for i in range(int(count))],
            )


class FakeCell:
    def __init__(self, axonal_sections):
        self.soma = [Section(name="soma")]
        self.axonal = SectionList(axonal_sections)
        self.all = SectionList()
        self.myelinated = SectionList()


def make_sim_and_cell(axonal_sections):
    cell = FakeCell(axonal_sections)
    sim = type("Sim", (), {"neuron": type("Neuron", (), {"h": FakeH(cell)})()})()
    return sim, cell


def test_replace_axon_with_taper_builds_stub_axon_and_myelin():
    axonal = [Section(length=50.0, nseg=6, diam=2.0) for _ in range(3)]
    sim, cell = make_sim_and_cell(axonal)

    replace_axon_with_taper(sim=sim, icell=cell)

    assert sim.neuron.h.executed == ["create axon[2]", "create myelin[1]"]
    assert len(sim.neuron.h.deleted) == 3
    assert [section.L for section in cell.axon] == [30.0, 30.0]
    assert all(section.nseg == 5 for section in cell.axon)
    assert cell.axon[0].connections[0][0] is cell.soma[0]
    assert cell.axon[1].connections[0][0] is cell.axon[0]
    assert cell.myelin[0].L == 1000
    assert cell.myelin[0].nseg == 5
    assert cell.myelin[0].connections[0][0] is cell.axon[1]
    assert cell.myelin[0].diam == pytest.approx(2.0)


def test_replace_axon_with_taper_handles_short_diameter_list():
    axonal = [Section(length=1.0, nseg=1, diam=3.0) for _ in range(3)]
    sim, cell = make_sim_and_cell(axonal)

    replace_axon_with_taper(sim=sim, icell=cell)

    assert cell.myelin[0].diam == pytest.approx(3.0)
    assert len(cell.axonal) > 3


def test_replace_axon_legacy_builds_two_sections():
    axonal = [Section(length=100.0, nseg=10, diam=1.5) for _ in range(2)]
    sim, cell = make_sim_and_cell(axonal)

    replace_axon_legacy(sim=sim, icell=cell)

    assert sim.neuron.h.executed == ["create axon[2]"]
    assert [section.L for section in cell.axon] == [30.0, 30.0]
    assert cell.axon[0].connections[0][0] is cell.soma[0]
    assert cell.axon[1].connections[0][0] is cell.axon[0]
    assert all(
        seg.diam == pytest.approx(1.5) for section in cell.axon for seg in section
    )


def test_synth_axon_creates_tapered_segment_and_myelin():
    axonal = [Section(length=20.0, nseg=2, diam=1.0)]
    sim, cell = make_sim_and_cell(axonal)
    params = [60.0, 2.0, 10.0, 0.5]

    synth_axon(sim=sim, icell=cell, params=params, scale=2.0)

    assert sim.neuron.h.executed == ["create axon[2]", "create myelin[1]"]
    assert [section.L for section in cell.axon] == [30.0, 30.0]
    assert all(section.nseg == 5 for section in cell.axon)
    expected_first = taper_function(0.0, *params[1:], scale=2.0)
    assert cell.axon[0].segments[0].diam == pytest.approx(expected_first)
    expected_last = taper_function(60.0, *params[1:], scale=2.0)
    assert cell.myelin[0].diam == pytest.approx(expected_last)
    assert cell.myelin[0].connections[0][0] is cell.axon[1]


def test_replace_axon_olfactory_bulb_builds_full_axon():
    axonal = [Section(length=10.0, nseg=1)]
    sim, cell = make_sim_and_cell(axonal)

    replace_axon_olfactory_bulb(sim=sim, icell=cell)

    assert sim.neuron.h.executed == [
        "create hillock",
        "create initialseg",
        "create node[5]",
        "create myelin[5]",
    ]
    assert cell.hillock.L == 5
    assert cell.hillock.nseg == 3
    assert all(seg.diam == pytest.approx(10.0925) for seg in cell.hillock)
    assert cell.initialseg.L == 30
    assert all(seg.diam == pytest.approx(1.5) for seg in cell.initialseg)
    assert all(section.L == 1 for section in cell.node)
    assert all(section.L == 1000 for section in cell.myelin)
    assert cell.hillock.connections[0][0] is cell.soma[0]
    assert cell.initialseg.connections[0][0] is cell.hillock
    assert cell.myelin[0].connections[0][0] is cell.initialseg
    assert cell.node[4].connections[0][0] is cell.myelin[4]
    assert len(cell.myelinated) == 5


def test_get_synth_axon_hoc_contains_procedure():
    hoc = get_synth_axon_hoc([60.0, 2.0, 10.0, 0.5])

    assert "proc replace_axon()" in hoc
    assert "create axon[2]" in hoc
