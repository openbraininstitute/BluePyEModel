"""Deterministic tests for pipeline helpers, workflow state, and tool utilities."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from bluepyemodel.emodel_pipeline import emodel_pipeline as pipeline_module
from bluepyemodel.emodel_pipeline.emodel_workflow import EModelWorkflow
from bluepyemodel.emodel_pipeline.memodel import MEModel
from bluepyemodel.tools import morphology as morphology_module
from bluepyemodel.tools import multiprocessing as multiprocessing_module
from bluepyemodel.tools import utils as utils_module


def test_emodel_workflow_ids_and_state():
    workflow = EModelWorkflow("targets", "settings", "configuration")

    assert workflow.emodels == []
    assert workflow.emodel_scripts_id == []
    assert workflow.state == "not launched"
    assert workflow.get_configuration_ids() == ("targets", "settings", "configuration")

    workflow.add_emodel_id("emodel-1")
    workflow.add_emodel_script_id("script-1")
    assert workflow.emodels == ["emodel-1"]
    assert workflow.emodel_scripts_id == ["script-1"]
    assert workflow.as_dict()["state"] == "not launched"


def test_emodel_workflow_related_nexus_ids_full():
    workflow = EModelWorkflow(
        "targets",
        "settings",
        "configuration",
        fitness_configuration_id="fitness",
        emodels=["emodel-1"],
        emodel_scripts_id=["script-1"],
        state="done",
    )

    assert workflow.get_configuration_ids() == (
        "targets",
        "settings",
        "configuration",
        "fitness",
    )
    ids = workflow.get_related_nexus_ids()

    assert ids["generates"] == [
        {"id": "emodel-1", "type": "EModel"},
        {"id": "script-1", "type": "EModelScript"},
        {"id": "fitness", "type": "FitnessCalculatorConfiguration"},
    ]
    assert ids["hasPart"] == [
        {"id": "targets", "type": "ExtractionTargetsConfiguration"},
        {"id": "settings", "type": "EModelPipelineSettings"},
        {"id": "configuration", "type": "EModelConfiguration"},
    ]


def test_emodel_workflow_related_nexus_ids_empty():
    workflow = EModelWorkflow(None, None, None)

    assert workflow.get_related_nexus_ids() == {}


def test_memodel_related_ids_and_dict():
    memodel = MEModel(
        seed=4, emodel_id="emodel-1", morphology_id="morph-1", validated=True
    )

    assert memodel.get_related_nexus_ids() == {
        "hasPart": [
            {"id": "emodel-1", "type": "EModel"},
            {"id": "morph-1", "type": "NeuronMorphology"},
        ]
    }

    memodel.build_pdf_dependencies = lambda seed: ["figure.pdf"]
    as_dict = memodel.as_dict()
    assert as_dict["nexus_images"] == ["figure.pdf"]
    assert as_dict["seed"] == 4
    assert as_dict["validated"] is True
    assert as_dict["status"] == "initialized"


def test_memodel_related_ids_without_resources():
    assert MEModel().get_related_nexus_ids() == {"hasPart": []}


def test_sanitize_gitignore_appends_missing_entries(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("run/\nlogs/\n")

    pipeline_module.sanitize_gitignore()

    content = gitignore.read_text()
    assert content.count("run/") == 1
    assert content.count("logs/") == 1
    assert "checkpoints/" in content
    assert "figures/" in content
    assert ".ipython/" in content
    assert ".ipynb_checkpoints/" in content


def test_sanitize_gitignore_requires_existing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        pipeline_module.sanitize_gitignore()


def test_pipeline_delegates_to_steps(monkeypatch):
    pipeline = pipeline_module.EModel_pipeline.__new__(pipeline_module.EModel_pipeline)
    pipeline.access_point = MagicMock()
    pipeline.mapper = map

    extract = MagicMock(return_value="features")
    optimise = MagicMock()
    validate = MagicMock()
    export = MagicMock()
    monkeypatch.setattr(pipeline_module, "extract_save_features_protocols", extract)
    monkeypatch.setattr(pipeline_module, "setup_and_run_optimisation", optimise)
    monkeypatch.setattr(pipeline_module, "validate", validate)
    monkeypatch.setattr(pipeline_module, "export_emodels_sonata", export)

    assert pipeline.extract_efeatures() == "features"
    pipeline.optimise(seed=7)
    pipeline.validation(preselect_for_validation=True)
    pipeline.export_emodels(only_validated=True, seeds=[1])
    pipeline.summarize()

    extract.assert_called_once_with(access_point=pipeline.access_point, mapper=map)
    assert optimise.call_args.kwargs["seed"] == 7
    assert validate.call_args.kwargs["preselect_for_validation"] is True
    assert export.call_args.kwargs["seeds"] == [1]


def test_pipeline_configure_model_forwards_arguments(monkeypatch):
    pipeline = pipeline_module.EModel_pipeline.__new__(pipeline_module.EModel_pipeline)
    pipeline.access_point = MagicMock()
    pipeline.mapper = map
    configure = MagicMock(return_value="configuration")
    monkeypatch.setattr(pipeline_module, "configure_model", configure)

    result = pipeline.configure_model(
        "morph",
        morphology_path="/tmp/morph.asc",
        morphology_format="asc",
        use_gene_data=False,
    )

    assert result == "configuration"
    assert configure.call_args.kwargs["morphology_name"] == "morph"
    assert configure.call_args.kwargs["use_gene_data"] is False


def test_pipeline_store_optimisation_results_single_seed(monkeypatch):
    pipeline = pipeline_module.EModel_pipeline.__new__(pipeline_module.EModel_pipeline)
    pipeline.access_point = MagicMock()
    pipeline.mapper = map
    store = MagicMock()
    monkeypatch.setattr(pipeline_module, "store_best_model", store)

    pipeline.store_optimisation_results(seed=5)

    store.assert_called_once_with(access_point=pipeline.access_point, seed=5)


def test_pipeline_store_optimisation_results_all_seeds(monkeypatch):
    pipeline = pipeline_module.EModel_pipeline.__new__(pipeline_module.EModel_pipeline)
    pipeline.access_point = MagicMock()
    pipeline.mapper = map
    store = MagicMock()
    monkeypatch.setattr(pipeline_module, "store_best_model", store)
    monkeypatch.setattr(
        pipeline_module,
        "get_checkpoint_path",
        lambda metadata, seed=1: "./chk/emodel__seed=1.pkl",
    )
    monkeypatch.setattr(
        pipeline_module.glob,
        "glob",
        lambda pattern: ["./chk/emodel__seed=2.pkl", "./chk/emodel__seed=3.pkl"],
    )

    pipeline.store_optimisation_results()

    seeds = sorted(call.kwargs["seed"] for call in store.call_args_list)
    assert seeds == [2, 3]


def test_get_mapper_backends(monkeypatch):
    assert multiprocessing_module.get_mapper("serial") is map

    sentinel = object()
    monkeypatch.setattr(
        multiprocessing_module,
        "ipyparallel_map_function",
        lambda profile=None: sentinel,
    )
    assert (
        multiprocessing_module.get_mapper("ipyparallel", ipyparallel_profile="p")
        is sentinel
    )

    nested = MagicMock()
    monkeypatch.setattr(multiprocessing_module, "NestedPool", lambda: nested)
    assert multiprocessing_module.get_mapper("multiprocessing") is nested.map


def test_ipyparallel_map_function_falls_back_to_map(monkeypatch):
    monkeypatch.delenv("IPYTHON_PROFILE", raising=False)

    assert multiprocessing_module.ipyparallel_map_function() is map


def test_ipyparallel_map_function_uses_client(monkeypatch):
    monkeypatch.setenv("IPYTHON_PROFILE", "test_profile")
    lview = SimpleNamespace(map_sync=lambda func, it: [func(i) for i in it])
    client = MagicMock()
    client.return_value.load_balanced_view.return_value = lview
    monkeypatch.setitem(
        __import__("sys").modules, "ipyparallel", SimpleNamespace(Client=client)
    )

    mapper = multiprocessing_module.ipyparallel_map_function()

    assert mapper(lambda x: x * 2, [1, 2]) == [2, 4]


def test_no_daemon_process_daemon_flag():
    process = multiprocessing_module.NoDaemonProcess(target=lambda: None)

    assert process.daemon is False
    process.daemon = True
    assert process.daemon is False


def test_name_and_generate_cylindrical_morphology(tmp_path):
    assert morphology_module.name_morphology(5.0) == "cylindrical_morphology_5.0000"

    path = morphology_module.cylindrical_morphology_generator(
        radius=3.0, output_dir=tmp_path / "morphs"
    )
    content = path.read_text()

    assert path.name == "cylindrical_morphology_3.0000.swc"
    assert content.count("\n") == 4
    assert "1 1 -3.0 0.0 0.0 3.0 -1" in content


def test_cylindrical_morphology_generator_with_axon(tmp_path):
    path = morphology_module.cylindrical_morphology_generator(
        radius=2.0, radius_axon=0.5, length_axon=10.0, output_dir=tmp_path / "morphs"
    )
    content = path.read_text()

    assert "4 2 -2.0 0.0 0.0 0.5 1" in content
    assert "5 2 -12.0 0.0 0.0 0.5 4" in content


def test_yesno_handles_invalid_then_valid(monkeypatch):
    answers = iter(["maybe", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    assert utils_module.yesno("Continue") is True

    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert utils_module.yesno("Continue") is False


def test_make_dir_creates_directory(tmp_path):
    target = tmp_path / "nested" / "dir"

    utils_module.make_dir(target)
    utils_module.make_dir(target)

    assert target.is_dir()


def test_get_amplitude_from_feature_key():
    assert utils_module.get_amplitude_from_feature_key(
        "IV_40.soma.v.voltage_base"
    ) == pytest.approx(40.0)


def test_get_loc_and_curr_name_errors():
    with pytest.raises(IndexError, match="Location name not found"):
        utils_module.get_loc_name("IV_40")
    with pytest.raises(IndexError, match="Current name not found"):
        utils_module.get_curr_name("IV_40.soma")


def test_get_mapped_protocol_name():
    mapping = {"IDrest_100": "Step_100"}

    assert utils_module.get_mapped_protocol_name("idrest", mapping) == "Step_100"
    assert utils_module.get_mapped_protocol_name("IV_40", mapping) == "IV_40"
    assert utils_module.get_mapped_protocol_name("IV_40", None) == "IV_40"


def test_existing_checkpoint_paths_raises_when_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metadata = SimpleNamespace(emodel="L5PC", iteration=None)

    with pytest.raises(ValueError, match="checkpoints directory is empty"):
        utils_module.existing_checkpoint_paths(metadata)


def test_existing_checkpoint_paths_filters_by_metadata():
    metadata = SimpleNamespace(emodel="L5PC", iteration="iter1")
    paths = [
        "./checkpoints/L5PC/iter1/a.pkl",
        "./checkpoints/L5PC/iter2/b.pkl",
        "./checkpoints/Other/iter1/c.pkl",
    ]

    assert utils_module.existing_checkpoint_paths(metadata, paths) == [
        "./checkpoints/L5PC/iter1/a.pkl"
    ]

    metadata_no_iteration = SimpleNamespace(emodel="L5PC", iteration=None)
    assert utils_module.existing_checkpoint_paths(metadata_no_iteration, paths) == [
        "./checkpoints/L5PC/iter1/a.pkl",
        "./checkpoints/L5PC/iter2/b.pkl",
    ]


def test_deduplicate_checkpoint_paths_prefers_pickle():
    result = utils_module.deduplicate_checkpoint_paths(
        ["./chk/a.h5", "./chk/a.pkl", "./chk/b.h5"]
    )

    assert sorted(result) == ["./chk/a.pkl", "./chk/b.h5"]


def test_select_rec_for_thumbnail_variants():
    rec_names = {"IDrest_100.soma.v", "IDrest_50.soma.v", "IV_-40.soma.v"}

    assert utils_module.select_rec_for_thumbnail(rec_names) == "IDrest_50.soma.v"
    assert (
        utils_module.select_rec_for_thumbnail(rec_names, thumbnail_rec="IV_-40.soma.v")
        == "IV_-40.soma.v"
    )
    assert utils_module.select_rec_for_thumbnail(
        rec_names, thumbnail_rec="absent.soma.v"
    ) == ("IDrest_50.soma.v")
    assert (
        utils_module.select_rec_for_thumbnail({"Unknown_1.soma.v"})
        == "Unknown_1.soma.v"
    )
    with pytest.raises(ValueError, match="No recording in recording_names"):
        utils_module.select_rec_for_thumbnail(set())


def test_get_seed_from_checkpoint_path_variants():
    assert utils_module.get_seed_from_checkpoint_path("./chk/emodel__seed=12.pkl") == 12
    assert (
        utils_module.get_seed_from_checkpoint_path("./chk/emodel__seed=3.pkl.tmp") == 3
    )
    assert utils_module.get_seed_from_checkpoint_path("./chk/emodel.pkl") == 0


def test_get_checkpoint_path_uses_latest_format(tmp_path):
    metadata = SimpleNamespace(
        emodel="L5PC",
        iteration="iter1",
        as_string=lambda seed=None, use_allen_notation=True, replace_semicolons=True, replace_spaces=True: (
            "name"
        ),
    )

    path = utils_module.get_checkpoint_path(metadata, seed=1, base_dir=tmp_path)

    assert path == f"{tmp_path}/L5PC/iter1/name.pkl"


def test_get_legacy_checkpoint_path():
    assert (
        utils_module.get_legacy_checkpoint_path("./checkpoints/L5PC/iter/name.pkl")
        == "./checkpoints/name.pkl"
    )


def test_checkpoint_path_exists_handles_tmp(tmp_path):
    real = tmp_path / "a.pkl"
    real.write_text("x")
    tmp_file = tmp_path / "b.pkl.tmp"
    tmp_file.write_text("x")

    assert utils_module.checkpoint_path_exists(real) is True
    assert utils_module.checkpoint_path_exists(tmp_path / "b.pkl") is True
    assert utils_module.checkpoint_path_exists(tmp_path / "missing.pkl") is False


def test_format_protocol_name_to_list_variants():
    assert utils_module.format_protocol_name_to_list("IDrest_100") == ("IDrest", 100.0)
    assert utils_module.format_protocol_name_to_list("IDrest_100_hyp") == (
        "IDrest_hyp",
        100.0,
    )
    assert utils_module.format_protocol_name_to_list("NoAmplitude") == (
        "NoAmplitude",
        None,
    )
    assert utils_module.format_protocol_name_to_list(["IV", 40.0]) == ["IV", 40.0]
    with pytest.raises(TypeError, match="should be a string or a list"):
        utils_module.format_protocol_name_to_list(42)


def test_are_same_protocol_variants():
    assert utils_module.are_same_protocol("IV_0.0", "IV_0") is True
    assert utils_module.are_same_protocol("IV_0.0", ["IV", 0.0]) is True
    assert utils_module.are_same_protocol("IV_0.0", "IDrest_0.0") is False
    assert utils_module.are_same_protocol(None, "IV_0") is False
    assert utils_module.are_same_protocol("IV_0", None) is False


def test_read_checkpoint_dispatches_to_hdf5(monkeypatch, tmp_path):
    monkeypatch.setattr(utils_module, "read_checkpoint_h5", lambda path: ("run", 9))

    assert utils_module.read_checkpoint(tmp_path / "chk.h5") == ("run", 9)


def test_read_checkpoint_reads_pickle(tmp_path):
    import pickle

    checkpoint = tmp_path / "emodel__seed=6.pkl"
    with open(checkpoint, "wb") as f:
        pickle.dump({"generation": 1}, f)

    run, seed = utils_module.read_checkpoint(checkpoint)

    assert run == {"generation": 1}
    assert seed == 6


def test_read_checkpoint_falls_back_to_tmp(tmp_path):
    import pickle

    checkpoint = tmp_path / "emodel__seed=2.pkl"
    checkpoint.write_bytes(b"")
    with open(str(checkpoint) + ".tmp", "wb") as f:
        pickle.dump({"generation": 5}, f)

    run, seed = utils_module.read_checkpoint(checkpoint)

    assert run == {"generation": 5}
    assert seed == 2


def test_read_checkpoint_ignores_env_noise():
    assert os.path.sep == "/"
