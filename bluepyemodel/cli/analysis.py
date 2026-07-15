"""Analysis CLI subcommand."""

import logging
import shutil
from pathlib import Path

import click


def _resolve_checkpoint_path(access_point, seed, checkpoint_path, checkpoints_dir):
    if checkpoint_path is not None:
        return Path(checkpoint_path)

    from bluepyemodel.tools.utils import get_checkpoint_path

    return get_checkpoint_path(
        access_point.emodel_metadata,
        seed=seed,
        base_dir=checkpoints_dir,
    )


@click.command()
@click.option("--seed", type=int, required=True, help="Random seed")
@click.option("--emodel", required=True, help="EModel name")
@click.option(
    "--workers", type=int, required=False, default=None, help="Number of parallel workers."
)
@click.option(
    "--checkpoints-dir",
    required=False,
    type=click.Path(path_type=Path),
    default=Path("./checkpoints"),
    help="Directory containing optimisation checkpoints.",
)
@click.option(
    "--checkpoint-path",
    required=False,
    type=click.Path(path_type=Path),
    default=None,
    help="Explicit path to an optimisation checkpoint (.pkl or .h5).",
)
@click.option(
    "--recipes-path",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to recipes file or directory",
)
@click.option(
    "--output-figures-dir",
    required=False,
    type=click.Path(path_type=Path),
    default=Path("./figures"),
    help="Directory to output analysis figures.",
)
@click.option(
    "--output-nodes-dir",
    required=False,
    type=click.Path(path_type=Path),
    default=Path("./nodes"),
    help="Directory to output nodes, hoc, morphology.",
)
@click.option(
    "--output-em-dir",
    required=False,
    type=click.Path(path_type=Path),
    default=Path("./em"),
)
def analyse(
    seed,
    emodel,
    workers,
    checkpoints_dir,
    checkpoint_path,
    recipes_path,
    output_figures_dir,
    output_nodes_dir,
    output_em_dir,
):
    """Analyse an optimisation checkpoint: store results, plot, export, and write EM JSON."""

    # fontTools is extremely noisy at INFO level
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    from bluepyemodel.access_point.local import LocalAccessPoint
    from bluepyemodel.emodel_pipeline.plotting import plot_models
    from bluepyemodel.export_emodel.export_emodel import export_emodels_sonata
    from bluepyemodel.optimisation import store_best_model
    from bluepyemodel.tools.create_em_json import create_em_json
    from bluepyemodel.tools.multiprocessing import NestedPool

    access_point = LocalAccessPoint(
        emodel=emodel,
        recipes_path=recipes_path,
    )
    resolved_checkpoint = _resolve_checkpoint_path(
        access_point,
        seed,
        checkpoint_path,
        checkpoints_dir,
    )
    store_best_model(
        access_point,
        seed=seed,
        checkpoint_path=str(resolved_checkpoint),
    )
    pp_settings = access_point.pipeline_settings

    tmp_figures_dir = output_figures_dir / "tmp"
    tmp_figures_dir.mkdir(parents=True, exist_ok=True)

    with NestedPool(processes=workers) as pool:
        plot_models(
            access_point=access_point,
            mapper=pool.map,
            seeds=[seed],
            figures_dir=tmp_figures_dir,
            plot_optimisation_progress=pp_settings.plot_optimisation_progress,
            optimiser=pp_settings.optimiser,
            plot_parameter_evolution=pp_settings.plot_parameter_evolution,
            plot_distributions=pp_settings.plot_distributions,
            plot_scores=pp_settings.plot_scores,
            plot_traces=pp_settings.plot_traces,
            plot_thumbnail=pp_settings.plot_thumbnail,
            plot_currentscape=pp_settings.plot_currentscape,
            plot_dendritic_ISI_CV=pp_settings.plot_dendritic_ISI_CV,
            plot_dendritic_rheobase=pp_settings.plot_dendritic_rheobase,
            plot_bAP_EPSP=pp_settings.plot_bAP_EPSP,
            plot_IV_curve=pp_settings.plot_IV_curves,
            plot_FI_curve_comparison=pp_settings.plot_FI_curve_comparison,
            plot_phase_plot=pp_settings.plot_phase_plot,
            plot_traces_comparison=pp_settings.plot_traces_comparison,
            run_plot_custom_sinspec=pp_settings.run_plot_custom_sinspec,
            IV_curve_prot_name=pp_settings.IV_curve_prot_name,
            FI_curve_prot_name=pp_settings.FI_curve_prot_name,
            phase_plot_settings=pp_settings.phase_plot_settings,
            sinespec_settings=pp_settings.sinespec_settings,
            custom_bluepyefe_cells_pklpath=pp_settings.custom_bluepyefe_cells_pklpath,
            custom_bluepyefe_protocols_pklpath=pp_settings.custom_bluepyefe_protocols_pklpath,
            only_validated=False,
            save_recordings=pp_settings.save_recordings,
            load_from_local=True,
        )

        # flatten images and put to output directory
        for image_file in list(tmp_figures_dir.rglob("*.pdf")) + list(
            tmp_figures_dir.rglob("*.png")
        ):
            target_file = output_figures_dir / image_file.name
            shutil.move(str(image_file), str(target_file))
        shutil.rmtree(tmp_figures_dir)

        export_emodels_sonata(
            access_point,
            only_validated=False,
            only_best=False,
            seeds=[seed],
            map_function=pool.map,
            output_dir=output_nodes_dir,
        )
        create_em_json(
            access_point,
            seed=seed,
            map_function=pool.map,
            output_dir=output_em_dir,
        )
