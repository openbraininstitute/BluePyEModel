Command line
============

The ``bluepyemodel`` command runs the steps of the pipeline. ``bluepyemodel --help`` lists the
commands, and ``bluepyemodel <command> --help`` their options.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - What it does
   * - ``extract``
     - Extract e-features from the recordings, with the targets of ``path_extract_config``, and
       write the features file of the recipe.
   * - ``configure-model``
     - Create a model configuration and write it over the ``params`` file of the recipe. With
       ``--use-gene-data``, select the ion channels from gene expression data
       (see :doc:`gene_data`).
   * - ``optimise``
     - Optimise the model for one seed. An existing checkpoint is continued.
   * - ``store``
     - Store the best model of the checkpoints in ``final.json``. ``--seed`` selects seeds;
       by default every checkpoint found is stored.
   * - ``validate``
     - Run the validation protocols on the stored models.
   * - ``plot``
     - Plot the stored models. ``--only-validated`` and ``--seed`` select models.
   * - ``export``
     - Export the stored models with ``--format hoc`` or ``--format sonata``. By default only
       the best model is exported (``--all-models`` exports every selected model), each to its
       own directory in ``export_emodels_<format>/``. ``--output-dir`` writes to a given
       directory, with a single ``--seed``.
   * - ``summarize``
     - Print the configuration and the stored models.
   * - ``run``
     - Run extract, optimise (for each ``--seed``), store, validate, plot and export. Extraction
       is skipped when the features file exists, unless ``--force-extraction``.
   * - ``analyse``
     - Store, plot and export to SONATA the model of one checkpoint, and write its EM json.

Options shared by all commands
------------------------------

``--emodel`` and ``--recipes-path`` (required)
    The e-model name and the recipes file.

``--etype``, ``--ttype``, ``--mtype``, ``--species``, ``--brain-region``, ``--iteration-tag``
    The metadata of the e-model. They are part of the names of the checkpoints and of the
    entries of ``final.json``, so give the same values to every step.

``--backend``, ``--workers``, ``--ipyparallel-profile``
    How simulations run in parallel: ``multiprocessing`` on all CPUs of the machine (the default,
    ``--workers`` limits the number of processes), ``ipyparallel`` on the engines of a running
    ipyparallel cluster, or ``serial``. ``optimise`` and ``analyse`` take ``--workers`` only.

``--log-level`` (before the command)
    For example ``bluepyemodel --log-level WARNING run ...``.

On a Slurm cluster
------------------

Optimisations with different seeds are independent: run each as its own job, then store,
validate, plot and export once they are all done. A job array runs one seed per task:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --array=1-10
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=32
    #SBATCH --time=24:00:00

    OPTIONS="--emodel L5PC --recipes-path config/recipes.json --etype cADpyr --species rat --brain-region SSCX"

    bluepyemodel optimise $OPTIONS --seed $SLURM_ARRAY_TASK_ID --workers $SLURM_CPUS_PER_TASK

When the array is finished:

.. code-block:: console

    bluepyemodel run $OPTIONS --seed 1 --seed 2 --seed 3 ...

``run`` continues each optimisation from its checkpoint; once an optimisation has reached its
last generation, nothing more is computed, so this stores, validates, plots and exports the
models of every seed. Or run ``store``, ``validate``, ``plot`` and ``export`` yourself.

To spread a single optimisation over several nodes, start an ipyparallel cluster in the job and
use ``--backend ipyparallel``; the `L5PC example
<https://github.com/openbraininstitute/BluePyEModel/tree/main/examples/L5PC>`_ shows how.
