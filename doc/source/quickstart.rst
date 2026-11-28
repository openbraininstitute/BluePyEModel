Quickstart
==========

This page builds an e-model from patch-clamp recordings with the command line only, using the
`simplecell example <https://github.com/openbraininstitute/BluePyEModel/tree/main/examples/simplecell>`_:
a single-compartment model with Hodgkin-Huxley sodium and potassium channels, fitted to recordings
of a rat cortical neuron.

Get the example
---------------

.. code-block:: console

    git clone https://github.com/openbraininstitute/BluePyEModel.git
    cd BluePyEModel/examples/simplecell
    sh download_ephys_data.sh

The script downloads the recordings to ``ephys_data/``. The example contains:

.. code-block:: none

    simplecell/
        config/
            recipes.json                       # the entry point: which files to use, and settings
            params/simple.json                 # mechanisms and parameters, with bounds to optimise
            extract_config/simplecell_config.json   # recordings and e-feature targets
            features/simplecell.json           # e-features, written by the extraction
        morphologies/simple.swc
        ephys_data/C060109A1-SR-C1/*.ibw       # downloaded recordings

These files are described in :doc:`configuration`.

Build the e-model
-----------------

.. code-block:: console

    bluepyemodel run --emodel simplecell --recipes-path config/recipes.json \
        --etype cADpyr --species rat --brain-region SSCX --seed 1 --force-extraction

``run`` executes every step of the pipeline:

1. **extract**: computes the e-features of the recordings at the target amplitudes and writes
   ``config/features/simplecell.json``. Without ``--force-extraction``, an existing features file
   is used as it is.
2. **optimise**: optimises the model parameters against those e-features. The checkpoint is
   written to ``checkpoints/``.
3. **store**: writes the best model of the optimisation to ``final.json``.
4. **validate**: runs the validation protocols (here ``IDrest_0.4``) on the model and records
   whether it passes.
5. **plot**: writes figures of the optimisation, traces, scores and parameters to ``figures/``.
6. **export**: writes the model in SONATA format (hoc template, nodes file and morphology) to
   ``export_emodels_sonata/``. Use ``--export-format hoc`` for a plain hoc template.

On a laptop this takes less than a minute, because the example uses a tiny optimisation budget:
5 generations of 20 individuals. The resulting model does not reproduce the recordings well. For
a real model, raise ``max_ngen`` (for example to 100) in the ``pipeline_settings`` of
``config/recipes.json``, and run several seeds:

.. code-block:: console

    bluepyemodel run --emodel simplecell --recipes-path config/recipes.json \
        --etype cADpyr --species rat --brain-region SSCX --seed 1 --seed 2 --seed 3

Run the steps one by one
------------------------

Each step is also a command, so you can run, inspect and rerun them separately:

.. code-block:: console

    OPTIONS="--emodel simplecell --recipes-path config/recipes.json --etype cADpyr --species rat --brain-region SSCX"

    bluepyemodel extract $OPTIONS
    bluepyemodel optimise $OPTIONS --seed 1
    bluepyemodel store $OPTIONS --seed 1
    bluepyemodel validate $OPTIONS
    bluepyemodel plot $OPTIONS
    bluepyemodel export $OPTIONS --format hoc
    bluepyemodel summarize $OPTIONS

Give every command the same e-model name and metadata (``--etype``, ``--species``,
``--brain-region``, ...): they are part of the names of the checkpoints and of the entries of
``final.json``, so a command given other values does not find the results of the previous ones.

Use the model
-------------

The exported hoc template can be loaded in NEURON:

.. code-block:: python

    from neuron import h

    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    h.load_file("export_emodels_hoc/<model directory>/model.hoc")
    cell = h.simplecell("export_emodels_hoc/<model directory>", "simple.swc")

From Python
-----------

The same steps are available from Python with :class:`~bluepyemodel.emodel_pipeline.emodel_pipeline.EModel_pipeline`:

.. code-block:: python

    from bluepyemodel.emodel_pipeline.emodel_pipeline import EModel_pipeline

    pipeline = EModel_pipeline(
        emodel="simplecell",
        etype="cADpyr",
        species="rat",
        brain_region="SSCX",
        recipes_path="./config/recipes.json",
    )
    pipeline.extract_efeatures()
    pipeline.optimise(seed=1)
    pipeline.store_optimisation_results()
    pipeline.validation()
    pipeline.plot()
    pipeline.export_emodels()

The `simplecell notebook <https://github.com/openbraininstitute/BluePyEModel/blob/main/examples/simplecell/simplecell.ipynb>`_
goes through these steps with explanations. The
`L5PC example <https://github.com/openbraininstitute/BluePyEModel/tree/main/examples/L5PC>`_
builds a detailed layer 5 pyramidal cell model, on a laptop or on a Slurm cluster.
