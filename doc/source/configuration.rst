Configuration files
===================

An e-model is described by a recipes file, which points to the other configuration files. Paths
in these files are relative to the e-model directory: the directory where the pipeline runs, or
``./run/<iteration tag>/`` when an iteration tag is given.

Recipes
-------

The recipes file is a json dictionary with one entry per e-model:

.. code-block:: json

    {
        "simplecell": {
            "morph_path": "./morphologies/",
            "morphology": [["simple", "simple.swc"]],
            "params": "config/params/simple.json",
            "features": "config/features/simplecell.json",
            "pipeline_settings": {
                "path_extract_config": "config/extract_config/simplecell_config.json",
                "optimiser": "SO-CMA",
                "max_ngen": 100,
                "validation_protocols": ["IDrest_0.4"]
            }
        }
    }

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Content
   * - ``morph_path``
     - Directory containing the morphology.
   * - ``morphology``
     - A list with one ``[name, file name]`` pair: the morphology the model is built on (``.asc``
       or ``.swc``).
   * - ``params``
     - The parameters file: mechanisms, distributions and parameters of the model.
   * - ``features``
     - The features file: e-features and protocols. Written by the extraction, read by every
       later step.
   * - ``pipeline_settings``
     - Settings of all the steps, see `Pipeline settings`_.
   * - ``morph_modifiers`` (optional)
     - Modifications of the morphology before simulation: names of functions of
       :mod:`bluepyemodel.evaluation.modifiers`, or ``[module path, function name]`` pairs for your
       own. If absent, the axon is replaced by a tapered axon initial segment
       (``replace_axon_with_taper``); ``[]`` keeps the morphology as it is. Can also be given in
       ``pipeline_settings``.
   * - ``multiloc_map`` (optional)
     - Names of groups of section lists used in the parameters file, if they differ from the
       defaults (see below).
   * - ``seclist_names``, ``secarray_names`` (optional)
     - Section list and section array names of the morphology, if they differ from NEURON's
       defaults.

Parameters
----------

The parameters file defines the mechanisms of each section list, the distributions of parameters
along the dendrites, and the parameters. A parameter given as one value is fixed; a parameter
given as ``[lower, upper]`` is optimised between these bounds.

.. code-block:: json

    {
        "mechanisms": {
            "all": {"mech": ["pas"]},
            "somatic": {"mech": ["NaTg", "SKv3_1"]},
            "somadend": {"mech": ["Ih"]}
        },
        "distributions": {
            "exp": {"fun": "(-0.8696 + 2.087*math.exp(({distance})*0.0031))*{value}"}
        },
        "parameters": {
            "global": [
                {"name": "v_init", "val": -80},
                {"name": "celsius", "val": 34}
            ],
            "all": [{"name": "g_pas", "val": [1e-05, 6e-05]}],
            "somatic": [{"name": "gNaTgbar_NaTg", "val": [0.0, 1.0]}],
            "somadend": [{"name": "gIhbar_Ih", "val": [0, 0.0002], "dist": "exp"}]
        }
    }

The locations are NEURON section lists (``somatic``, ``axonal``, ``basal``, ``apical``,
``myelinated``), ``global``, or these groups of section lists:

.. list-table::
   :header-rows: 1

   * - Group
     - Section lists
   * - ``all``, ``allact``
     - apical, basal, somatic, axonal
   * - ``alldend``
     - apical, basal
   * - ``somadend``, ``allnoaxon``
     - apical, basal, somatic
   * - ``somaxon``
     - axonal, somatic

A distribution's ``fun`` is a Python expression of ``{distance}`` (the distance to the soma, in
µm) and ``{value}`` (the parameter value). Extra parameters of a distribution are listed in its
``parameters`` and given, with their bounds, under the location ``distribution_<name>``. Mechanisms
compiled from ``.mod`` files are read from the ``mechanisms/`` directory of the e-model directory.

Extraction targets
------------------

The pipeline setting ``path_extract_config`` points to a json file that lists the recordings and
the e-features to extract from them:

.. code-block:: json

    {
        "files": [
            {
                "cell_name": "C060109A1-SR-C1",
                "filename": "X_IDrest_ch0_326",
                "ecodes": {"IDrest": {"ljp": 14.0, "ton": 700, "toff": 2700}},
                "other_metadata": {
                    "i_file": "./ephys_data/C060109A1-SR-C1/X_IDrest_ch0_326.ibw",
                    "v_file": "./ephys_data/C060109A1-SR-C1/X_IDrest_ch1_326.ibw",
                    "i_unit": "A",
                    "v_unit": "V",
                    "t_unit": "s"
                }
            },
            {
                "cell_name": "C010600C1-MT-C1",
                "filename": "C010600C1-MT-C1",
                "filepath": "./ephys_data/C010600C1-MT-C1.nwb",
                "ecodes": {"IDRest": {}}
            }
        ],
        "targets": [
            {"efeature": "Spikecount", "protocol": "IDrest", "amplitude": 150, "tolerance": 10},
            {"efeature": "voltage_base", "protocol": "IDrest", "amplitude": 150, "tolerance": 10}
        ],
        "protocols_rheobase": ["IDthresh"]
    }

``files`` lists the recordings. Each entry belongs to a cell (``cell_name``) and gives the
protocols it contains in ``ecodes``, with their stimulus timing when the file does not have it
(``ton`` and ``toff`` in ms, and the liquid junction potential ``ljp`` in mV). Igor (``.ibw``)
recordings have separate current and voltage files, given in ``other_metadata``; NWB recordings
are given with ``filepath``.

``targets`` lists the e-features to extract: the eFEL feature name, the protocol, the amplitude
(in percent of the rheobase, or in nA with ``extract_absolute_amplitudes``) and the tolerance
around that amplitude within which sweeps are averaged. Optional keys are ``efeature_name``,
``weight`` and ``efel_settings``.

``protocols_rheobase`` (optional) lists the protocols used to compute the rheobase of each cell.
It is needed when the amplitudes are relative to the rheobase.

The L5PC example builds this file with Python from the recordings found on disk, with
:class:`~bluepyemodel.efeatures_extraction.targets_configurator.TargetsConfigurator`.

Features
--------

The features file is written by the extraction and contains:

* ``efeatures``: for each target, the protocol, recording, eFEL feature name, mean and standard
  deviation (``original_std``), and number of values averaged (``sample_size``);
* ``protocols``: for each protocol and amplitude, the stimuli, the recordings and whether the
  protocol is used for validation (``validation``) instead of optimisation.

It can also be written by hand, in which case extraction is not needed.

Pipeline settings
-----------------

The ``pipeline_settings`` of a recipe configure every step: extraction (``path_extract_config``,
``extract_absolute_amplitudes``, ``name_rmp_protocol``, ``name_Rin_protocol``, ...),
optimisation (``optimiser``, ``optimisation_params``, ``max_ngen``,
``optimisation_checkpoint_period``, ...), validation (``validation_protocols``,
``validation_function``, ...), plotting (``plot_currentscape``, ``plot_IV_curves``, ...) and the
ion channel selector (``name_gene_map``, ``path_ic_map``). Every setting, with its default, is
listed in :class:`~bluepyemodel.emodel_pipeline.emodel_settings.EModelPipelineSettings`.

Output files
------------

``final.json``
    The stored models, one entry per e-model and seed, with their parameters, scores on the
    optimisation protocols (``fitness``) and on the validation protocols
    (``validation_fitness``), total ``score`` and ``validated`` flag.

``checkpoints/<e-model>/<iteration tag>/``
    The optimisation checkpoints, one per seed. An optimisation started again with an existing
    checkpoint continues from its last generation.
