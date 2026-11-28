Models from gene data
=====================

Instead of listing the mechanisms of a model by hand in the parameters file, the ion channel
selector (:mod:`bluepyemodel.icselector`) chooses them from the ion channel genes expressed by the
cell type, and sets the bounds and fixed values of their parameters. Cell types are identified by
their t-type and me-type; the correspondence between the two can be obtained with the mapping of
`Roussel et al., 2023 <https://doi.org/10.1371/journal.pcbi.1010058>`_.

Input files
-----------

The selector reads two files:

**Gene expression table** (csv)
    One row per cell type and modality, with the columns ``me-type``, ``t-type`` and
    ``modality``, followed by one column per gene. The modality ``presence`` gives whether each
    gene is expressed (``1`` or ``0``); other modalities give, for example, the distribution of a
    channel along the dendrites.

**Ion channel map** (json)
    Maps genes to mechanisms (with the location and the model to use), and defines the
    parameters and distributions of these mechanisms, with their bounds.

Example files are in
`examples/others/icselector <https://github.com/openbraininstitute/BluePyEModel/tree/main/examples/others/icselector>`_.

Configure a model
-----------------

Give the two files in the ``pipeline_settings`` of the recipe, with paths relative to the e-model
directory:

.. code-block:: json

    "pipeline_settings": {
        "name_gene_map": "icselector/gene_mapping_example.csv",
        "path_ic_map": "icselector/icmapping_example.json"
    }

and the cell type of the model with the metadata options:

.. code-block:: console

    bluepyemodel configure-model --emodel L3_TPC_C_cAC --recipes-path config/recipes.json \
        --ttype "0103 L2/3 IT CTX Glut_1" --mtype "L3_TPC:C" --etype cAC \
        --morphology-name C060114A5 --use-gene-data

or from Python:

.. code-block:: python

    pipeline = EModel_pipeline(
        emodel="L3_TPC_C_cAC",
        ttype="0103 L2/3 IT CTX Glut_1",
        mtype="L3_TPC:C",
        etype="cAC",
        recipes_path="./config/recipes.json",
    )
    pipeline.configure_model(morphology_name="C060114A5", use_gene_data=True)

The selector looks up the t-type and the me-type ``<mtype>_<etype>`` in the gene expression
table. ``--morphology-name`` is a morphology file of the morphology directory, without its
extension.

The configuration, with the selected mechanisms, parameters and distributions, is written over
the ``params`` file of the recipe. Compile the ``.mod`` files of the selected mechanisms in the
``mechanisms/`` directory, then build the model as usual, starting with ``extract`` or
``optimise``.

Use the selector on its own
---------------------------

:class:`~bluepyemodel.icselector.icselector.ICSelector` can also be used without the pipeline:

.. code-block:: python

    from bluepyemodel.icselector.icselector import ICSelector

    selector = ICSelector("icmapping_example.json", "gene_mapping_example.csv")
    parameters, mechanisms, distributions, versions = selector.get_cell_config_from_ttype(
        ["0103 L2/3 IT CTX Glut_1", "L3_TPC:C_cAC"]
    )
