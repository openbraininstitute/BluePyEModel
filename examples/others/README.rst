======================
Miscellaneous Examples
======================

This folder contains various examples demonstrating different functionalities and use cases.

icselector : Ion Channel Selector
=================================

ICSelector selects ion channel models from the genes expressed by a cell type, and assigns the optimisation parameter bounds and fixed values of the model. It reads two local files:

* a gene expression table (csv) with the columns ``me-type``, ``t-type`` and ``modality``, followed by one column per gene, for example `gene_mapping_example.csv <./icselector/gene_mapping_example.csv>`_;
* an ion channel map (json) that maps genes to mechanisms, parameters and distributions, for example `icmapping_example.json <./icselector/icmapping_example.json>`_.

To run ICSelector on its own on these files:

.. code-block:: bash

    cd icselector
    sh ./test_icselector.sh

To configure a model from gene data in the e-model building pipeline, give the two files in the ``pipeline_settings`` of the recipe, with paths relative to the e-model directory:

.. code-block:: json

    "pipeline_settings": {
        "name_gene_map": "icselector/gene_mapping_example.csv",
        "path_ic_map": "icselector/icmapping_example.json"
    }

Then create the pipeline with the cell type of the model, and configure the model from gene data:

.. code-block:: python

    pipeline = EModel_pipeline(
        emodel="L3_TPC_C_cAC",
        ttype="0103 L2/3 IT CTX Glut_1",
        mtype="L3_TPC:C",
        etype="cAC",
        recipes_path="./config/recipes.json",
    )
    pipeline.configure_model(morphology_name="C060114A5", use_gene_data=True)

``morphology_name`` is the name of a morphology file in the morphology directory, without its extension. The selector looks up the t-type and the me-type ``<mtype>_<etype>`` in the gene expression table. The resulting configuration is written over the ``params`` file of the recipe.
