Changelog
=========

Earlier releases are listed on the `GitHub releases page <https://github.com/openbraininstitute/BluePyEModel/releases>`_.

2.0.0 (unreleased)
------------------

Removed
~~~~~~~

- The Nexus access point (``bluepyemodel.access_point.nexus`` and ``forge_access_point``), the
  Luigi workflow (``bluepyemodel.tasks``), their examples, and the ``nexus`` and ``luigi`` extras.
  No Nexus deployment with the data the access point expects exists any more, and the Luigi tasks
  need the archived ``bbp-workflow``. They remain available in BluePyEModel 1.x.
- ``get_access_point("nexus")``, ``EModel_pipeline(data_access_point="nexus")`` and
  ``EModel_pipeline_nexus`` now raise a ``ValueError``. The Nexus arguments of ``EModel_pipeline``
  are ignored with a ``DeprecationWarning``, and ``copy_and_compile_mechanisms`` does nothing.

Added
~~~~~

- Command line commands for every step: ``extract``, ``configure-model``, ``store``,
  ``validate``, ``plot``, ``export``, ``summarize``, and ``run`` for the whole pipeline. Every
  command takes the e-model metadata (``--etype``, ``--species``, ...) and a parallel backend.
  ``optimise`` and ``analyse`` also take the metadata options.
- Models configured from gene data with the local access point:
  ``LocalAccessPoint.load_channel_gene_expression``, ``load_ic_map`` and ``get_t_types``, and the
  pipeline setting ``path_ic_map``.
- ``export_emodels_hoc`` takes ``output_dir``.
- Documentation: installation, quickstart, concepts, configuration files, command line, and models
  from gene data.

Fixed
~~~~~

- The me-type given to the ion channel selector contained a run of spaces.
- ``protocols_rheobase`` is optional in the extraction config.

Changed
~~~~~~~

- ``scripts/met_type_ic_profile_generator.py`` moved out of the package: it read a csv file when
  imported.
