Installation
============

BluePyEModel needs Python 3.11 or later, on Linux or macOS. On Windows, use the Windows
Subsystem for Linux (WSL).

Install it with pip:

.. code-block:: console

    pip install bluepyemodel

This also installs the NEURON simulator, `BluePyOpt <https://github.com/openbraininstitute/BluePyOpt>`_,
`BluePyEfe <https://github.com/openbraininstitute/BluePyEfe>`_ and
`eFEL <https://github.com/openbraininstitute/eFEL>`_.

Compiling mechanisms
--------------------

Models that use ion channel mechanisms (``.mod`` files) need them compiled with NEURON's
``nrnivmodl``, which needs a C compiler. Run it in the directory where you will run the pipeline:

.. code-block:: console

    nrnivmodl mechanisms

This creates a directory named after your architecture (``x86_64`` or ``arm64``) that NEURON loads
automatically. Models that only use NEURON's built-in mechanisms (``hh``, ``pas``) do not need this
step.

Optional dependencies
---------------------

* ``bluepyemodel[test]``: to run the tests.
* ``bluepyemodel[docs]``: to build this documentation.

Parallel optimisations on several machines use ipyparallel, which is installed with BluePyEModel;
see :doc:`cli`.

Earlier versions
----------------

The Nexus access point and the Luigi workflow were removed in BluePyEModel 2.0, because no Nexus
deployment with the data they expect exists any more. They remain available in BluePyEModel 1.x:

.. code-block:: console

    pip install "bluepyemodel<2"
