# BluePyEModel Examples

This directory includes various examples demonstrating how to use BluePyEModel:

- [**L5PC**](./L5PC/README.rst): Provides a guide for setting up and running the E-Model building pipeline locally on a personal computer or on a Slurm cluster. This is an example based on the juvenile (P14) rat somatosensory cortex layer 5 thick-tufted pyramidal cell (L5TPC) e-model optimisation of the [SSCxEModelExamples](https://github.com/BlueBrain/SSCxEModelExamples/tree/main/optimization) repository.

- [**simplecell**](./simplecell/simplecell.ipynb): Demonstrates creating a single-compartment neuron model with two parameters using rheobase independent optimisation.

- [**others**](./others/README.rst): Contains various examples demonstrating different functionalities and use cases.
    - [**ICSelector**](./others/icselector/icselector_example.py): Selects ion channel models using gene mappings; assigns optimisation parameter bounds and fixed values for a model.