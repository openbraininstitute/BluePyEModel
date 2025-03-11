"""
Copyright 2025 Open Brain Institute

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


class SimulatableNeuron:
    """Whole neuron model, including links to morphology, ion channels models and hoc."""

    def __init__(
            self,
            description,
            emodel_script_id,
            mechanism_ids,
            morphology_id,
            holding_current=None,
            threshold_current=None,
            validated=False,
        ):
        """Init

        from_circuit and synaptomes, that might be present in the database,
        are not implemented here.
        name, eModel, eType, subject, brainRegion are automatically filled in by forge_access_point for now using emodel_metadata
        
        Args:
            description (str): description of the model
            emodel_script_id (str): ID of the hoc model in the database
            mechanism_ids (list of str): IDs of the ion channel models in the database
            morphology_id (str): ID of the morphology in the database
            holding_current (float): holding current to use in protocols (in nA)
            threshold_current (float): current at which the cell starts firing (in nA)
            validated (bool): whether the model has been validated by user
        """
        # check if brain region and subject are automatically added
        self.description = description
        self.emodel_script_id = emodel_script_id
        self.mechanism_ids = mechanism_ids
        self.morphology_id = morphology_id
        self.holding_current = holding_current
        self.threshold_current = threshold_current
        self.validated = validated

    def as_dict(self):
        """Used for the storage of the object"""
        return vars(self)
