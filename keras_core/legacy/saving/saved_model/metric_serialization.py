"""Classes and functions implementing Metrics SavedModel serialization."""

import tensorflow as tf

from keras_core.saving import object_registration
from keras_core.legacy.saving.saved_model import constants
from keras_core.legacy.saving.saved_model import layer_serialization


class MetricSavedModelSaver(layer_serialization.LayerSavedModelSaver):
    """Metric serialization."""

    @property
    def object_identifier(self):
        return constants.METRIC_IDENTIFIER

    def _python_properties_internal(self):
        metadata = dict(
            class_name=object_registration.get_registered_name(type(self.obj)),
            name=self.obj.name,
            dtype=self.obj.dtype,
        )
        metadata.update(layer_serialization.get_serialized(self.obj))
        if self.obj._build_input_shape is not None:
            metadata["build_input_shape"] = self.obj._build_input_shape
        return metadata

    def _get_serialized_attributes_internal(self, unused_serialization_cache):
        return (
            dict(variables=tf.__internal__.tracking.wrap(self.obj.variables)),
            # TODO(b/135550038): save functions to enable saving custom metrics.
            {},
        )