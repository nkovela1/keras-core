"""Classes and functions implementing to Model SavedModel serialization."""

from keras_core.legacy.saving import saving_utils
from keras_core.legacy.saving.saved_model import constants
from keras_core.legacy.saving.saved_model import layer_serialization
from keras_core.legacy.saving.saved_model import save_impl


class ModelSavedModelSaver(layer_serialization.LayerSavedModelSaver):
    """Model SavedModel serialization."""

    @property
    def object_identifier(self):
        return constants.MODEL_IDENTIFIER

    def _python_properties_internal(self):
        metadata = super()._python_properties_internal()
        # Network stateful property is dependent on the child layers.
        metadata.pop("stateful")
        spec = self.obj._get_save_spec(dynamic_batch=False, inputs_only=False)
        metadata["full_save_spec"] = spec
        # save_spec is saved for forward compatibility on older TF versions.
        metadata["save_spec"] = None if spec is None else spec[0][0]

        metadata.update(
            saving_utils.model_metadata(
                self.obj, include_optimizer=True, require_config=False
            )
        )
        return metadata

    def _get_serialized_attributes_internal(self, serialization_cache):
        default_signature = None

        # Create a default signature function if this is the only object in the
        # cache (i.e. this is the root level object).
        if len(serialization_cache[constants.KERAS_CACHE_KEY]) == 1:
            default_signature = save_impl.default_save_signature(self.obj)

        # Other than the default signature function, all other attributes match
        # with the ones serialized by Layer.
        objects, functions = super()._get_serialized_attributes_internal(
            serialization_cache
        )
        functions["_default_save_signature"] = default_signature
        return objects, functions


class SequentialSavedModelSaver(ModelSavedModelSaver):
    @property
    def object_identifier(self):
        return constants.SEQUENTIAL_IDENTIFIER