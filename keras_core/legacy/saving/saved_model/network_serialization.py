"""Classes and functions implementing to Network SavedModel serialization."""

from keras_core.legacy.saving.saved_model import constants
from keras_core.legacy.saving.saved_model import model_serialization


# FunctionalModel serialization is pretty much the same as Model serialization.
class NetworkSavedModelSaver(model_serialization.ModelSavedModelSaver):
    """Network serialization."""

    @property
    def object_identifier(self):
        return constants.NETWORK_IDENTIFIER