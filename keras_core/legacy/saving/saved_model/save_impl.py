from keras_core import models
from absl import logging


def should_skip_serialization(layer):
    """Skip serializing extra objects and functions if layer inputs aren't
    set."""
    saved_model_input_spec_set = (
        isinstance(layer, models.Model)
        and layer._saved_model_inputs_spec is not None
    )
    if not layer.built and not saved_model_input_spec_set:
        logging.warning(
            "Skipping full serialization of Keras layer {}, because "
            "it is not built.".format(layer)
        )
        return True
    return False

def wrap_layer_objects(layer, serialization_cache):
    """Returns extra trackable objects to attach to the serialized layer.

    Args:
      layer: Keras Layer object.
      serialization_cache: Dictionary shared between all objects during
        serialization.

    Returns:
      A dictionary containing all checkpointable objects from a
      SerializedAttributes object. See LayerAttributes and ModelAttributes for
      entire list of objects
    """
    # Wrap all regularization losses as tf.functions.
    # First, generate list of all regularization losses in this layer and
    # sublayers.
    all_losses = layer._callable_losses[:]
    for child_layer in utils.list_all_layers(layer):
        all_losses.extend(child_layer._callable_losses)
    # Next, wrap all loss functions as tf.functions. Use the serialization cache
    # to store already-wrapped functions.
    keras_loss_cache = serialization_cache.setdefault("keras_losses", {})
    wrapped_loss_functions = []
    for loss_fn in all_losses:
        if loss_fn in keras_loss_cache:
            wrapped_loss_functions.append(keras_loss_cache[loss_fn])
        else:
            wrapped_loss = _wrap_unconditional_loss(
                loss_fn, len(keras_loss_cache)
            )
            keras_loss_cache[loss_fn] = wrapped_loss
            wrapped_loss_functions.append(wrapped_loss)
    wrapped_layer_losses = [
        keras_loss_cache[fn] for fn in layer._callable_losses[:]
    ]

    layer_metrics = tf.__internal__.tracking.wrap(
        {m.name: m for m in layer._metrics}
    )

    return dict(
        variables=tf.__internal__.tracking.wrap(layer.variables),
        trainable_variables=tf.__internal__.tracking.wrap(layer.trainable_variables),
        non_trainable_variables=tf.__internal__.tracking.wrap(
            layer.non_trainable_variables
        ),
        layers=tf.__internal__.tracking.wrap(utils.list_all_layers(layer)),
        metrics=tf.__internal__.tracking.wrap(layer.metrics),
        regularization_losses=tf.__internal__.tracking.wrap(
            wrapped_loss_functions
        ),
        layer_regularization_losses=tf.__internal__.tracking.wrap(
            wrapped_layer_losses
        ),
        layer_metrics=layer_metrics,
    )


class LayerCall:
    """Function that triggers traces of other functions in the same
    collection."""

    def __init__(self, call_collection, call_fn, name):
        """Initializes a LayerCall object.

        Args:
          call_collection: a LayerCallCollection, which contains the other layer
            call functions (e.g. call_with_conditional_losses, call). These
            functions should be traced with the same arguments.
          call_fn: A call function.
          name: Name of the call function.
        """
        self.call_collection = call_collection
        self.wrapped_call = tf.function(
            layer_call_wrapper(call_collection, call_fn, name)
        )

    def _maybe_trace(self, args, kwargs):
        # Trigger traces of other call functions + extra training-arg traces.
        if tracing_enabled():
            self.call_collection.add_trace(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self._maybe_trace(args, kwargs)
        return self.wrapped_call(*args, **kwargs)

    def get_concrete_function(self, *args, **kwargs):
        self._maybe_trace(args, kwargs)
        return self.wrapped_call.get_concrete_function(*args, **kwargs)