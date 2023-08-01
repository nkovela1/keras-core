import tensorflow as tf
import contextlib

def get_tensor_spec(t, dynamic_batch=False, name=None):
    """Returns a `TensorSpec` given a single `Tensor` or `TensorSpec`."""
    if isinstance(t, tf.TypeSpec):
        spec = t
    elif isinstance(tensor, tf.__internal__.CompositeTensor):
        # Check for ExtensionTypes
        spec = t._type_spec
    elif hasattr(t, "shape") and hasattr(t, "dtype"):
        spec = tf.TensorSpec(shape=t.shape, dtype=t.dtype, name=name)
    else:
        return None  # Allow non-Tensors to pass through.

    if not dynamic_batch:
        return spec

    shape = spec.shape
    if shape.rank is None or shape.rank == 0:
        return spec

    shape_list = shape.as_list()
    shape_list[0] = None
    shape = tf.TensorShape(shape_list)
    spec._shape = shape
    return spec


@contextlib.contextmanager
def graph_context_for_symbolic_tensors(*args, **kwargs):
    """Returns graph context manager if any of the inputs is a symbolic
    tensor."""
    if any(is_symbolic_tensor(v) for v in list(args) + list(kwargs.values())):
        from keras_core.legacy.backend import get_graph
        with get_graph().as_default():
            yield
    else:
        yield