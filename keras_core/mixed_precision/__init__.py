from keras_core.mixed_precision.dtype_policy import DTypePolicy
from keras_core.mixed_precision.dtype_policy import dtype_policy
from keras_core.mixed_precision.dtype_policy import set_dtype_policy
from keras_core.saving import serialization_lib


def resolve_policy(identifier):
    if identifier is None:
        return dtype_policy()
    if isinstance(identifier, DTypePolicy):
        return identifier
    if isinstance(identifier, str):
        return DTypePolicy(identifier)
    if isinstance(identifier, dict):
        return serialization_lib.deserialize_keras_object(identifier)
    raise ValueError(
        "Cannot interpret `dtype` argument. Expected a string "
        f"or an instance of DTypePolicy. Received: dtype={identifier}"
    )


def serialize_policy(policy):
    """Serializes `DTypePolicy` instances."""
    policy = resolve_policy(policy)
    return serialization_lib.serialize_keras_object(policy)


def deserialize_policy(config, custom_objects=None):
    """Deserializes a config to a `DTypePolicy` instance."""
    module_objects = {"DTypePolicy": DTypePolicy}
    policy = serialization_lib.deserialize_keras_object(
        config,
        module_objects=module_objects,
        custom_objects=custom_objects,
    )
    return resolve_policy(policy)