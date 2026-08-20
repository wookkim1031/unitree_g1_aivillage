from __future__ import annotations

import copy
import importlib
import numbers
from collections.abc import Mapping, MutableMapping
from dataclasses import is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from utils import TrainConfig


def _format_path(path: tuple[object, ...]) -> str:
    return ".".join(str(part) for part in path) or "<root>"


def _is_config_object(value: Any) -> bool:
    """Return True for nested config objects, but not for mappings or callables."""
    return (
        not isinstance(value, (str, bytes, Path, Enum, Mapping))
        and not callable(value)
        and (is_dataclass(value) or hasattr(value, "__dict__"))
    )


def _copy_untyped(value: Any) -> Any:
    """Copy values inserted into previously empty dictionaries."""
    if isinstance(value, Mapping):
        return {
            copy.deepcopy(key): _copy_untyped(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_copy_untyped(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_copy_untyped(item) for item in value)

    return copy.deepcopy(value)


def _load_symbol(spec: str) -> Any:
    """
    Load a callable from a YAML string such as:

        my_package.rewards:custom_reward
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise TypeError(
            "Callable overrides must use 'module.path:attribute', "
            f"got {spec!r}"
        )

    module_name, attribute_path = spec.split(":", 1)
    value = importlib.import_module(module_name)

    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)

    if not callable(value):
        raise TypeError(f"Resolved YAML symbol is not callable: {spec!r}")

    return value


def _coerce_like(
    incoming: Any,
    current: Any,
    path: tuple[object, ...],
) -> Any:
    """
    Convert YAML values to the same broad container/type shape as the
    existing configuration value.
    """
    field_path = _format_path(path)

    # Explicit null is allowed for optional fields such as cnn_cfg,
    # terrain_generator, frictionloss, etc.
    if incoming is None:
        return None

    # Function fields can be overridden with "module:attribute".
    if callable(current):
        return _load_symbol(incoming)

    # Nested config object assigned as a complete mapping.
    if _is_config_object(current):
        if not isinstance(incoming, Mapping):
            raise TypeError(
                f"{field_path} must be a mapping when overriding a config object"
            )

        result = copy.deepcopy(current)
        return _merge_object(result, incoming, path)

    # Enum values may be written either by enum value or enum member name.
    if isinstance(current, Enum):
        if not isinstance(incoming, str):
            raise TypeError(f"{field_path} must be a string enum value")

        enum_type = type(current)

        try:
            return enum_type(incoming)
        except ValueError:
            try:
                return enum_type[incoming]
            except KeyError as exc:
                valid = [member.name for member in enum_type]
                raise ValueError(
                    f"Invalid value {incoming!r} for {field_path}. "
                    f"Expected one of: {valid}"
                ) from exc

    # Preserve tuple-valued configuration fields.
    if isinstance(current, tuple):
        if not isinstance(incoming, (list, tuple)):
            raise TypeError(f"{field_path} must be a YAML sequence")

        incoming_values = list(incoming)

        # For fixed-size tuples, coerce each position using the existing
        # element type when possible.
        if len(current) == len(incoming_values):
            return tuple(
                _coerce_like(value, old_value, path + (index,))
                for index, (value, old_value) in enumerate(
                    zip(incoming_values, current)
                )
            )

        # For homogeneous tuples such as (0.3, 1.2), use the first
        # element as the type template.
        if len(current) == 1:
            return tuple(
                _coerce_like(value, current[0], path + (index,))
                for index, value in enumerate(incoming_values)
            )

        return tuple(copy.deepcopy(incoming_values))

    # Preserve list-valued fields such as gpu_ids and wandb_tags.
    if isinstance(current, list):
        if not isinstance(incoming, (list, tuple)):
            raise TypeError(f"{field_path} must be a YAML sequence")

        if current:
            return [
                _coerce_like(value, current[0], path + (index,))
                for index, value in enumerate(incoming)
            ]

        return [_copy_untyped(value) for value in incoming]

    # Recursively merge dictionary-valued fields.
    if isinstance(current, Mapping):
        if not isinstance(incoming, Mapping):
            raise TypeError(f"{field_path} must be a YAML mapping")

        result = copy.deepcopy(current)

        if not isinstance(result, MutableMapping):
            result = dict(result)

        return _merge_mapping(result, incoming, path)

    # Handle common scalar types and catch accidental type mistakes.
    if isinstance(current, bool):
        if not isinstance(incoming, bool):
            raise TypeError(f"{field_path} must be a boolean")
        return incoming

    if isinstance(current, int) and not isinstance(current, bool):
        if (
            isinstance(incoming, numbers.Real)
            and not isinstance(incoming, bool)
            and float(incoming).is_integer()
        ):
            return int(incoming)

        raise TypeError(f"{field_path} must be an integer")

    if isinstance(current, float):
        if isinstance(incoming, numbers.Real) and not isinstance(incoming, bool):
            return float(incoming)

        raise TypeError(f"{field_path} must be numeric")

    if isinstance(current, str):
        if not isinstance(incoming, str):
            raise TypeError(f"{field_path} must be a string")
        return incoming

    if isinstance(current, Path):
        if not isinstance(incoming, str):
            raise TypeError(f"{field_path} must be a path string")
        return Path(incoming)

    # For fields whose current value is None or a custom scalar type,
    # retain the YAML value.
    return copy.deepcopy(incoming)


def _merge_mapping(
    target: MutableMapping[Any, Any],
    patch: Mapping[Any, Any],
    path: tuple[object, ...],
) -> MutableMapping[Any, Any]:
    """
    Recursively update a dictionary.

    Unknown keys are rejected in non-empty dictionaries. Empty dictionaries
    such as curriculum, metrics, and recorders may receive new entries.
    """
    for key, incoming in patch.items():
        key_path = path + (key,)

        if key not in target:
            # Empty maps are extension points in the supplied config.
            if len(target) == 0:
                target[key] = _copy_untyped(incoming)
                continue

            raise KeyError(
                f"Unknown YAML key: {_format_path(key_path)}"
            )

        current = target[key]

        if isinstance(incoming, Mapping) and _is_config_object(current):
            _merge_object(current, incoming, key_path)

        elif isinstance(incoming, Mapping) and isinstance(current, Mapping):
            updated = _merge_mapping(
                copy.deepcopy(current),
                incoming,
                key_path,
            )
            target[key] = updated

        else:
            target[key] = _coerce_like(
                incoming,
                current,
                key_path,
            )

    return target


def _merge_object(
    target: Any,
    patch: Mapping[str, Any],
    path: tuple[object, ...],
) -> Any:
    """Recursively apply a YAML mapping to a config object."""
    if not isinstance(patch, Mapping):
        raise TypeError(
            f"{_format_path(path)} must be a YAML mapping"
        )

    for key, incoming in patch.items():
        if not isinstance(key, str):
            raise TypeError(
                f"Object field names must be strings; "
                f"got {key!r} at {_format_path(path)}"
            )

        key_path = path + (key,)

        if not hasattr(target, key):
            raise KeyError(
                f"Unknown YAML key: {_format_path(key_path)}"
            )

        current = getattr(target, key)

        if isinstance(incoming, Mapping) and _is_config_object(current):
            _merge_object(current, incoming, key_path)

        elif isinstance(incoming, Mapping) and isinstance(current, Mapping):
            updated = _merge_mapping(
                copy.deepcopy(current),
                incoming,
                key_path,
            )
            setattr(target, key, updated)

        else:
            setattr(
                target,
                key,
                _coerce_like(incoming, current, key_path),
            )

    return target


def load_and_overwrite_train_config(t: TrainConfig, p: Path) -> TrainConfig:
    """
    Load a partial YAML configuration and apply it to a TrainConfig.

    Parameters
    ----------
    t:
        Existing base configuration.
    p:
        Path to a YAML override file.

    Returns
    -------
    TrainConfig
        A deep-copied configuration with YAML values applied.

    Notes
    -----
    - The input configuration is not mutated.
    - YAML paths follow the Python attribute hierarchy.
    - Missing YAML fields retain their original values.
    - Unknown fields raise KeyError.
    - Tuples are written as YAML lists and converted back to tuples.
    """
    p = Path(p)

    if not p.is_file():
        raise FileNotFoundError(f"YAML configuration not found: {p}")

    with p.open("r", encoding="utf-8") as file:
        patch = yaml.safe_load(file)

    if patch is None:
        patch = {}

    if not isinstance(patch, Mapping):
        raise TypeError(
            f"The YAML root must be a mapping, got {type(patch).__name__}"
        )

    result = copy.deepcopy(t)
    _merge_object(result, patch, ())

    return result