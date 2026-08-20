"""Load, decode, and override typed JSON configuration trees.

The companion JSON file stores constructor-like objects using this format:

    {"__class__": "ClassName", "args": {"field": value}}

Tuples, slices, enum values, callables, and dictionaries with non-string keys
are tagged so their Python types survive a JSON round trip.

Without registries, load_config_json returns ConfigNode/CallableRef/EnumRef
objects. To instantiate the actual project configuration classes, pass maps
from the names in the JSON to the imported classes/functions/enums.
"""

from __future__ import annotations

import copy
import importlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class CallableRef:
    """A callable name that could not be resolved from a callable registry."""

    name: str


@dataclass(frozen=True)
class EnumRef:
    """An enum value that could not be resolved from an enum registry."""

    type_name: str
    name: str
    value: Any


@dataclass(frozen=True)
class SliceRef:
    """A slice retained as a value when decoding without a project class."""

    start: Any = None
    stop: Any = None
    step: Any = None

    def to_slice(self) -> slice:
        return slice(self.start, self.stop, self.step)


@dataclass
class ConfigNode:
    """Fallback representation for a typed config with no registered class."""

    type_name: str
    fields: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.fields[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, name: str) -> Any:
        return self.fields[name]

    def __repr__(self) -> str:
        return f"{self.type_name}({self.fields!r})"


def _import_symbol(path: str) -> Any:
    """Resolve a dotted path such as ``package.module:Symbol``."""
    if ":" in path:
        module_name, symbol_name = path.split(":", 1)
    else:
        module_name, separator, symbol_name = path.rpartition(".")
        if not separator:
            raise ValueError(f"Cannot import symbol without a module path: {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _lookup(registry: Mapping[str, Any] | None, key: str) -> Any | None:
    if not registry or key not in registry:
        return None
    value = registry[key]
    return _import_symbol(value) if isinstance(value, str) else value


def decode_config_value(
    value: Any,
    *,
    class_registry: Mapping[str, Any] | None = None,
    callable_registry: Mapping[str, Any] | None = None,
    enum_registry: Mapping[str, Any] | None = None,
) -> Any:
    """Decode one value from the tagged JSON representation."""
    if isinstance(value, list):
        return [
            decode_config_value(
                item,
                class_registry=class_registry,
                callable_registry=callable_registry,
                enum_registry=enum_registry,
            )
            for item in value
        ]

    if not isinstance(value, dict):
        return value

    if "__tuple__" in value:
        return tuple(
            decode_config_value(
                item,
                class_registry=class_registry,
                callable_registry=callable_registry,
                enum_registry=enum_registry,
            )
            for item in value["__tuple__"]
        )

    if "__slice__" in value:
        start, stop, step = value["__slice__"]
        decoded = [
            decode_config_value(
                item,
                class_registry=class_registry,
                callable_registry=callable_registry,
                enum_registry=enum_registry,
            )
            for item in (start, stop, step)
        ]
        return slice(*decoded)

    if "__callable__" in value:
        name = value["__callable__"]
        resolved = _lookup(callable_registry, name)
        return resolved if resolved is not None else CallableRef(name)

    if "__enum__" in value:
        enum_info = value["__enum__"]
        type_name = enum_info["type"]
        member_name = enum_info["name"]
        enum_cls = _lookup(enum_registry, type_name)
        if enum_cls is not None:
            if issubclass(enum_cls, Enum):
                try:
                    return enum_cls[member_name]
                except KeyError:
                    return enum_cls(enum_info["value"])
        return EnumRef(type_name, member_name, enum_info.get("value"))

    if "__dict__" in value:
        return {
            decode_config_value(
                key,
                class_registry=class_registry,
                callable_registry=callable_registry,
                enum_registry=enum_registry,
            ): decode_config_value(
                item,
                class_registry=class_registry,
                callable_registry=callable_registry,
                enum_registry=enum_registry,
            )
            for key, item in value["__dict__"]
        }

    if "__class__" in value:
        type_name = value["__class__"]
        args = decode_config_value(
            value.get("args", {}),
            class_registry=class_registry,
            callable_registry=callable_registry,
            enum_registry=enum_registry,
        )
        cls = _lookup(class_registry, type_name)
        if cls is not None:
            if isinstance(args, dict):
                return cls(**args)
            return cls(args)
        return ConfigNode(type_name=type_name, fields=args)

    return {
        key: decode_config_value(
            item,
            class_registry=class_registry,
            callable_registry=callable_registry,
            enum_registry=enum_registry,
        )
        for key, item in value.items()
    }


def load_config_json(
    path: str | Path,
    *,
    class_registry: Mapping[str, Any] | None = None,
    callable_registry: Mapping[str, Any] | None = None,
    enum_registry: Mapping[str, Any] | None = None,
) -> Any:
    """Load one JSON configuration file and decode its tagged values."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return decode_config_value(
        raw,
        class_registry=class_registry,
        callable_registry=callable_registry,
        enum_registry=enum_registry,
    )


def _merge_raw(base: Any, override: Any) -> Any:
    """Recursively merge mappings; lists and tagged scalar values are replaced.

    Typed nodes accept both a typed override (``{"args": {...}}``) and a
    concise override (``{"field": value}``).
    """
    if not isinstance(base, dict) or not isinstance(override, dict):
        return copy.deepcopy(override)

    # Tagged values represent atomic Python values. Replace them as a whole.
    tagged_keys = {"__tuple__", "__slice__", "__callable__", "__enum__", "__dict__"}
    if tagged_keys.intersection(base) or tagged_keys.intersection(override):
        return copy.deepcopy(override)

    # A typed config node is merged through its args payload. This is what
    # makes both forms below work at every nesting level:
    #   {"args": {"agent": {"args": {"...": ...}}}}
    #   {"agent": {"...": ...}}
    if "__class__" in base:
        if "__class__" in override and override["__class__"] != base["__class__"]:
            return copy.deepcopy(override)

        if "args" in override:
            override_args = override["args"]
            extra_fields = {
                key: value
                for key, value in override.items()
                if key not in {"__class__", "args"}
            }
            if extra_fields:
                override_args = _merge_raw(override_args, extra_fields)
        else:
            override_args = {
                key: value for key, value in override.items() if key != "__class__"
            }

        result = copy.deepcopy(base)
        result["args"] = _merge_raw(base.get("args", {}), override_args)
        return result

    # Replacing a plain node with a typed node is intentional.
    if "__class__" in override:
        return copy.deepcopy(override)

    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = _merge_raw(result[key], value) if key in result else copy.deepcopy(value)
    return result


def merge_config_data(base_data: Any, override_data: Any) -> Any:
    """Return a merged raw JSON tree without mutating either input tree."""
    return _merge_raw(base_data, override_data)


def load_config_pair(
    baseline_path: str | Path,
    override_path: str | Path,
    *,
    output_path: str | Path | None = None,
    class_registry: Mapping[str, Any] | None = None,
    callable_registry: Mapping[str, Any] | None = None,
    enum_registry: Mapping[str, Any] | None = None,
) -> Any:
    """Load baseline + override JSON, optionally save the merged JSON, then decode it.

    The override file can contain only the fields that should change. For a
    typed root, either of these forms is valid:

        {"args": {"agent": {"args": {"algorithm": {"args": {"learning_rate": 0.0003}}}}}}

    or, more conveniently, a partial tree without type wrappers:

        {"agent": {"algorithm": {"learning_rate": 0.0003}}}

    Lists and tagged values, such as tuples, replace the corresponding baseline
    value rather than merging element by element.
    """
    with Path(baseline_path).open("r", encoding="utf-8") as handle:
        baseline = json.load(handle)
    with Path(override_path).open("r", encoding="utf-8") as handle:
        override = json.load(handle)

    merged = merge_config_data(baseline, override)
    if output_path is not None:
        with Path(output_path).open("w", encoding="utf-8") as handle:
            json.dump(merged, handle, indent=2)
            handle.write("\n")

    return decode_config_value(
        merged,
        class_registry=class_registry,
        callable_registry=callable_registry,
        enum_registry=enum_registry,
    )


# A descriptive alias for callers who prefer an explicit function name.
load_baseline_with_overrides = load_config_pair


__all__ = [
    "CallableRef",
    "ConfigNode",
    "EnumRef",
    "SliceRef",
    "decode_config_value",
    "load_config_json",
    "merge_config_data",
    "load_config_pair",
    "load_baseline_with_overrides",
]
