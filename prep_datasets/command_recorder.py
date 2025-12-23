#!/usr/bin/env python3
"""
Redis/Valkey Command Recorder

Generic command recorder that accepts ANY Redis/Valkey command and generates
schema YAML + binary data files for replay by valkey-bench-rs.

Design Philosophy:
- Works like a Redis client - record any valid command
- User declares field types explicitly (no magic inference)
- Supports vectors, strings, numbers, tags, and binary data
- Developers write Python to generate custom datasets

Usage:
    from command_recorder import CommandRecorder, Vector, Tag, Text, Numeric
    import numpy as np

    rec = CommandRecorder(name="product_dataset")

    # Declare schema upfront (recommended)
    rec.declare_field("embedding", "vector", dim=128, dtype="float32")
    rec.declare_field("category", "tag", max_bytes=32)
    rec.declare_field("price", "numeric", dtype="float64")
    rec.declare_field("description", "text", max_bytes=256)

    # Record commands
    for i in range(1000):
        vec = np.random.randn(128).astype(np.float32)
        rec.record("HSET", f"product:{i}",
                   "embedding", vec,
                   "category", "electronics",
                   "price", 99.99,
                   "description", "A great product")

    # Generate output
    rec.generate("datasets/products")

Alternative - inline type wrappers:
    rec = CommandRecorder()
    rec.record("HSET", "vec:1",
               "embedding", Vector(vec_array),
               "category", Tag("electronics"),
               "price", Numeric(99.99),
               "name", Text("Widget"))
"""

import yaml
import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from collections import defaultdict
from enum import Enum, auto


# === Type Wrappers for Explicit Typing ===

class TypedValue:
    """Base class for explicitly typed values."""
    pass


@dataclass
class Vector(TypedValue):
    """Vector field value."""
    value: np.ndarray
    dtype: str = "float32"

    def __post_init__(self):
        if not isinstance(self.value, np.ndarray):
            self.value = np.array(self.value, dtype=np.float32)
        if self.dtype == "float32":
            self.value = self.value.astype(np.float32)
        elif self.dtype == "float16":
            self.value = self.value.astype(np.float16)

    @property
    def dim(self) -> int:
        return len(self.value)


@dataclass
class Tag(TypedValue):
    """Tag field value (for filtering)."""
    value: str

    def __post_init__(self):
        self.value = str(self.value)


@dataclass
class Text(TypedValue):
    """Text field value."""
    value: str

    def __post_init__(self):
        self.value = str(self.value)


@dataclass
class Numeric(TypedValue):
    """Numeric field value."""
    value: Union[int, float]
    dtype: str = "float64"

    def __post_init__(self):
        if self.dtype in ("int32", "int64", "u32", "u64"):
            self.value = int(self.value)
        else:
            self.value = float(self.value)


@dataclass
class Blob(TypedValue):
    """Binary blob field value."""
    value: bytes

    def __post_init__(self):
        if not isinstance(self.value, bytes):
            self.value = bytes(self.value)


# === Field Definition ===

@dataclass
class FieldDef:
    """Schema field definition."""
    name: str
    field_type: str  # vector, text, tag, numeric, blob
    dtype: Optional[str] = None
    dim: Optional[int] = None
    max_bytes: Optional[int] = None
    distance_metric: Optional[str] = None  # For vectors: l2, cosine, ip

    def byte_size(self) -> int:
        """Compute byte size for this field."""
        if self.field_type == 'vector':
            dtype_sizes = {'float32': 4, 'float16': 2, 'uint8': 1, 'int8': 1}
            return self.dim * dtype_sizes.get(self.dtype, 4)
        elif self.field_type in ('text', 'tag', 'blob'):
            return self.max_bytes
        elif self.field_type == 'numeric':
            dtype_sizes = {'int32': 4, 'int64': 8, 'float32': 4, 'float64': 8, 'u32': 4, 'u64': 8}
            return dtype_sizes.get(self.dtype, 8)
        raise ValueError(f"Unknown field type: {self.field_type}")

    def to_yaml_dict(self) -> Dict[str, Any]:
        """Convert to YAML-compatible dict."""
        d = {'name': self.name, 'type': self.field_type}
        if self.dtype:
            d['dtype'] = self.dtype
        if self.dim:
            d['dimensions'] = self.dim
        if self.max_bytes:
            d['max_bytes'] = self.max_bytes
        if self.field_type in ('text', 'tag'):
            d['encoding'] = 'utf8'
            d['length'] = 'fixed'
        return d


# === Recorded Command ===

@dataclass
class RecordedField:
    """A recorded field with value."""
    name: str
    value: Any
    field_def: Optional[FieldDef] = None


@dataclass
class RecordedCommand:
    """A recorded Redis command."""
    command: str
    key: str
    fields: List[RecordedField]


# === Main Recorder ===

class CommandRecorder:
    """
    Generic Redis/Valkey command recorder.

    Records any command and generates schema + binary files.
    """

    # Commands with field-value pair arguments (after key)
    HASH_COMMANDS = {'HSET', 'HMSET'}

    def __init__(self, name: str = "recorded_dataset"):
        self.name = name
        self.commands: List[RecordedCommand] = []
        self.keys: List[str] = []
        self.key_set: Set[str] = set()

        # Declared schema (field_name -> FieldDef)
        self._schema: Dict[str, FieldDef] = {}

        # Track field stats for auto-sizing (max lengths, etc.)
        self._field_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'max_len': 0,
            'count': 0,
        })

    # === Schema Declaration ===

    def declare_field(self, name: str, field_type: str, **kwargs) -> 'CommandRecorder':
        """
        Declare a field's type in the schema.

        Args:
            name: Field name
            field_type: One of: vector, text, tag, numeric, blob
            **kwargs: Type-specific options:
                - vector: dim (required), dtype="float32", distance_metric="l2"
                - text: max_bytes (optional, auto-sized if not given)
                - tag: max_bytes (optional, auto-sized if not given)
                - numeric: dtype="float64"
                - blob: max_bytes (required)

        Returns:
            self (for chaining)

        Examples:
            rec.declare_field("embedding", "vector", dim=128)
            rec.declare_field("category", "tag", max_bytes=32)
            rec.declare_field("price", "numeric", dtype="float64")
            rec.declare_field("name", "text", max_bytes=100)
        """
        if field_type == 'vector':
            if 'dim' not in kwargs:
                raise ValueError("Vector field requires 'dim' parameter")
            self._schema[name] = FieldDef(
                name=name,
                field_type='vector',
                dim=kwargs['dim'],
                dtype=kwargs.get('dtype', 'float32'),
                distance_metric=kwargs.get('distance_metric', 'l2')
            )
        elif field_type == 'text':
            self._schema[name] = FieldDef(
                name=name,
                field_type='text',
                max_bytes=kwargs.get('max_bytes')  # None = auto-size
            )
        elif field_type == 'tag':
            self._schema[name] = FieldDef(
                name=name,
                field_type='tag',
                max_bytes=kwargs.get('max_bytes')  # None = auto-size
            )
        elif field_type == 'numeric':
            self._schema[name] = FieldDef(
                name=name,
                field_type='numeric',
                dtype=kwargs.get('dtype', 'float64')
            )
        elif field_type == 'blob':
            if 'max_bytes' not in kwargs:
                raise ValueError("Blob field requires 'max_bytes' parameter")
            self._schema[name] = FieldDef(
                name=name,
                field_type='blob',
                max_bytes=kwargs['max_bytes']
            )
        else:
            raise ValueError(f"Unknown field type: {field_type}")

        return self

    # === Command Recording ===

    def record(self, command: str, key: str, *args) -> 'CommandRecorder':
        """
        Record a Redis/Valkey command.

        Args:
            command: The command name (e.g., "HSET", "SET")
            key: The key for this command
            *args: Field-value pairs for HSET, or single value for SET

        Returns:
            self (for chaining)

        Examples:
            # With declared schema
            rec.record("HSET", "vec:1", "embedding", vec, "category", "electronics")

            # With type wrappers
            rec.record("HSET", "vec:1",
                       "embedding", Vector(vec),
                       "category", Tag("electronics"))

            # Simple SET
            rec.record("SET", "key:1", "value", Text("hello"))
        """
        command = command.upper()

        # Track unique keys
        if key not in self.key_set:
            self.key_set.add(key)
            self.keys.append(key)

        # Parse field-value pairs
        if command in self.HASH_COMMANDS:
            if len(args) % 2 != 0:
                raise ValueError(f"{command} requires field-value pairs")

            fields = []
            for i in range(0, len(args), 2):
                field_name = str(args[i])
                field_value = args[i + 1]
                field_def = self._get_or_infer_field(field_name, field_value)
                fields.append(RecordedField(field_name, field_value, field_def))
                self._track_field_stats(field_name, field_value)
        else:
            # Generic command - treat remaining args as positional
            fields = []
            for i, arg in enumerate(args):
                field_name = f'_arg{i}'
                field_def = self._get_or_infer_field(field_name, arg)
                fields.append(RecordedField(field_name, arg, field_def))
                self._track_field_stats(field_name, arg)

        self.commands.append(RecordedCommand(command, key, fields))
        return self

    def _get_or_infer_field(self, name: str, value: Any) -> FieldDef:
        """Get declared field def or infer from value."""
        # Check if already declared
        if name in self._schema:
            return self._schema[name]

        # Infer from typed value
        if isinstance(value, Vector):
            field_def = FieldDef(
                name=name,
                field_type='vector',
                dim=value.dim,
                dtype=value.dtype,
                distance_metric='l2'
            )
        elif isinstance(value, Tag):
            field_def = FieldDef(
                name=name,
                field_type='tag',
                max_bytes=None  # Auto-size later
            )
        elif isinstance(value, Text):
            field_def = FieldDef(
                name=name,
                field_type='text',
                max_bytes=None
            )
        elif isinstance(value, Numeric):
            field_def = FieldDef(
                name=name,
                field_type='numeric',
                dtype=value.dtype
            )
        elif isinstance(value, Blob):
            field_def = FieldDef(
                name=name,
                field_type='blob',
                max_bytes=None
            )
        elif isinstance(value, np.ndarray):
            # Auto-detect as vector
            dtype = 'float32' if value.dtype == np.float32 else 'float16' if value.dtype == np.float16 else 'float32'
            field_def = FieldDef(
                name=name,
                field_type='vector',
                dim=len(value),
                dtype=dtype,
                distance_metric='l2'
            )
        elif isinstance(value, (int, float)):
            dtype = 'int64' if isinstance(value, int) else 'float64'
            field_def = FieldDef(
                name=name,
                field_type='numeric',
                dtype=dtype
            )
        elif isinstance(value, bytes):
            field_def = FieldDef(
                name=name,
                field_type='blob',
                max_bytes=None
            )
        else:
            # Default to text
            field_def = FieldDef(
                name=name,
                field_type='text',
                max_bytes=None
            )

        # Store in schema
        self._schema[name] = field_def
        return field_def

    def _track_field_stats(self, name: str, value: Any):
        """Track field statistics for auto-sizing."""
        stats = self._field_stats[name]
        stats['count'] += 1

        # Extract raw value from wrapper
        if isinstance(value, TypedValue):
            raw_value = value.value
        else:
            raw_value = value

        # Track max length for text/tag/blob
        if isinstance(raw_value, str):
            stats['max_len'] = max(stats['max_len'], len(raw_value.encode('utf-8')))
        elif isinstance(raw_value, bytes):
            stats['max_len'] = max(stats['max_len'], len(raw_value))

    def _finalize_schema(self):
        """Finalize schema with auto-sized fields."""
        for name, field_def in self._schema.items():
            if field_def.max_bytes is None and field_def.field_type in ('text', 'tag', 'blob'):
                stats = self._field_stats.get(name, {'max_len': 0})
                max_len = max(stats['max_len'], 32)  # Minimum 32 bytes
                # Round up to power of 2
                field_def.max_bytes = self._round_up_power2(max_len)

    def _round_up_power2(self, n: int) -> int:
        """Round up to next power of 2."""
        if n <= 0:
            return 1
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        return n + 1

    # === Generation ===

    def generate(self, output_base: Union[str, Path]) -> Tuple[Path, Path]:
        """
        Generate schema YAML and binary data files.

        Args:
            output_base: Base path without extension

        Returns:
            Tuple of (schema_path, data_path)
        """
        if not self.commands:
            raise ValueError("No commands recorded")

        output_base = Path(output_base)
        output_base.parent.mkdir(parents=True, exist_ok=True)

        schema_path = output_base.with_suffix('.yaml')
        data_path = output_base.with_suffix('.bin')

        # Finalize schema (auto-size fields)
        self._finalize_schema()

        # Determine primary command
        command_counts = defaultdict(int)
        for cmd in self.commands:
            command_counts[cmd.command] += 1
        primary_command = max(command_counts, key=command_counts.get)

        # Build and write schema
        schema = self._build_schema(primary_command)
        with open(schema_path, 'w') as f:
            yaml.dump(schema, f, default_flow_style=False, sort_keys=False)
        print(f"Schema written to {schema_path}")

        # Write binary data
        self._write_binary(schema, data_path, primary_command)
        print(f"Binary data written to {data_path}")

        return schema_path, data_path

    def _build_schema(self, primary_command: str) -> Dict[str, Any]:
        """Build schema dictionary."""
        # Get fields in consistent order
        field_names = sorted(self._schema.keys())
        fields = [self._schema[name] for name in field_names]

        # Compute max key length
        max_key_len = max(len(k.encode('utf-8')) for k in self.keys)
        max_key_len = self._round_up_power2(max(max_key_len, 32))

        # Build field metadata (distance metrics, etc.)
        field_metadata = {}
        for f in fields:
            if f.field_type == 'vector' and f.distance_metric:
                field_metadata[f.name] = {'distance_metric': f.distance_metric}
            elif f.field_type == 'tag':
                field_metadata[f.name] = {'index_type': 'tag'}

        schema = {
            'version': 1,
            'metadata': {
                'name': self.name,
                'description': f'Recorded dataset: {len(self.commands)} {primary_command} commands'
            },
            'replay': {
                'command': primary_command,
            },
            'record': {
                'fields': [f.to_yaml_dict() for f in fields]
            },
            'sections': {
                'records': {'count': len(self.keys)},
                'keys': {
                    'present': True,
                    'encoding': 'utf8',
                    'length': 'fixed',
                    'max_bytes': max_key_len
                }
            }
        }

        if field_metadata:
            schema['field_metadata'] = field_metadata

        return schema

    def _write_binary(self, schema: Dict, data_path: Path, primary_command: str):
        """Write binary data file."""
        fields = [self._schema[f['name']] for f in schema['record']['fields']]
        key_max_bytes = schema['sections']['keys']['max_bytes']
        record_size = sum(f.byte_size() for f in fields)

        # Build key -> field data mapping
        key_data: Dict[str, Dict[str, Any]] = defaultdict(dict)
        for cmd in self.commands:
            if cmd.command != primary_command:
                continue
            for rf in cmd.fields:
                key_data[cmd.key][rf.name] = rf.value

        with open(data_path, 'wb') as f:
            # Write records
            for key in self.keys:
                data = key_data.get(key, {})
                self._write_record(f, fields, data, record_size)

            # Write keys
            for key in self.keys:
                key_bytes = key.encode('utf-8')
                f.write(key_bytes[:key_max_bytes])
                padding = key_max_bytes - min(len(key_bytes), key_max_bytes)
                f.write(b'\x00' * padding)

    def _write_record(self, f, fields: List[FieldDef], data: Dict[str, Any], record_size: int):
        """Write a single record."""
        start_pos = f.tell()

        for field in fields:
            value = data.get(field.name)

            # Extract from wrapper
            if isinstance(value, TypedValue):
                value = value.value

            if field.field_type == 'vector':
                if value is not None:
                    if isinstance(value, np.ndarray):
                        arr = value.astype(np.float32 if field.dtype == 'float32' else np.float16)
                    else:
                        arr = np.array(value, dtype=np.float32)
                    f.write(arr.tobytes())
                else:
                    f.write(b'\x00' * field.byte_size())

            elif field.field_type in ('text', 'tag'):
                max_bytes = field.max_bytes
                if value is not None:
                    value_bytes = str(value).encode('utf-8')
                else:
                    value_bytes = b''
                f.write(value_bytes[:max_bytes])
                padding = max_bytes - min(len(value_bytes), max_bytes)
                f.write(b'\x00' * padding)

            elif field.field_type == 'blob':
                max_bytes = field.max_bytes
                value_bytes = value if isinstance(value, bytes) else b''
                f.write(value_bytes[:max_bytes])
                padding = max_bytes - min(len(value_bytes), max_bytes)
                f.write(b'\x00' * padding)

            elif field.field_type == 'numeric':
                fmt = {
                    'float64': '<d', 'float32': '<f',
                    'int64': '<q', 'int32': '<i',
                    'u64': '<Q', 'u32': '<I'
                }.get(field.dtype, '<d')
                f.write(struct.pack(fmt, value if value is not None else 0))

        # Ensure correct record size
        written = f.tell() - start_pos
        if written < record_size:
            f.write(b'\x00' * (record_size - written))

    def summary(self) -> str:
        """Return a summary of recorded data."""
        lines = [f"CommandRecorder: {self.name}"]
        lines.append(f"  Commands: {len(self.commands)}")
        lines.append(f"  Unique keys: {len(self.keys)}")
        lines.append(f"  Fields:")
        for name, field in sorted(self._schema.items()):
            if field.field_type == 'vector':
                lines.append(f"    {name}: vector[{field.dim}] ({field.dtype})")
            elif field.field_type in ('text', 'tag', 'blob'):
                lines.append(f"    {name}: {field.field_type} (max {field.max_bytes} bytes)")
            else:
                lines.append(f"    {name}: {field.field_type} ({field.dtype})")
        return '\n'.join(lines)


# === Convenience Functions ===

def record_vectors(name: str, key_prefix: str, vectors: np.ndarray,
                   vector_field: str = "embedding",
                   metadata: Optional[Dict[str, List[Any]]] = None,
                   metadata_types: Optional[Dict[str, str]] = None) -> CommandRecorder:
    """
    Convenience function to record vector HSET commands.

    Args:
        name: Dataset name
        key_prefix: Prefix for keys (e.g., "vec:")
        vectors: Numpy array of shape (N, dim)
        vector_field: Name of the vector field
        metadata: Dict of field_name -> list of values (one per vector)
        metadata_types: Dict of field_name -> type ("tag", "text", "numeric")

    Returns:
        Configured CommandRecorder
    """
    rec = CommandRecorder(name=name)

    # Declare vector field
    n_vectors, dim = vectors.shape
    rec.declare_field(vector_field, "vector", dim=dim, dtype="float32")

    # Declare metadata fields
    if metadata and metadata_types:
        for field_name, field_type in metadata_types.items():
            if field_type == "numeric":
                rec.declare_field(field_name, "numeric", dtype="float64")
            else:
                rec.declare_field(field_name, field_type)

    # Record commands
    for i in range(n_vectors):
        args = [vector_field, vectors[i]]
        if metadata:
            for field_name, values in metadata.items():
                args.extend([field_name, values[i]])
        rec.record("HSET", f"{key_prefix}{i:012d}", *args)

    return rec


# === Examples ===

def example_declared_schema():
    """Example: Using declared schema (recommended)."""
    rec = CommandRecorder(name="product_catalog")

    # Declare schema upfront
    rec.declare_field("embedding", "vector", dim=128, dtype="float32", distance_metric="cosine")
    rec.declare_field("category", "tag", max_bytes=32)
    rec.declare_field("price", "numeric", dtype="float64")
    rec.declare_field("name", "text", max_bytes=128)

    # Record data
    np.random.seed(42)
    categories = ["electronics", "clothing", "books", "home"]

    for i in range(100):
        vec = np.random.randn(128).astype(np.float32)
        rec.record("HSET", f"product:{i:06d}",
                   "embedding", vec,
                   "category", categories[i % 4],
                   "price", 10.0 + (i % 100),
                   "name", f"Product {i}")

    rec.generate("datasets/product_catalog")
    print(rec.summary())


def example_type_wrappers():
    """Example: Using type wrappers for inline type specification."""
    rec = CommandRecorder(name="user_vectors")

    np.random.seed(42)

    for i in range(100):
        vec = np.random.randn(64).astype(np.float32)
        rec.record("HSET", f"user:{i:06d}",
                   "embedding", Vector(vec),
                   "tier", Tag("premium" if i % 3 == 0 else "standard"),
                   "score", Numeric(i * 1.5),
                   "bio", Text(f"User biography for user {i}"))

    rec.generate("datasets/user_vectors")
    print(rec.summary())


def example_convenience():
    """Example: Using convenience function."""
    np.random.seed(42)

    vectors = np.random.randn(500, 256).astype(np.float32)
    categories = ["cat_a", "cat_b", "cat_c", "cat_d"]

    rec = record_vectors(
        name="quick_dataset",
        key_prefix="item:",
        vectors=vectors,
        vector_field="vec",
        metadata={
            "category": [categories[i % 4] for i in range(500)],
            "score": [float(i) for i in range(500)]
        },
        metadata_types={
            "category": "tag",
            "score": "numeric"
        }
    )

    rec.generate("datasets/quick_dataset")
    print(rec.summary())


if __name__ == '__main__':
    import sys

    examples = {
        'declared': example_declared_schema,
        'wrappers': example_type_wrappers,
        'convenience': example_convenience,
    }

    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example in examples:
            examples[example]()
        else:
            print(f"Unknown example: {example}")
            print(f"Available: {', '.join(examples.keys())}")
    else:
        print("Usage: python command_recorder.py <example>")
        print(f"Available examples: {', '.join(examples.keys())}")
        print("\nRunning all examples...")
        for name, fn in examples.items():
            print(f"\n=== {name} ===")
            fn()
