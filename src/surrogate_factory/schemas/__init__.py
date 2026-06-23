"""
 Copyright (c) 2025 Airbus Operations S. L. This file is part of project Surrogate Factory released under the Airbus Inner Source shared-maintenance
 """

from . import default


def schema_from_yaml_file(stage: str, yaml_path: str) -> dict:
    """Build a minimal JSON schema by inferring types from an existing YAML metadata file."""
    import yaml

    def _infer_schema(value) -> dict:
        if isinstance(value, dict):
            return {
                "type": "object",
                "properties": {k: _infer_schema(v) for k, v in value.items()},
            }
        elif isinstance(value, list):
            return {"type": "array", "items": {"type": "string"}}
        elif isinstance(value, bool):
            return {"type": "boolean"}
        elif isinstance(value, int):
            return {"type": "integer"}
        elif isinstance(value, float):
            return {"type": "number"}
        else:
            return {"type": "string"}

    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}
        stage_data = data.get(stage, data)
        return _infer_schema(stage_data)
    except Exception:
        return {"type": "object", "properties": {}}
