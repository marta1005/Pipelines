"""
 Copyright (c) 2025 Airbus Operations S. L. This file is part of project Surrogate Factory released under the Airbus Inner Source shared-maintenance
 """

import importlib
import inspect
import logging


class Catalog:
    """Registry of callable methods organized by category."""

    def __init__(self, logger=None):
        self._methods = {}
        self.logger = logger or logging.getLogger(__name__)

    def load_config(self, config: dict):
        """Load methods from config dict: {category: {name: import_path}}."""
        for category, methods in config.items():
            if category not in self._methods:
                self._methods[category] = {}
            for name, import_path in methods.items():
                try:
                    module_path, attr = import_path.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    self._methods[category][name] = getattr(module, attr)
                    self.logger.debug(f"Catalog: loaded '{name}' from '{import_path}'")
                except Exception as e:
                    self.logger.warning(f"Catalog: could not load '{name}' ({import_path}): {e}")
                    self._methods[category][name] = import_path

    def get_method_names(self, category_key: str) -> list:
        """Return method names for a given category."""
        return list(self._methods.get(category_key, {}).keys())

    def get_all_method_names(self) -> list:
        """Return all method names across all categories, deduplicated."""
        names = []
        for methods in self._methods.values():
            names.extend(methods.keys())
        return list(dict.fromkeys(names))

    def get_method_template(self, category_key: str, method_name: str) -> dict:
        """Return a default params template inferred from a method's signature."""
        method = self._methods.get(category_key, {}).get(method_name)
        if callable(method):
            try:
                sig = inspect.signature(method)
                return {
                    k: (v.default if v.default is not inspect.Parameter.empty else None)
                    for k, v in sig.parameters.items()
                    if k not in ("self", "cls")
                }
            except (ValueError, TypeError):
                pass
        return {}

    def get_method_arguments(self, category_key: str, method_name: str) -> list:
        """Return argument names for a method."""
        method = self._methods.get(category_key, {}).get(method_name)
        if callable(method):
            try:
                sig = inspect.signature(method)
                return [k for k in sig.parameters if k not in ("self", "cls")]
            except (ValueError, TypeError):
                pass
        return []

    def get_method(self, method_name: str):
        """Return a callable by name, searching across all categories."""
        for methods in self._methods.values():
            if method_name in methods:
                return methods[method_name]
        raise KeyError(f"Method '{method_name}' not found in catalog. Available: {self.get_all_method_names()}")

    @property
    def methods(self) -> dict:
        """Return flat dict of all registered methods {name: callable}."""
        result = {}
        for methods in self._methods.values():
            result.update(methods)
        return result
