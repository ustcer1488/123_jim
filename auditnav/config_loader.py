#!/usr/bin/env python3
"""
config_loader.py — Shared configuration loader for all AuditNav nodes.

Usage in any node:
    from config_loader import load_config
    cfg = load_config()          # loads default_params.yaml from CWD/config/
    cfg = load_config("/path/to/params.yaml")   # explicit path
"""

import os
import yaml


_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "default_params.yaml"
)


def load_config(path: str = None) -> dict:
    """
    Load the YAML parameter file and return it as a nested dict.

    Resolution order:
      1. Explicit `path` argument
      2. Environment variable AUDITNAV_CONFIG
      3. <this_file's_dir>/config/default_params.yaml

    Parameters
    ----------
    path : str, optional
        Absolute or relative path to a YAML config file.

    Returns
    -------
    dict
        Nested configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If no config file can be located.
    """
    if path is None:
        path = os.environ.get("AUDITNAV_CONFIG", _DEFAULT_CONFIG_PATH)

    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"[AuditNav] Config file not found: {path}\n"
            f"  → Set AUDITNAV_CONFIG env var or pass --config <path>"
        )

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    # Inject API key from environment (never stored in YAML)
    api_key = os.environ.get("SILICONFLOW_API_KEY", "")
    if "api" not in cfg:
        cfg["api"] = {}
    cfg["api"]["key"] = api_key

    # Resolve data.base_dir relative to CWD
    base_dir = cfg.get("data", {}).get("base_dir", "data")
    base_dir = os.path.join(os.getcwd(), base_dir)
    os.makedirs(base_dir, exist_ok=True)
    cfg["data"]["base_dir"] = base_dir

    return cfg


def get(cfg: dict, *keys, default=None):
    """
    Safe nested key access.

    Example
    -------
    speed = get(cfg, "navigator", "base_speed", default=0.5)
    """
    node = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, None)
        if node is None:
            return default
    return node
