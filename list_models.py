#!/usr/bin/env python3
"""List models from an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


def _load_config_endpoint(config_path: Path) -> str | None:
    if not config_path.exists():
        return None
    with config_path.open("rb") as f:
        config = tomllib.load(f)
    openai_section = config.get("openai")
    if isinstance(openai_section, dict):
        api_url = openai_section.get("api_url")
        if isinstance(api_url, str) and api_url.strip():
            return api_url.strip()
    return None


def _resolve_endpoint(cli_endpoint: str | None, config_path: Path) -> str:
    if cli_endpoint:
        return cli_endpoint
    env_endpoint = os.getenv("OPENAI_API_URL")
    if env_endpoint:
        return env_endpoint
    config_endpoint = _load_config_endpoint(config_path)
    if config_endpoint:
        return config_endpoint
    return "http://localhost:5000/v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List models available on an OpenAI-compatible endpoint."
    )
    parser.add_argument(
        "--endpoint",
        help="Base URL of the OpenAI-compatible endpoint (e.g. https://host/v1).",
    )
    parser.add_argument(
        "--key-env",
        default="LLM_API_KEY",
        help="Environment variable name for the API key (default: LLM_API_KEY).",
    )
    parser.add_argument(
        "--config",
        default="user.config.toml",
        help="Path to config file used for endpoint fallback (default: user.config.toml).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30).",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = _parse_args()

    api_key = os.getenv(args.key_env)
    if not api_key:
        print(f"Missing API key in environment variable: {args.key_env}", file=sys.stderr)
        return 1

    endpoint = _resolve_endpoint(args.endpoint, Path(args.config))
    client = OpenAI(base_url=endpoint, api_key=api_key, timeout=args.timeout)

    try:
        models_page = client.models.list()
        models_data: list[Any] = list(models_page.data)
    except Exception as exc:
        print(f"Failed to list models from {endpoint}: {exc}", file=sys.stderr)
        return 1

    model_ids = sorted(
        [getattr(model, "id", None) for model in models_data if getattr(model, "id", None)]
    )
    print(f"Endpoint: {endpoint}")
    print(f"Models found: {len(model_ids)}")
    for model_id in model_ids:
        print(model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
