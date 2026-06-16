"""Minimal CLI shim for swarm-engine."""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AetherForge Swarm — multi-agent orchestration")
    parser.add_argument("--version", action="version", version="aetherforge-swarm 1.0.0")
    parser.parse_args(argv)
    print("AetherForge Swarm CLI — use aetherforge swarm <subcommand>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
