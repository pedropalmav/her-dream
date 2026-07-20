"""Shared argparse setup for the experiment entry points."""

import argparse


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the `--logdir` / `--device` / `--seed` args common to every script.

    Scripts add their own bespoke args on top of these.
    """
    parser.add_argument("--logdir", required=True, help="Run dir with .hydra/config.yaml and latest.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser
