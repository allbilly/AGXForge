"""Shared backend selection; importing a route does not load its native helper."""
import argparse
from agxforge.runtime.macos import source_identity


def select_backend(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", choices=("metal", "iogpu"), default="metal",
                        help="Metal carrier (default) or experimental direct IOGPU")
    args, remaining = parser.parse_known_args(argv)
    if args.backend == "iogpu":
        from agxforge.runtime.iogpu import Executor, EXECUTOR_NAME
    else:
        from agxforge.runtime.macos import Executor, EXECUTOR_NAME
    return args.backend, Executor, EXECUTOR_NAME, remaining
