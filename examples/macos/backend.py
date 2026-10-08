"""Shared backend selection; importing a route does not load its native helper."""
import argparse
from agxforge.runtime.macos import source_identity


def select_backend(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    _backend_options(parser)
    args, remaining = parser.parse_known_args(argv)
    if args.backend == "iogpu":
        from agxforge.runtime.iogpu import Executor, EXECUTOR_NAME
    else:
        from agxforge.runtime.macos import Executor, EXECUTOR_NAME
    return args.backend, Executor, EXECUTOR_NAME, remaining


def select_compiler(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    _compiler_options(parser)
    args, remaining = parser.parse_known_args(argv)
    if args.mesa_compiler and args.compiler != "mesa":
        parser.error("--mesa-compiler requires --compiler mesa")
    if args.compiler == "mesa":
        from agxforge.g13 import mesa
        def factory(output):
            return mesa.Compiler(output, compiler=args.mesa_compiler or mesa.DEFAULT_COMPILER)
    else: factory = None
    return factory, remaining


def _backend_options(parser):
    parser.add_argument("--backend", choices=("metal", "iogpu"), default="metal",
                        help="Metal carrier (default) or experimental direct IOGPU")


def _compiler_options(parser):
    parser.add_argument("--compiler", choices=("g13", "mesa"), default="g13",
                        help="AGXForge G13 (default) or Mesa NIR/AGX code generation")
    parser.add_argument("--mesa-compiler", help="optional path to the built Mesa adapter")


def add_backend_options(parser):
    _backend_options(parser)
    _compiler_options(parser)
