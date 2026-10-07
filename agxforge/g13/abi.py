"""Logical G13 scalar ABI v1. Register counts are in 16-bit halfwords."""
from dataclasses import dataclass, asdict
import hashlib
from .decode import validate


@dataclass(frozen=True)
class Binding:
    name: str
    slot: int
    access: str
    min_bytes: int
    alignment: int = 4


@dataclass(frozen=True)
class G13Program:
    name: str
    code: bytes
    bindings: tuple[Binding, ...]
    register_halfs: int
    uniform_halfs: int
    workgroup_size: int = 32
    entry_offset: int = 0
    target: str = "g13g"
    abi_version: int = 1
    reserved_register_halfs: tuple = (0, 1)
    builtins: tuple = ()
    bounds_policy: str = "complete_groups"
    numerical: str = "binary32; hardware FTZ and approximate transcendentals"
    origin: str = "handwritten"
    threadgroup_bytes: int = 0
    scratch_bytes: int = 0
    logical_threads: int = 0

    def __post_init__(self):
        if self.target != "g13g" or self.abi_version != 1 or self.entry_offset != 0:
            raise ValueError("unsupported G13 program target/version/entry")
        if type(self.code) is not bytes or not isinstance(self.bindings, tuple):
            raise ValueError("immutable shader bytes and bindings required")
        if not 1 <= self.register_halfs <= 248 or not 1 <= self.uniform_halfs <= 256:
            raise ValueError("register footprint outside no-spill scalar profile")
        if not 32 <= self.workgroup_size <= 1024 or self.workgroup_size % 32:
            raise ValueError("workgroup must contain complete SIMD groups")
        if self.scratch_bytes or self.threadgroup_bytes:
            raise ValueError("scratch/shared-memory execution is not established")
        slots = [b.slot for b in self.bindings]
        if len(set(slots)) != len(slots) or sorted(slots) != list(range(len(slots))):
            raise ValueError("bindings must have unique dense slots")
        if self.uniform_halfs != len(slots) * 4:
            raise ValueError("each buffer pointer requires four uniform halfwords")
        if self.bounds_policy not in ("complete_groups", "masked"):
            raise ValueError("unsupported bounds policy")
        if type(self.logical_threads) is not int or self.logical_threads <= 0:
            raise ValueError("a fixed logical launch extent is required")
        for b in self.bindings:
            if (not b.name or type(b.slot) is not int or type(b.min_bytes) is not int or
                b.access not in ("read", "write", "read_write") or b.min_bytes <= 0 or b.alignment not in (2, 4, 8, 16)):
                raise ValueError("invalid buffer binding")
        if len({b.name for b in self.bindings}) != len(self.bindings):
            raise ValueError("buffer names must be unique")
        if not isinstance(self.builtins, tuple) or not set(self.builtins) <= {0, 48, 52, 53, 80}:
            raise ValueError("unsupported builtin requirements")
        if not isinstance(self.reserved_register_halfs, tuple) or any(
                type(r) is not int or not 0 <= r < self.register_halfs for r in self.reserved_register_halfs):
            raise ValueError("invalid reserved registers")
        listing = validate(self.code, self.register_halfs, self.uniform_halfs)
        actual_builtins = {i.fields["SR"] for i in listing if i.mnemonic == "get_sr"}
        if set(self.builtins) != actual_builtins:
            raise ValueError("builtin declaration disagrees with shader")

    @property
    def code_hash(self):
        return hashlib.sha256(self.code).hexdigest()

    def descriptor(self):
        result = asdict(self)
        result.pop("code")
        result.update(code_sha256=self.code_hash, code_bytes=len(self.code),
                      constants=[], relocations=[], helper_required=False)
        return result
