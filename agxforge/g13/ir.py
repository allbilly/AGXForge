"""Shared scalar SSA objects plus G13 math conveniences, without altering G17."""
from agxforge.g17.ir import (Buffer, Function, Block, Op, Value, Imm,
                            I16, I32, F16, F32, ELEM_BYTES, Builder as ScalarBuilder)


class Builder(ScalarBuilder):
    def fsub(self, a, b, type=I32, name=None):
        return self._def("fsub", [a, b], type, name)

    def bitcast(self, a, type=I32, name=None):
        return self._def("bitcast", [a], type, name)

    def rcp(self, a, type=I32, name=None):
        return self._def("rcp", [a], type, name)
