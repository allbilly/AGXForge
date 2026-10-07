"""Fail if the unverified native path tries to invoke CPU reference helpers."""
from argparse import Namespace
from pathlib import Path
import sys
root=Path('/home/asahi/AGXForge')
sys.path.insert(0,str(root))
sys.path.insert(0,str(root/'examples/asahi'))
import qwen

def refused(*args,**kwargs):
    raise AssertionError('CPU tensor reference reached from unverified execution')
qwen.Reference=refused
qwen.Checkpoint.floats=refused
report=qwen.run(Namespace(checkpoint=root/'models/asahi/qwen2.5-0.5b',
    output=root/'results/asahi-model-no-reference', token_ids='9707',
    prompt='', generate=1, capacity=32, verify=False, va_slot=11))
assert report['status']=='COMPLETED_UNVERIFIED'
assert report['dispatches']==630 and report['tensor_fallbacks']==[]
assert report['checks'][0]['next_token']==271
print('Native execution completed with CPU tensor reference helpers disabled')
