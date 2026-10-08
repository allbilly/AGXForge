import json
from pathlib import Path
from unittest.mock import patch
from agxforge.g13 import cc, mesa
from agxforge.g13.qwen import Checkpoint, QwenPlan
from agxforge.g13.gpt2 import GPT2Plan
root = Path('/Users/yeren/AGXForge')
expected = json.loads((root/'build/macos/mesa-default-programs-before.json').read_text())
actual = {}
independent = {}
for plan_type, checkpoint_path in ((QwenPlan, 'models/asahi/qwen2.5-0.5b'), (GPT2Plan, 'models/macos/gpt2')):
    checkpoint = Checkpoint(root/checkpoint_path)
    plan = plan_type(checkpoint)
    actual[plan_type.__name__] = {name: program.descriptor() for name, program in plan.programs.items()}
    with patch.object(cc, 'compile_function', side_effect=AssertionError('authored G13 codegen called')):
        compiler = mesa.Compiler(root/'build/macos/mesa-final-independent-plans'/plan_type.__name__)
        separate = plan_type(checkpoint, compiler=compiler)
    independent[plan_type.__name__] = {name: program.descriptor() for name, program in separate.programs.items()}
actual = json.loads(json.dumps(actual))
if actual != expected: raise AssertionError('default program descriptors changed')
result = dict(status='PASS', default_descriptors_unchanged=43, authored_codegen_blocked=True,
              independently_mesa_compiled=43, default_programs=actual, mesa_programs=independent)
(root/'build/macos/mesa-compiler-regression.json').write_text(json.dumps(result, indent=2)+'\n')
print('PASS: 43 default program descriptors unchanged; 43 Mesa programs compiled with authored G13 codegen blocked')
