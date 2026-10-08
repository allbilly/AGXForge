#!/usr/bin/env python3
"""Check retained code, compiler provenance, launches and scalar GPU outputs.

Full model tensor arrays are omitted. Their bounds were recomputed by the
recorded full-bundle audits; this verifies their retained reports and hashes.
"""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from agxforge.g13.abi import Binding, G13Program
from agxforge.g13.mesa import from_binary
from agxforge.g13.metal import code_range


def digest(data): return hashlib.sha256(data).hexdigest()
def read(path): return json.loads(path.read_text())
def unpack(path): return gzip.decompress(path.read_bytes())
def artifact(folder, name):
    path = folder/name
    return path.read_bytes() if path.exists() else unpack(folder/(name+'.gz'))


def launch_check(launch, compiler_descriptors, executor, files, folder):
    p = launch['program']
    if (p not in compiler_descriptors or launch['status'] != 'COMPLETED' or
            launch['executor'] != executor or not launch['fence_completed'] or
            not launch['canaries_intact'] or launch['logical_threads'] != p['logical_threads']):
        raise ValueError('launch/compiler/completion contract differs')
    if files[folder+'/shader.bin'] != p['code_sha256']:
        raise ValueError('submitted shader differs from compiler')
    states = ('metal-archive.bin',) if executor == 'G13 machine code through Metal' else (
        'uniforms.bin', 'usc.bin', 'cdm.bin', 'iogpu-submit.bin', 'iogpu-pages.bin')
    for name in states:
        if files[folder+'/'+name] != launch[name]['sha256']:
            raise ValueError('launch state differs from audited image')
    if executor == 'direct IOGPU G13':
        completion = bytes.fromhex(launch['completion'])
        tokens = launch['notification_tokens']
        if len(completion) != 80 or len(tokens) != 2 or not all(tokens) or tokens[0] == tokens[1]:
            raise ValueError('missing native notifications')
        for index, token in enumerate(tokens):
            cookie, start, end, status, reserved = struct.unpack_from('<5Q', completion, index*40)
            if cookie != token or not start or end < start or status or reserved:
                raise ValueError('failed native notification')
    return states


def scalar_check(folder, row):
    actual, expected = artifact(folder, 'slot-0-after.bin'), artifact(folder, 'expected.bin')
    if not actual or not expected: raise ValueError('missing scalar output')
    formats = dict(float32='f', float64='d', uint32='I', int32='i', uint16='H')
    if not row['atol'] and not row['rtol']:
        if row['actual_dtype'] != row['expected_dtype']:
            values = [v[0] for v in struct.iter_unpack('<'+formats[row['actual_dtype']], actual)]
            actual = struct.pack('<'+formats[row['expected_dtype']]*len(values), *values)
        if actual != expected: raise ValueError('exact scalar output differs')
        return
    a = [v[0] for v in struct.iter_unpack('<'+formats[row['actual_dtype']], actual)]
    e = [v[0] for v in struct.iter_unpack('<'+formats[row['expected_dtype']], expected)]
    if len(a) != len(e): raise ValueError('scalar extent differs')
    for value, reference in zip(a, e):
        if math.isfinite(reference):
            if not math.isfinite(value) or abs(value-reference) > row['atol']+row['rtol']*abs(reference):
                raise ValueError('scalar result exceeds numerical bounds')
        elif value != reference: raise ValueError('scalar nonfinite result differs')


def verify():
    index = read(ROOT/'artifact-sha256.json')
    actual = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in ROOT.rglob('*')
              if p.is_file() and p.name != 'artifact-sha256.json'}
    if actual != index: raise ValueError('compact artifacts changed or are missing')
    receipt, build = read(ROOT/'receipt.json'), read(ROOT/'mesa-build-identity.json')
    if set(receipt['runs']) != {f'{suite}-{backend}' for suite in ('kernels', 'gpt2', 'qwen')
                               for backend in ('metal', 'native')}:
        raise ValueError('six-run verification scope differs')
    if (receipt['status'] != 'PASS' or receipt['throughput_benchmarked'] or
            build['compiler'] != 'Mesa 26.2.4 AGX' or not build['archive_verified'] or
            not build['compiler_sources_verified'] or
            build['pinned_archive_sha256'] != 'bce5f7fbebb934373b86c999a064d52fb5065878dc57f287f95346648ec832e9' or
            build['adapter_sha256'] != build['sources']['agxforge-bridge/bridge.c']):
        raise ValueError('compiler provenance/scope differs')
    build_hash = digest((ROOT/'mesa-build-identity.json').read_bytes())
    compilations = {}
    for folder in sorted((ROOT/'compilations').iterdir()):
        identity = read(folder/'compiler-identity.json')
        if (folder.name != digest((folder/'compiler-identity.json').read_bytes()) or
                identity['compiler_sha256'] != build['compiler_sha256'] or
                identity['build_identity_sha256'] != build_hash or identity['bridge_protocol_version'] != 2 or
                identity['protocol_sha256'] != digest((folder/'input.ir').read_bytes())):
            raise ValueError('compilation receipt differs')
        d = identity['descriptor']
        fields = {key: d[key] for key in G13Program.__dataclass_fields__ if key != 'code'}
        fields['bindings'] = tuple(Binding(**b) for b in fields['bindings'])
        for key in ('builtins', 'reserved_register_halfs'): fields[key] = tuple(fields[key])
        p = G13Program(code=(folder/'shader.bin').read_bytes(), **fields)
        admitted = from_binary(p, p.code, read(folder/'metadata.json'))
        if (p.code_hash != identity['code_sha256'] or p.descriptor() != admitted.descriptor() or
                json.loads(json.dumps(p.descriptor())) != d or d['origin'] != 'Mesa 26.2.4 NIR -> AGX'):
            raise ValueError('native code/ABI differs')
        compilations[str(folder.relative_to(ROOT))] = identity
    total_launches, total_tensors = 0, 0
    sources, models = [], {}
    for name, entry in receipt['runs'].items():
        folder = ROOT/name
        summary, platform = read(folder/'summary.json'), read(folder/'platform.json')
        source = read(folder/'source-sha256.json'); sources.append(source)
        executor = 'direct IOGPU G13' if name.endswith('native') else 'G13 machine code through Metal'
        if (summary['status'] != 'PASS' or summary['compiler'] != receipt['compiler'] or
                platform['executor'] != executor or platform['gpu'] != 'Apple M1' or
                platform['build'] != '26A434' or platform['macos'] != '27.0.1' or
                source['tools/mesa_agx/bridge.c'] != build['adapter_sha256']):
            raise ValueError('run compiler/platform/source differs')
        if name.endswith('native') and (platform['metal_api_calls'] or not platform['agx_metal_absent']):
            raise ValueError('native execution used Metal')
        audit_bytes = unpack(folder/'full-audit.json.gz')
        audit = json.loads(audit_bytes); files = audit['files']
        if (digest(audit_bytes) != entry['full_audit_sha256'] or audit['status'] != 'PASS' or
                audit['launches'] != entry['launches'] or audit['tensor_checks'] != entry['tensor_checks'] or
                len(files) != entry['artifact_hashes']): raise ValueError('full audit identity differs')
        for file in ('summary.json', 'platform.json', 'source-sha256.json', 'plan.json', 'bounds.json', 'checkpoint.json'):
            if (folder/file).exists() and digest((folder/file).read_bytes()) != files[file]:
                raise ValueError('compact root differs from full bundle')
        compiler_index = read(folder/'compiler-index.json')
        if len(compiler_index) != entry['compilation_records']: raise ValueError('missing compilations')
        plan = read(folder/'plan.json') if (folder/'plan.json').exists() else summary
        provenance = plan['compiler_provenance']['programs']
        descriptors = []
        for kernel, path in compiler_index.items():
            identity = compilations[path]
            if provenance[kernel] != dict(receipt=f'compiler/{kernel}/compiler-identity.json', **identity):
                raise ValueError('plan differs from compiler receipts')
            for file in ('input.ir', 'shader.bin', 'metadata.json', 'compiler-identity.json', 'shader.nir', 'compiler.log'):
                if digest((ROOT/path/file).read_bytes()) != files[f'compiler/{kernel}/{file}']:
                    raise ValueError('compact compiler output differs from full bundle')
            descriptors.append(identity['descriptor'])
        if 'verified' in summary:
            launches = json.loads(unpack(folder/'launch-receipts.json.gz'))
            reports = json.loads(unpack(folder/'tensor-checks.json.gz'))
            target = (12, 3264, 2508, 21) if name.startswith('gpt2') else (6, 3780, 2754, 22)
            if (not summary['verified'] or summary['tensor_fallbacks'] or summary['executor'] != executor or
                    (summary['completed_positions'], summary['dispatches'], entry['tensor_checks'], len(descriptors)) != target or
                    len(summary['checks']) != target[0] or len(reports) != target[0] or len(launches) != target[1] or
                    Counter(json.dumps(d, sort_keys=True) for d in plan['programs'].values()) !=
                    Counter(json.dumps(d, sort_keys=True) for d in descriptors)):
                raise ValueError('model coverage/fallback/plan differs')
            tensors, count = {}, 0
            for position in summary['checks']:
                path = f"token-{position['position']:03d}/checks.json"
                data = reports[path]
                if digest(data.encode()) != files[path]: raise ValueError('tensor reports differ from full audit')
                rows = json.loads(data)
                if position['status'] != 'PASS' or not position['top_token_matches'] or len(rows) != position['tensors']:
                    raise ValueError('model tensor/argmax coverage differs')
                for row in rows:
                    tensor_path = path.removesuffix('checks.json')+row['tensor']+'.f32'
                    if row['status'] != 'PASS' or row['actual_sha256'] != files[tensor_path]:
                        raise ValueError('tensor output differs from full audit')
                    tensors[(path, row['tensor'])] = row['actual_sha256']; count += 1
            if count != target[2]: raise ValueError('missing model tensors')
            predictions = [p['next_token'] for p in summary['checks'][len(summary['input_tokens'])-1:]]
            if predictions != summary['generated_tokens']: raise ValueError('generated tokens differ')
            models[name] = (summary, plan, tensors)
        else:
            launches = {str(p.relative_to(folder)): p.read_text() for p in folder.glob('[0-9]*/launch.json')}
            rows = []
            if len(launches) != entry['kernel_checks'] or entry['kernel_checks'] != 112:
                raise ValueError('kernel coverage differs')
            for path, text in launches.items():
                local = (folder/path).parent; row = read(local/'check.json'); rows.append(row)
                launch = json.loads(text)
                if (row['status'] != 'PASS' or not row['fence_completed'] or not row['canaries_intact'] or
                        row['logical_threads'] != launch['logical_threads']):
                    raise ValueError('scalar correctness/completion receipt differs')
                states = launch_check(launch, descriptors, executor, files, str(Path(path).parent))
                if len(artifact(local, 'slot-0-after.bin')) != launch['program']['bindings'][0]['min_bytes']:
                    raise ValueError('scalar output extent differs')
                for file in (*states, 'shader.bin', 'slot-0-after.bin', 'expected.bin', 'check.json'):
                    if digest(artifact(local, file)) != files[str(Path(path).parent/file)]:
                        raise ValueError('retained scalar artifacts differ from full audit')
                scalar_check(local, row)
                if executor == 'G13 machine code through Metal':
                    archive = artifact(local, 'metal-archive.bin'); offset, capacity = code_range(archive)
                    code = artifact(local, 'shader.bin')
                    if (offset != launch['code_offset'] or capacity != launch['code_capacity'] or
                            archive[offset:offset+len(code)] != code): raise ValueError('carrier placement differs')
                else:
                    addresses = {b['binding']: b['gpu_address'] for b in launch['buffers']}
                    pointers = [addresses[b['name']] for b in sorted(launch['program']['bindings'], key=lambda b: b['slot'])]
                    if artifact(local, 'uniforms.bin') != struct.pack('<'+'Q'*len(pointers), *pointers):
                        raise ValueError('native buffer pointers differ')
            if Counter(json.dumps(r, sort_keys=True) for r in rows) != Counter(json.dumps(r, sort_keys=True) for r in summary['checks']):
                raise ValueError('kernel checks differ from summary')
        for path, text in launches.items():
            if digest(text.encode()) != files[path]: raise ValueError('launch receipt differs from full audit')
            launch_check(json.loads(text), descriptors, executor, files, str(Path(path).parent))
        total_launches += len(launches); total_tensors += entry['tensor_checks']
    if any(source != sources[0] for source in sources): raise ValueError('run source cohort differs')
    cross = read(ROOT/'cross-backend.json')
    for model in ('gpt2', 'qwen'):
        metal, native = models[model+'-metal'], models[model+'-native']
        if (metal[0]['generated_tokens'] != native[0]['generated_tokens'] or metal[1:] != native[1:] or
                cross[model] != dict(status='PASS', tensors_bit_identical=len(metal[2]), program_descriptors_identical=True)):
            raise ValueError('cross-transport tensor/compiler comparison differs')
    regression = read(ROOT/'compiler-regression.json')
    if (regression['status'] != 'PASS' or not regression['authored_codegen_blocked'] or
            regression['default_descriptors_unchanged'] != 43 or regression['independently_mesa_compiled'] != 43 or
            sum(len(p) for p in regression['default_programs'].values()) != 43 or
            regression['mesa_programs']['GPT2Plan'] != models['gpt2-metal'][1]['programs'] or
            regression['mesa_programs']['QwenPlan'] != models['qwen-metal'][1]['programs'] or
            'Ran 87 tests' not in (ROOT/'unit-checks.log').read_text()):
        raise ValueError('compiler independence/default regression differs')
    print(f'PASS: {len(index)} hashes, {len(compilations)} compiler outputs, {total_launches} launches, '
          f'{total_tensors} model tensor reports, 224 scalar output checks, 43 unchanged default programs')


if __name__ == '__main__': verify()
