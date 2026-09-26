#!/usr/bin/env python3
"""Compare generic and model-sized linked images, with optional exact S3 replay."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from production_check import footprint
from benchmark_qemu import Firmware, BUILD
from transport import stable


def inspect(build, specialized):
    build = Path(build).resolve()
    result = footprint(build)
    project = json.loads((build / 'project_description.json').read_text())
    prefix = project['c_compiler'][:-3]
    symbols = subprocess.check_output([prefix + 'nm', '-S', str(build / 'ruvector_decision.elf')], text=True)
    result['static_symbols'] = {parts[3]: int(parts[1], 16) for line in symbols.splitlines()
                                if len(parts := line.split()) == 4 and parts[2] in ('b', 'B')
                                and parts[3] in ('workspace', 'ctx')}
    if set(result['static_symbols']) != {'workspace', 'ctx'}:
        raise ValueError('workspace/context symbols missing')
    commands = json.loads((build / 'compile_commands.json').read_text())
    consumers = {}
    for entry in commands:
        name = Path(entry['file']).name
        if name not in ('rvdecision.c', 'app.c', 'profile.c', 'sensor.c', 'main.c'):
            continue
        if '/esp32-decision/' not in entry['file']:
            continue
        command = entry['command']
        capacities = dict(re.findall(r'-D(RD_MAX_(?:DIMS|CLASSES))=(\d+)', command))
        if specialized and set(capacities) != {'RD_MAX_DIMS', 'RD_MAX_CLASSES'}:
            raise ValueError('missing PUBLIC capacity definitions: ' + name)
        if not specialized and capacities:
            raise ValueError('generic image unexpectedly specialized')
        consumers[name] = capacities
    if set(consumers) != {'rvdecision.c', 'app.c', 'profile.c', 'sensor.c', 'main.c'}:
        raise ValueError('not all ABI consumers verified')
    if len({json.dumps(v, sort_keys=True) for v in consumers.values()}) != 1:
        raise ValueError('inconsistent ABI capacities')
    result['consumer_capacities'] = consumers
    if specialized:
        includes = project['build_component_info']['main']['include_dirs']
        main_dir = Path(project['build_component_info']['main']['dir'])
        headers = [(Path(p) if Path(p).is_absolute() else main_dir / p) / 'model.h' for p in includes]
        header = next(p.resolve() for p in headers if p.is_file())
        rules = (build / 'build.ninja').read_text().splitlines()
        regenerate = next(line for line in rules if line.startswith('build build.ninja:'))
        if str(header) not in regenerate:
            raise ValueError('model changes do not trigger CMake reconfiguration')
        result['model_configure_dependency_verified'] = True
    return result


def replay(generic, compact, qemu, vectors):
    rows = json.loads(Path(vectors).read_text())
    if not rows:
        raise ValueError('empty replay')
    BUILD.mkdir(exist_ok=True)
    results = {}; answers = {}
    for name, build in [('generic', generic), ('compact', compact)]:
        fw = Firmware(qemu, Path(build) / 'merged-binary.bin', 'workspace-' + name)
        try:
            meta = fw.response()
            if not meta['selftest_pass'] or meta['target'] != 'esp32s3':
                raise ValueError('S3 boot gate failed')
            command = lambda r: 'sensor ' + ' '.join(map(str, r['raw']))
            for row in rows:
                if 'error' in fw.query(command(row)):
                    raise ValueError('warmup failed')
            if fw.query('energy 16').get('runs') != 16:
                raise ValueError('energy warmup failed')
            before = fw.query('meta')['free_heap']
            answers[name] = [stable(fw.query(command(row))) for row in rows]
            if any('error' in answer for answer in answers[name]):
                raise ValueError('valid input failed')
            if fw.query('energy 256').get('runs') != 256 or not fw.query('selftest')['selftest_pass']:
                raise ValueError('post replay command failed')
            after = fw.query('meta')['free_heap']
            if before != after:
                raise ValueError('warmed heap changed')
            results[name] = {'meta': meta, 'heap_before': before, 'heap_after': after,
                             'initial_heap_growth': meta['free_heap'] - before,
                             'warmup_rows': len(rows), 'replay_rows': len(rows),
                             'answers_sha256': hashlib.sha256(json.dumps(answers[name], sort_keys=True).encode()).hexdigest()}
        finally:
            fw.close()
    if answers['generic'] != answers['compact']:
        raise ValueError('exact replay failed')
    for key in ('model_sha256', 'kernel_sha256', 'quant_bits', 'cpu_hz', 'core', 'profile_capacity'):
        if results['generic']['meta'][key] != results['compact']['meta'][key]:
            raise ValueError('identity mismatch: ' + key)
    results['exact_replay_match'] = True
    results['free_heap_recovered'] = results['compact']['heap_after'] - results['generic']['heap_after']
    if results['free_heap_recovered'] <= 0:
        raise ValueError('heap recovery missing')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generic', type=Path, required=True)
    parser.add_argument('--compact', type=Path, required=True)
    parser.add_argument('--qemu')
    parser.add_argument('--vectors', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.qemu and not args.vectors:
        parser.error('--qemu requires --vectors')
    report = {'execution': 'linked ELF analysis', 'physical_hardware': False,
              'generic': inspect(args.generic, False), 'compact': inspect(args.compact, True)}
    for key in ('model_sha256', 'target', 'profile_buffer_bytes'):
        if report['generic'][key] != report['compact'][key]:
            raise ValueError('paired identity mismatch: ' + key)
    report['static_symbol_bytes_recovered'] = (sum(report['generic']['static_symbols'].values()) -
                                               sum(report['compact']['static_symbols'].values()))
    if report['static_symbol_bytes_recovered'] <= 0:
        raise ValueError('no static memory improvement')
    report['application_bytes_recovered'] = report['generic']['application_bytes'] - report['compact']['application_bytes']
    if args.qemu:
        report['emulator'] = replay(args.generic, args.compact, args.qemu, args.vectors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('generic', 'compact', 'emulator')}))


if __name__ == '__main__':
    main()
