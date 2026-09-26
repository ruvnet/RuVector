"""Check specialization at ABI, protocol and build metadata boundaries."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from e2e import ROOT, CORE, INCLUDE
from quantize import export
from transport import stable


def compile_model(folder, model, rows, bits, specialized, profile='portable'):
    folder.mkdir(parents=True, exist_ok=True)
    export(model, folder / 'model.h', rows, bits)
    flags = []
    if specialized:
        flags += [f'-DRD_MAX_DIMS={model["dims"]}', f'-DRD_MAX_CLASSES={len(model["labels"])}']
    if profile != 'portable':
        flags += ['-DRD_PAIR_DOT']
    if profile == 'esp32s3':
        flags += ['-DRD_LIBM_ROUND']
    binary = folder / ('compact' if specialized else 'generic')
    subprocess.run(['gcc', '-std=c11', '-O3', '-fno-fast-math', '-ffp-contract=off',
                    '-Wall', '-Wextra', '-Werror', *flags, '-I', str(folder), '-I', str(INCLUDE),
                    '-I', str(ROOT / 'main'), str(CORE), str(ROOT / 'main/app.c'),
                    str(ROOT / 'main/profile.c'), str(ROOT / 'main/sensor.c'),
                    str(ROOT / 'tests/host_main.c'), '-lm', '-o', str(binary)], check=True)
    return binary


def replay(binary, rows, sensor=False):
    commands = [('sensor ' if sensor else 'infer ') + ' '.join(map(str, r['raw' if sensor else 'features']))
                for r in rows]
    commands += ['infer nan', 'infer 0', 'selftest']
    process = subprocess.run([str(binary)], input='\n'.join(commands) + '\n',
                             text=True, capture_output=True, check=True, timeout=60)
    replies = [json.loads(line) for line in process.stdout.splitlines()]
    if len(replies) != len(commands) + 1 or not replies[0]['selftest_pass']:
        raise AssertionError('boot or reply count failed')
    if any('error' in reply for reply in replies[1:1 + len(rows)]):
        raise AssertionError('valid input rejected')
    if not all('error' in reply for reply in replies[-3:-1]) or not replies[-1]['selftest_pass']:
        raise AssertionError('malformed input or recovery failed')
    return replies[0], [stable(reply) for reply in replies[1:]]


class WorkspaceTests(unittest.TestCase):
    def test_exact_decisions_for_all_heads_precisions_and_kernel_profiles(self):
        fixtures = ROOT / 'build-host/fixtures'
        names = [('prototype', 16), ('probe', 16), ('score', 16), ('logistic', 16),
                 ('similarity', 16), ('wide', 16), ('prototype', 8), ('similarity', 8)]
        with tempfile.TemporaryDirectory() as temporary:
            for name, bits in names:
                model = json.loads((fixtures / f'{name}.json').read_text())
                rows = json.loads((fixtures / f'{name}.vectors.json').read_text())
                for profile in ('portable', 'esp32s3', 'esp32c6'):
                    with self.subTest(name=name, bits=bits, profile=profile):
                        folder = Path(temporary) / f'{name}-{bits}-{profile}'
                        generic = replay(compile_model(folder, model, rows, bits, False, profile), rows)
                        compact = replay(compile_model(folder, model, rows, bits, True, profile), rows)
                        self.assertEqual(generic[1], compact[1])
                        self.assertEqual(generic[0]['model_sha256'], compact[0]['model_sha256'])
                        self.assertLessEqual(compact[0]['workspace_bytes'], generic[0]['workspace_bytes'])

    def test_sensor_replay_and_memory_reduction(self):
        model = json.loads((ROOT / 'build-sensor/snapshot.json').read_text())
        rows = json.loads((ROOT / 'build-sensor/validation.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            generic = replay(compile_model(folder, model, rows, 8, False), rows, True)
            compact = replay(compile_model(folder, model, rows, 8, True), rows, True)
            self.assertEqual(generic[1], compact[1])
            self.assertEqual(generic[0]['model_sha256'], compact[0]['model_sha256'])
            self.assertEqual(generic[0]['workspace_bytes'] - compact[0]['workspace_bytes'], 1692)
            self.assertGreater(generic[0]['static_app_bytes'], compact[0]['static_app_bytes'])

    def test_invalid_capacities_fail_compilation(self):
        for dims, classes in [(0, 2), (769, 2), (5, 0), (5, 17)]:
            with self.subTest(dims=dims, classes=classes):
                result = subprocess.run(['gcc', '-x', 'c', '-fsyntax-only', '-I', str(INCLUDE),
                                         f'-DRD_MAX_DIMS={dims}', f'-DRD_MAX_CLASSES={classes}', '-'],
                                        input='#include "rvdecision.h"\n', text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)

    def test_model_larger_than_capacity_rejected_before_access(self):
        source = '''#include "rvdecision.h"
#include <assert.h>
int main(void) {
    rd_context ctx;
    const int8_t values[6] = {1,0,0,0,0,0};
    rd_row rows[3] = {{values,5,1.f},{values,5,1.f},{values,5,1.f}};
    rd_model m = {.version=RD_VERSION,.quant_bits=8,.dims=5,.classes=2,
        .kind=RD_CHOICE,.head=RD_PROTOTYPE,.prototypes=rows,.prototype_count=2,
        .abstain_scale=.5f,.logit_scale=1.f,.temperature=.1f};
    assert(rd_init(&ctx,&m,.6f,.4f)==RD_OK);
    m.dims=6;
    for (int i=0;i<3;++i) rows[i].length=6;
    assert(rd_init(&ctx,&m,.6f,.4f)==RD_BAD_MODEL);
    assert(ctx.model==NULL);
    m.dims=5; m.classes=3; m.prototype_count=3;
    for (int i=0;i<3;++i) rows[i].length=5;
    assert(rd_init(&ctx,&m,.6f,.4f)==RD_BAD_MODEL);
    assert(ctx.model==NULL);
    return 0;
}'''
        with tempfile.TemporaryDirectory() as temporary:
            src = Path(temporary) / 'bounds.c'; src.write_text(source)
            binary = Path(temporary) / 'bounds'
            subprocess.run(['gcc', '-std=c11', '-DRD_MAX_DIMS=5', '-DRD_MAX_CLASSES=2',
                            '-fsanitize=address,undefined', '-I', str(INCLUDE), str(CORE), str(src),
                            '-lm', '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_cmake_literal_metadata_and_legacy_fallback(self):
        helper = ROOT / 'components/rvdecision/model_capacity.cmake'
        cases = [('#define RD_MODEL_DIMS 5\n#define RD_MODEL_CLASSES 2\n', '5;2'),
                 ('#define RD_MODEL_DIMS 32\n', '32;16'),
                 ('#define RD_MODEL_DIMS 768\n#define RD_MODEL_CLASSES 16\n', '768;16'),
                 ('#define RD_MODEL_DIMS 0\n', None),
                 ('#define RD_MODEL_DIMS 769\n', None),
                 ('#define RD_MODEL_DIMS 5\n#define RD_MODEL_CLASSES 17\n', None),
                 ('#define RD_MODEL_DIMS (5)\n', None),
                 ('#define RD_MODEL_DIMS 5\n#define RD_MODEL_DIMS 6\n', None),
                 ('#define RD_MODEL_CLASSES 2\n', None)]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for contents, expected in cases:
                with self.subTest(contents=contents):
                    (folder / 'model.h').write_text(contents)
                    script = folder / 'check.cmake'
                    script.write_text(f'include("{helper}")\nrd_model_capacity("{folder}/model.h" D C)\n'
                                      f'file(WRITE "{folder}/result" "${{D}};${{C}}")\n')
                    result = subprocess.run(['cmake', '-P', str(script)], capture_output=True, text=True)
                    if expected is None:
                        self.assertNotEqual(result.returncode, 0)
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual((folder / 'result').read_text(), expected)


if __name__ == '__main__':
    unittest.main()
