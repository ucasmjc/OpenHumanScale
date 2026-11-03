#!/usr/bin/env python3
"""Check source closure, hashes, and portable resource paths without importing ML libraries."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    errors = []
    modules = {}
    namespaces = set()
    sources = sorted((root / 'src').rglob('*.py'))
    for path in sources:
        parts = list(path.relative_to(root / 'src').with_suffix('').parts)
        if parts[-1] == '__init__':
            parts.pop()
        module = '.'.join(parts)
        modules[module] = path
        namespaces.update('.'.join(parts[:i]) for i in range(1, len(parts)))
    for path in [*sources, *sorted((root / 'scripts').glob('*.py'))]:
        try:
            tree = ast.parse(path.read_bytes())
        except SyntaxError as exc:
            errors.append(f'{path.relative_to(root)}: {exc}')
            continue
        if path not in modules.values():
            continue
        module = next(k for k, v in modules.items() if v == path)
        package = module if path.name == '__init__.py' else module.rsplit('.', 1)[0]
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    base = importlib.util.resolve_name('.' * node.level + base, package)
                names = [base]
                # Resolve relative `from . import module` references as well.
                if node.module is None:
                    names += [base + '.' + a.name for a in node.names]
            for name in names:
                if name.split('.')[0] in ('fastwam', 'openhumanscale') and name not in modules and name not in namespaces:
                    errors.append(f'{path.relative_to(root)}: missing local import {name}')
    manifest = json.loads((root / 'release/source_manifest.json').read_text())
    for row in manifest['files']:
        path = root / row['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['exported_sha256']:
            errors.append(f'Source differs from export manifest: {row["path"]}')
    # Inspect tracked candidates only; local bindings and generated outputs are private.
    candidates = [*sources, *sorted((root / 'configs').rglob('*.yaml'))]
    for path in candidates:
        if path.name == 'paths.local.yaml':
            continue
        if re.search(r'/mnt/(?:pfs|bos)/|/home/ss-oss/|/Users/|192\.168\.\d+\.\d+', path.read_text()):
            errors.append(f'Internal infrastructure reference: {path.relative_to(root)}')
    hf = json.loads((root / 'release/huggingface.json').read_text())
    if hf['namespace'] != 'DAGroup-PKU' or len(hf['collections']) != 1:
        errors.append('HF layout must use one collection under DAGroup-PKU')
    return dict(source_files=len(sources), copied_source_files=len(manifest['files']),
                reference_profiles=len(manifest['profiles']), errors=errors,
                static_checks_passed=not errors, runtime_verified=False)


if __name__ == '__main__':
    result = check()
    print(json.dumps(result, indent=2))
    sys.exit(1 if result['errors'] else 0)
