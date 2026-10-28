"""Execute improved notebooks in fresh kernels, preserving original notebooks."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', help='Notebook stems or filenames to execute.')
    args = parser.parse_args()
    for name, relative in {'MPLCONFIGDIR': 'matplotlib', 'IPYTHONDIR': 'ipython',
                           'JUPYTER_CONFIG_DIR': 'jupyter/config',
                           'JUPYTER_RUNTIME_DIR': 'jupyter/runtime'}.items():
        path = ROOT / 'work' / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))
    # Keep linear algebra deterministic and modest on ordinary laptops.
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ.setdefault(name, '1')
    import nbformat
    from nbclient import NotebookClient
    from hospital_ml.data import sha256, write_json
    import pandas, sklearn, numpy
    versions = {'python': platform.python_version(), 'platform': platform.system(),
                'pandas': pandas.__version__, 'scikit_learn': sklearn.__version__, 'numpy': numpy.__version__}
    notebooks = sorted((ROOT / 'notebooks').glob('*.ipynb'))
    if args.only:
        wanted = set(args.only)
        notebooks = [path for path in notebooks if path.name in wanted or path.stem in wanted]
        if len(notebooks) != len(wanted):
            raise ValueError('One or more requested notebooks were not found.')
    records = []
    # A partial rerun must retain valid evidence for the untouched notebooks.
    manifest_path = ROOT / 'reports/execution_manifest.json'
    if args.only and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(key) != value for key, value in versions.items()):
            raise ValueError('Environment changed; run all notebooks to refresh execution evidence.')
        rerun_paths = {str(path.relative_to(ROOT)) for path in notebooks}
        for record in previous['notebooks']:
            if record['notebook'] not in rerun_paths:
                if sha256(ROOT / record['notebook']) != record['sha256']:
                    raise ValueError('An unselected notebook changed; run all notebooks.')
                records.append(record)
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=1200, kernel_name='python3',
                                resources={'metadata': {'path': str(ROOT)}})
        # NotebookClient starts a fresh kernel for every notebook; no hidden state.
        client.execute()
        code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
        errors = [output for cell in code_cells for output in cell.get('outputs', []) if output.output_type == 'error']
        if errors:
            raise RuntimeError(f'{path.name} contains execution errors.')
        if [cell.execution_count for cell in code_cells] != list(range(1, len(code_cells) + 1)):
            raise RuntimeError(f'Nonsequential execution in {path.name}.')
        nbformat.write(notebook, path)
        records.append({'notebook': str(path.relative_to(ROOT)), 'executed_code_cells': len(code_cells),
                        'errors': 0, 'sha256': sha256(path)})
        print(f'Executed {path.name}: {len(code_cells)} code cells, zero errors', flush=True)
    write_json(manifest_path, {
        **versions,
        'raw_sha256': sha256(ROOT / 'data/raw/hospital_2012_2017.csv'),
        'execution': 'Fresh kernel per notebook; sequential code execution; no network/download/install cells',
        'notebooks': sorted(records, key=lambda record: record['notebook']),
    })


if __name__ == '__main__':
    main()
