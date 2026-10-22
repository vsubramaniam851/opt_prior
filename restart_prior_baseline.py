import argparse
import fcntl
import json
import re
import shutil

from datetime import datetime, timezone
from pathlib import Path

def restart_plan(root, experiment):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', experiment):
        raise ValueError('Use an experiment name, not a path or a wildcard')
    root = Path(root).resolve()
    folder = root / 'logs' / experiment
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError('The experiment log directory must exist and not be a symlink')
    with (folder / 'args.json').open() as handle:
        args = json.load(handle)
    with (folder / 'prior_alignment.json').open() as handle:
        metadata = json.load(handle)
    if args.get('baseline') is not True or metadata.get('baseline') is not True:
        raise ValueError('This restart helper only archives baselines, never aligned initializations')
    if args.get('exp_name') != experiment or not args.get('optimization_prior') or metadata.get('version') != 2:
        raise ValueError('Experiment metadata does not describe the requested v2 baseline')
    final = root / 'saved_models' / f'ft_final_{experiment}.pt'
    if final.exists() or list(folder.glob('ft_results_seed*.json')):
        raise FileExistsError('This baseline has completion outputs; refusing to archive a completed result')
    paths = [folder, root / 'saved_models' / f'orig_{experiment}.pt', root / 'saved_models' / f'ft_{experiment}.pt',
             root / 'saved_models' / experiment]
    paths = [path for path in paths if path.exists() or path.is_symlink()]
    if any(path.is_symlink() for path in paths):
        raise ValueError('Refusing to move checkpoint or experiment symlinks')
    return root, paths

def restart_baseline(root, experiment, apply = False, confirm_stopped = False):
    root, paths = restart_plan(root, experiment)
    result = {'experiment': experiment, 'action': 'preview', 'paths': [str(path.relative_to(root)) for path in paths]}
    if not apply:
        return result
    if not confirm_stopped:
        raise ValueError('Check that all jobs writing this experiment have stopped, then use --confirm_stopped')
    locks = root / 'logs' / 'array_locks'
    locks.mkdir(parents = True, exist_ok = True)
    with (locks / f'{experiment}.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('An array launcher still holds this experiment lock; no files were moved')
        root, paths = restart_plan(root, experiment)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        backup = root / 'restart_archives' / experiment / stamp
        backup.mkdir(parents = True, exist_ok = False)
        result.update({'action': 'archiving', 'backup': str(backup), 'moved': []})
        manifest = backup / 'restart_manifest.json'
        manifest.write_text(json.dumps(result, indent = 2))
        try:
            for source in paths:
                relative = source.relative_to(root)
                target = backup / relative
                target.parent.mkdir(parents = True, exist_ok = True)
                shutil.move(str(source), str(target))
                result['moved'].append(str(relative))
                manifest.write_text(json.dumps(result, indent = 2))
        except BaseException:
            for relative in reversed(result['moved']):
                source = backup / relative
                target = root / relative
                if source.exists() and not target.exists():
                    shutil.move(str(source), str(target))
            result['action'] = 'rolled_back'
            manifest.write_text(json.dumps(result, indent = 2))
            raise
        result['action'] = 'archived'
        manifest.write_text(json.dumps(result, indent = 2))
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default = '.')
    parser.add_argument('--experiment', required = True)
    parser.add_argument('--apply', action = 'store_true')
    parser.add_argument('--confirm_stopped', action = 'store_true')
    args = parser.parse_args()
    print(json.dumps(restart_baseline(args.root, args.experiment, args.apply, args.confirm_stopped), indent = 2))

if __name__ == '__main__':
    main()
