from pathlib import Path
import json


PACKAGE_NAME = 'qcar2_autonomy'


def get_source_package_root():
    here = Path(__file__).resolve()

    for parent in here.parents:
        if parent.name == PACKAGE_NAME and (parent / 'package.xml').exists():
            return parent

    for parent in here.parents:
        if parent.name == 'install':
            workspace_root = parent.parent
            for candidate in (
                workspace_root / 'ros2' / 'src' / PACKAGE_NAME,
                workspace_root / 'src' / PACKAGE_NAME,
            ):
                if (candidate / 'package.xml').exists():
                    return candidate

    cwd = Path.cwd().resolve()
    search_roots = [cwd] + list(cwd.parents)
    for root in search_roots:
        for candidate in (
            root / 'src' / PACKAGE_NAME,
            root / 'Development' / 'ros2' / 'src' / PACKAGE_NAME,
        ):
            if (candidate / 'package.xml').exists():
                return candidate

    return here.parent.parent


def get_recording_maps_dir():
    maps_dir = get_source_package_root() / 'recording_maps'
    maps_dir.mkdir(parents=True, exist_ok=True)
    return maps_dir


def find_latest_recording_map():
    maps_dir = get_recording_maps_dir()
    json_files = sorted(maps_dir.glob('*.json'), key=lambda p: p.stat().st_mtime)
    return json_files[-1] if json_files else None


def load_recording_map(path):
    return json.loads(Path(path).read_text())
