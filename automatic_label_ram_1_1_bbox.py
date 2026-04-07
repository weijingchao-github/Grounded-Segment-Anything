#!/usr/bin/env python3
import importlib.util
import os


def _load_bbox_source_module():
    source_path = os.path.join(
        os.path.dirname(__file__), 'automatic_label_ram_1_0_validation.py'
    )
    module_name = 'grounded_sam_automatic_label_ram_1_0_validation_source'
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to load module spec from {source_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SOURCE_MODULE = _load_bbox_source_module()


def main():
    _SOURCE_MODULE.main()


if __name__ == '__main__':
    main()
