import json


def parse_cfg(cfg_file):
    with open(cfg_file, 'r') as f:
        data = f.read()
        data = json.loads(data)
    return data
