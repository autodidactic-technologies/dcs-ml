import json

def dict_to_str(d: dict) -> str:
    return json.dumps(d, indent=2, sort_keys=True)