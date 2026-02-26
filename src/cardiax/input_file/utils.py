
from pathlib import Path
import yaml

# Tell PyYAML to treat Path objects exactly like normal strings when dumping
def path_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))

# Register it for both standard and safe dumpers
yaml.add_multi_representer(Path, path_representer)
yaml.representer.SafeRepresenter.add_representer(Path, path_representer)

def get_dict(d: dict, key: str) -> dict:
    """
    Safely retrieves a sub-dictionary. 
    If the key is missing OR the value is None, returns {}.
    """
    val = d.get(key)
    return val if val is not None else {}

def get_Path(d: dict, key: str) -> Path:
    """
    Safely retrieves a Path object from a dictionary. 
    If the key is missing OR the value is None, returns None.
    """
    val = d.get(key)
    return Path(val) if val is not None else None