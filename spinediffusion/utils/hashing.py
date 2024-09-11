import hashlib


def dict_to_str(d: dict) -> str:
    """Recursively converts a dictionary into a string representation.

    Args:
        d (dict): The dictionary to convert.
    """

    def recursive_items(dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                yield (key, dict_to_str(value))
            else:
                yield (key, value)

    return (
        "{"
        + ", ".join(f"{key}: {value}" for key, value in sorted(recursive_items(d)))
        + "}"
    )


def hash_dict(d: dict) -> str:
    """
    Converts a dictionary to a string and then hashes it.

    Args:
        d (dict): The dictionary to hash.
    """
    dict_str = dict_to_str(d)
    return hashlib.sha256(dict_str.encode()).hexdigest()


def hash_dict_keys(d: dict) -> str:
    """
    Hashes the keys of a dictionary.

    Args:
        d (dict): The dictionary to hash.
    """
    keys_str = str(list(d.keys()))
    return hashlib.sha256(keys_str.encode()).hexdigest()
