import json


def dumper(obj):
    """JSON serializer for path objects not serializable by default json code.

    Args:
        obj (object): The object to serialize.

    Returns:
        str: The serialized object.
    """
    try:
        return obj.toJSON()
    except AttributeError:
        return str(obj)


def find_test_param(config_dict: dict, test_param: str) -> str:
    """Find the value of a test parameter in a nested dictionary.

    Args:
        config_dict (dict): The dictionary of config files loaded as
        dictionaries.
        test_param (str): The parameter to search for.

    Returns:
        _type_: _description_
    """
    if test_param in config_dict:
        return config_dict[test_param]
    else:
        for value in config_dict.values():
            if isinstance(value, dict):
                result = find_test_param(value, test_param)
                if result is not None:
                    return result

    return None
