import json


def dumper(obj):
    """JSON serializer for path objects not serializable by default json code"""
    try:
        return obj.toJSON()
    except AttributeError:
        return str(obj)
