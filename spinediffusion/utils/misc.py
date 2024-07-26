import json

import diffusers
import torch
import torchmetrics

import spinediffusion


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


def instantiate_model_from_logs(df_run, config, ckpt_path):
    """Instantiate a model from the logs.

    Args:
        df_run (pd.DataFrame): The dataframe of the run logs.
        config (dict): The configuration dictionary.
        ckpt_path (str): The path to the checkpoint.

    Returns:
        pl.LightningModule: The instantiated model.
    """
    model = eval(config["model"]["init_args"]["model"]["class_path"])(
        **config["model"]["init_args"]["model"]["init_args"]
    )
    if isinstance(config["model"]["init_args"]["scheduler"], dict):
        scheduler = eval(config["model"]["init_args"]["scheduler"]["class_path"])(
            **config["model"]["init_args"]["scheduler"]["init_args"]
        )
    else:
        scheduler = eval(config["model"]["init_args"]["scheduler"])()
    loss = eval(config["model"]["init_args"]["loss"]["class_path"])(
        **config["model"]["init_args"]["loss"]["init_args"]
    )
    metrics = []
    for metric_dict in config["model"]["init_args"]["metrics"].values():
        metric = eval(metric_dict["class_path"])(**metric_dict["init_args"])
        metrics.append(metric)

    lightning_model = eval(config["model"]["class_path"]).load_from_checkpoint(
        ckpt_path, model=model, scheduler=scheduler, loss=loss, metrics=metrics
    )

    return lightning_model
