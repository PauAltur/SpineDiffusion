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


def instantiate_model_from_logs(config, ckpt_path):
    """Instantiate a model from the logs.

    Args:
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


def instantiate_datamodule_from_logs(
    config,
    system,
    data_dir="P:/Projects/LMB_4Dspine/back_scan_database",
    cache_dir="P:/Projects/LMB_4Dspine/Iship_Pau_Altur_Pastor/3_database/cache",
    num_subjects=None,
    predict_size=None,
):
    """Instantiate a datamodule from the config file.

    Args:
        config (dict): The configuration dictionary.
        system (str): The operating system on which the code is running.
    """
    if system == "Windows":
        config["data"]["init_args"]["data_dir"] = data_dir
    elif system == "Linux":
        config["data"]["init_args"]["cache_dir"] = cache_dir

    if "conditional" in config["data"]["init_args"]:
        config["data"]["init_args"].pop("conditional")

    if num_subjects is not None:
        config["data"]["init_args"]["num_subjects"] = num_subjects

    if predict_size is not None:
        config["data"]["init_args"]["predict_size"] = predict_size

    datamodule = eval(config["data"]["class_path"])(**config["data"]["init_args"])

    return datamodule
