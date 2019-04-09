import argparse
from pathlib import Path
import json


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("function", type=str)
    parser.add_argument("config", type=str)
    parser.add_argument("run_dir", type=str)
    parser.add_argument("root", type=str)
    parser.add_argument("--overwrite", type=str, default="{}")
    
    args = vars(parser.parse_args())

    with open(args["config"]) as f:
        json_args = json.load(f)

        for k,v in json.loads(args["overwrite"]).items():
            json_args[k] = v

    valid_args = [
            "noc",
            "channels", "c",
            "meta",
            "image_base",
            "split_dir",
            "run_dir",
            "epochs",
            "model_hdf5",
            "model",
            "batch_size",
            "image_width",
            "image_height",
            "update_freq",
            "freq_type",
            "dropout",
            "l2",
            "learning_rate",
            "learning_rate_decay",
            "epochs_per_decay",
            "momentum",
            "optimizer",
            "augmentation",
            "schedule",
            "comet_project",
            "sample_weights",
            "param_grid",
            "h5_data",
            "es_patience",
            "es_epsilon",
            "beta1",
            "epsilon",
            "skip_n_folds",
            "model_hdf5",
            "embedding_output",
            "layer"
        ]

    for k, v in json_args.items():
        if k not in valid_args:
            raise ValueError("%s is not a valid argument." % k)
        args[k] = v

    if "param_grid" in args:
        for params in args["param_grid"]:
            for k, _ in params.items():
                if k not in valid_args:
                    raise ValueError("%s from param_grid is not a valid argument" % k)

    args["augmentation"] = args["augmentation"] == 1

    if "sample_weights" in args:
        if len(args["sample_weights"]) != args["noc"]:
            raise ValueError("Length of provided sample weights does not equal number of classes.")
    else:
        args["sample_weights"] = [1.0]*args["noc"]

    for k in ["meta", "split_dir", "h5_data", "model_hdf5"]:
        if k in args:
            args[k] = str(Path(args["root"], args[k]))
            if not Path(args[k]).exists():
                raise FileNotFoundError("Can't find %s" % args[k])

    return args
