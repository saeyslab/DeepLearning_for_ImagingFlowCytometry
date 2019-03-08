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

    for k, v in json_args.items():
        if k not in [
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
            "param_grid"
        ]:
            raise ValueError("%s is not a valid argument." % k)
        args[k] = v

    args["augmentation"] = args["augmentation"] == 1

    if "sample_weights" in args:
        if len(args["sample_weights"]) != args["noc"]:
            raise ValueError("Length of provided sample weights does not equal number of classes.")
    else:
        args["sample_weights"] = [1.0]*args["noc"]

    for k in ["meta", "image_base", "split_dir"]:
        args[k] = str(Path(args["root"], args[k]))
        if not Path(args[k]).exists():
            raise FileNotFoundError("Can't find %s" % args[k])

    return args
