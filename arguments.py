import argparse
from pathlib import Path
import json


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("function", type=str)
    parser.add_argument("config", type=str)
    
    args = parser.parse_args()

    func = args.function 
    with open(args.config) as f:
        args = json.load(f)

    for k, _ in args.items():
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
            "l2"
        ]:
            raise ValueError("%s is not a valid argument." % k)

    args["function"] = func
    
    if not Path(args["meta"]).exists():
        raise FileNotFoundError("Can't find %s" % args["meta"])
    if not Path(args["image_base"]).exists():
        raise FileNotFoundError("Can't find %s" % args["image_base"])

    return args
