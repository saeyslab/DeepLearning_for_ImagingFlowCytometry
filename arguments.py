import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("function", type=str)

    # no default
    parser.add_argument("--noc",type=int)
    parser.add_argument("--channels", "-c", type=int, nargs="+")
    parser.add_argument("--meta", type=str)
    parser.add_argument("--image_base", type=str)
    parser.add_argument("--split_dir", type=str)

    # default
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_width", type=int, default=90)
    parser.add_argument("--image_height", type=int, default=90)

    args = parser.parse_args()

    if not Path(args.meta).exists():
        raise FileNotFoundError("Can't find %s" % args.meta)
    if not Path(args.image_base).exists():
        raise FileNotFoundError("Can't find %s" % args.image_base)

    return args
