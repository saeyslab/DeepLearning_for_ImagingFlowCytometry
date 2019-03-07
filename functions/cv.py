import main

def run(args):
    experiment = main.prerun()

    from os import sep

    split_dir = Path(args["split_dir"])
    for i, fold in enumerate(split_dir.iterdir()):
        if fold.is_dir():
            run = Path(args["run_dir"], str(fold).split(sep)[-1])
            Path(run).mkdir()
            train(fold, run, i, cv=experiment)