import os
import argparse
import subprocess


def run_pipeline(games_path):
    """
    Runs the pipeline of data preprocessing and model training.
    In the end models will be created in outputs folder.
    """
    basepath = os.path.dirname(os.path.realpath(__file__))
    output = os.path.join(os.path.dirname(basepath), 'outputs')
    os.makedirs(output, exist_ok=True)

    # preprocess games walkthrough
    subprocess.run(["python", os.path.join(basepath, 'datasets.py'), games_path, "--output", output])

    # preprocess data for NER
    subprocess.run(["python", os.path.join(basepath, 'nerdataset.py'), "--output", output])

    # train NER model
    subprocess.run(["python", os.path.join(basepath, 'nertrain.py'), "--output", output])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('games_path', type=str, help="path to the games files")
    args = parser.parse_args()
    run_pipeline(args.games_path)
