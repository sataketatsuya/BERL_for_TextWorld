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
    config_file = os.path.join(os.path.dirname(basepath), 'config', 'config.yaml')
    if not os.path.exists(config_file):
        raise NotImplementedError()

    # preprocess games walkthrough
    print('start preprocessing games walkthrough')
    subprocess.run(["python", os.path.join(basepath, 'dataset.py'), games_path, "--output", output])
    print('done')

    # preprocess data for NER
    print('start preprocessing data for NER')
    subprocess.run(["python", os.path.join(basepath, 'nerdataset.py'), "--output", output])
    print('done')

    # train NER model
    print('start training NER model')
    subprocess.run(["python", os.path.join(basepath, 'nertrain.py'), "--output", output])
    print('done')

    # train NER BERT Agent
    print('start training NER BERT Agent')
    games_path = os.path.join(games_path, 'train')
    subprocess.run(["python", os.path.join(basepath, 'agenttrain.py'),
                    "--output", output, "--config_file", config_file, "--games", games_path])
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--games_path', type=str, help="path to the games files")
    args = parser.parse_args()
    run_pipeline(args.games_path)
