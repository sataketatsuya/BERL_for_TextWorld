import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import joblib

import torch
import textworld
import textworld.gym
from textworld import EnvInfos
import gym
from agent import NerBertAgent
from commandgenerator import CommandModel
from nerdataset import get_category


DEBUG = True


def get_cv_games(datapath, block=''):
    games = [game for game in os.listdir(os.path.join(datapath, block)) if game.endswith('.ulx')]
    return [os.path.join(datapath, block, game) for game in games]


def load_agent(checkpoint_directory):
    return joblib.load(os.path.join(checkpoint_directory, 'ner_bert_agent.pkl'))


def train(args):
    """ train the ner bert agent """
    checkpoint_directory = os.path.join(args.output, 'ner_bert_agent')

    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)

    train_games = get_cv_games(args.datapath, 'train')
    gamefile = train_games[0] # Training for just a game
    if DEBUG:
        print('Game name:', gamefile)

    try:
        agent = load_agent(checkpoint_directory)
        if DEBUG:
            print('Successfully loaded agent. file name is ner_bert_agent.pkl')
    except:
        agent = NerBertAgent(args)
    agent.train()

    requested_infos = EnvInfos(
        max_score=True, won=True, lost=True,                            # Handicap 0
        description=True, inventory=True, objective=True,               # Handicap 1
        verbs=True, command_templates=True,                             # Handicap 2
        entities=False,                                                 # Handicap 3
        extras=[""],                                                    # Handicap 4
        admissible_commands=False)                                      # Handicap 5

    env_id = textworld.gym.register_games([gamefile], requested_infos)
    env = gym.make(env_id)
    gamesteps = []
    obs, infos = env.reset()

    for episode in tqdm(range(args.episodes)):
        obs, infos = env.reset()

        scores = 0
        done = False
        num_steps = 0
        total_reward = 0
        while not done and num_steps <= args.max_steps:
            command = agent.act(obs, scores, done, infos)
            if command is None:
                break
            obs, scores, done, infos = env.step(command)
            num_steps += 1

            if scores != 0:
                total_reward += 1
                print(f'Agent got a reward. Command is {command}')

            # if episode % 10 == 0: # save the agent
            #     joblib.dump(agent, os.path.join(checkpoint_directory, 'ner_bert_agent.pkl'), compress=True)

        max_score = infos['max_score']
        print(f'Episode {episode}: Max score is {max_score}. Agent got {total_reward}.')
        agent.act(obs, scores, done, infos) # Let the agent know the game is done.
        gamesteps.append(num_steps)
        # np.save('gamesteps', gamesteps)


def initialize_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=str,
                        help="path for output modelxs")
    parser.add_argument('--datapath', required=True, type=str,
                        help="path to the folder games")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Total batch size for training.")
    parser.add_argument('--hidden_size', default=256, type=int,
                        help="Number of rows from dataset to use (default 0 uses all data)")
    parser.add_argument("--episodes", required=True, type=int,
                        help="The episodes number being executed")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="Learning rate for model training")
    parser.add_argument("--gamma", default=0.9, type=float,
                        help="Gamma for model training")
    parser.add_argument("--max_steps", default=100, type=int,
                        help="Max steps in an episode")
    parser.add_argument("--validate", action='store_true',
                        help="Run validation")
    parser.add_argument("--batch_size_eval", default=1, type=int,
                        help="Total batch size for evaluation.")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="Max sequence size in model")
    parser.add_argument("--clean", action='store_true',
                        help="Remove previous model checkpoints")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                            "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                            "bert-base-multilingual-cased, bert-base-chinese.")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initialize_random_seed(args.episodes)

    train(args)
