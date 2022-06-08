import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from time import time
import joblib
import glob

import torch
import textworld
import textworld.gym
from textworld import EnvInfos
import gym
from agent import NerBertAgent
from utils import get_points

def get_cv_games(datapath, block=''):
    games = [game for game in os.listdir(os.path.join(datapath, block)) if game.endswith('.ulx')]
    return [os.path.join(datapath, block, game) for game in games]


class Environment:
    """
    Wrapper for the TextWorld Environment.
    """
    def __init__(self, games_dir, max_nb_steps=100, batch_size=1):
        self.games = self.get_games(games_dir)
        self.max_nb_steps = max_nb_steps
        self.batch_size = batch_size
        self.env = self.setup_env()

    def step(self, commands):
        return self.env.step(commands)

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def manual_game(self):
        try:
            done = False
            self.env.reset()
            nb_moves = 0
            while not done:
                self.env.render()
                command = input("Input ")
                nb_moves += 1
                obs, scores, dones, infos = self.env.step([command])

            self.env.render()  # Final message.
        except KeyboardInterrupt:
            pass  # Quit the game.

        print("Played {} steps, scoring {} points.".format(nb_moves, scores[0]))

    def setup_env(self):
        requested_infos = self.select_additional_infos()
        self.games = ['/home/satake/ledeepchef/game/tw-cooking-recipe2+take2+cut+open-BnYEixa9iJKmFZxO.ulx']
        env_id = textworld.gym.register_games(self.games, requested_infos,
                                            max_episode_steps=self.max_nb_steps)
        return gym.make(env_id)

    def get_games(self, games_dir):
        games = []
        for game in [games_dir]:
            if os.path.isdir(game):
                games += glob.glob(os.path.join(game, "*.ulx"))
            else:
                games.append(game)
        games = [os.path.join(os.getcwd(), game) for game in games]
        print("{} games found for training.".format(len(games)))
        return games

    def select_additional_infos(self) -> EnvInfos:
        request_infos = EnvInfos(
            max_score=True, won=True, lost=True,                            # Handicap 0
            description=True, inventory=True, objective=True,               # Handicap 1
            verbs=True, command_templates=True,                             # Handicap 2
            entities=False,                                                 # Handicap 3
            extras=['walkthrough'],                                         # Handicap 4
            admissible_commands=False)                                      # Handicap 5

        return request_infos


class Trainer:
    def __init__(self, args):
        self.agent = NerBertAgent(args)
        self.env = Environment(args.games)
        self.checkpoint_directory = args.output

    def load_agent(self):
        return joblib.load(os.path.join(self.checkpoint_directory, 'ner_bert_agent', 'ner_bert_agent.pkl'))

    def train(self):
        self.start_time = time()

        for epoch_no in range(1, self.agent.nb_epochs + 1):
            for game_no in tqdm(range(1000)):
                obs, infos = self.env.reset()
                self.agent.train()

                score = 0
                done = False
                steps = 0
                while not done:
                    # Increase step counts.
                    steps += 1
                    command = self.agent.act(obs, score, done, infos)
                    obs, score, done, infos = self.env.step(command)

                print('Game Result Won:{} Lost:{}'.format(infos['won'], infos['lost']))
                print('Max score :{}, Agent score : {}'.format(infos['max_score'], score))
                if infos['won']:
                    print('Walkthrough steps :{}, Agent steps :{}'.format(len(infos['extra.walkthrough']), steps))
                # Let the agent know the game is done.
                self.agent.act(obs, score, done, infos)
                if game_no % 100 == 0:
                    joblib.dump(self.agent, os.path.join(self.checkpoint_directory, 'ner_bert_agent', 'ner_bert_agent.pkl'))
                    print('saved agent')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an agent.")
    parser.add_argument('--output', required=True, type=str,
                        help="path for output modelxs")
    parser.add_argument('--config_file', required=True, type=str,
                        help='File of the config fot training.')
    parser.add_argument('--games', required=True, type=str,
                        help='Directory of the games used for training.')
    parser.add_argument('--gpu', action='store', type=str, default=None,
                        help='Set cuda index for training.')
    args = parser.parse_args()

    Trainer(args).train()
