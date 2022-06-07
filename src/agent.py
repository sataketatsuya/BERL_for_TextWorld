import re
import os
import yaml
import pprint
from typing import Mapping, Any, Dict
from collections import defaultdict, namedtuple
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from command_scorer import CommandScorer
from textutils import CompactPreprocessor
from bertner import Ner
from commandgenerator import CommandModel
from ner import extract_entities
from nerdataset import get_all_entities, templates


class NerBertAgent:
    """ Simple Neural Agent for playing TextWorld games. """
    def __init__(self, args, verbose=False) -> None:
        # Load the config file
        config_file = args.config_file
        with open(config_file) as reader:
            self.config = yaml.safe_load(reader)
        if verbose:
            pprint.pprint(self.config, width=1)

        # choose device
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        if args.gpu is not None and args.gpu > 1:
            self.device = 'cuda:{}'.format(args.gpu)

        # training settings
        self.batch_size = self.config['training']['batch_size']
        self.max_nb_steps_per_episode = self.config['training']['max_nb_steps_per_episode']
        self.nb_epochs = self.config['training']['nb_epochs']

        self.id2word = ["<PAD>", "<UNK>"]
        self.word2id = {w: i for i, w in enumerate(self.id2word)}

        #　load trained NER Recognition Model
        self.ner_datapath = os.path.join(args.output, 'nermodel')
        self.ner = Ner(self.ner_datapath)
        self.custom_template = self.config['training']['custom_template']

        # Command Generator Model
        self.langmodel = CommandModel()
        self.cp = CompactPreprocessor()

        self.model = CommandScorer(self.device, self.config)
        self.optimizer = optim.Adam(self.model.parameters(), self.config['training']['optimizer']['learning_rate'])

        self.mode = "test"
        self.model_updates = 0
        self.no_train_step = 0

        # regex for hifen words workaround
        self.rgx = re.compile(r'\b(\w+\-\w+)\b')
        self.hifen_map = {}
        self.hifen_rev_map = {}

    def train(self):
        # capture the recipe when agent examines the cookbook
        self.recipe = 'missing recipe'
        self.reading = False
        self.prepared = False

        self.mode = "train"
        self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
        self.transitions = []
        self.model.reset_hidden()
        self.last_score = 0
        self.previous_action = None

    def test(self):
        self.mode = "test"
        self.model.reset_hidden()

    def _get_word_id(self, word):
        if word not in self.word2id:
            if len(self.word2id) >= self.config['model']['max_seq_length']:
                return self.word2id["<UNK>"]

            self.id2word.append(word)
            self.word2id[word] = len(self.word2id)

        return self.word2id[word]

    def _tokenize(self, text):
        # Simple tokenizer: strip out all non-alphabetic characters.
        text = re.sub("[^a-zA-Z0-9\- ]", " ", text)
        word_ids = list(map(self._get_word_id, text.split()))
        return word_ids

    def _process(self, texts):
        texts = list(map(self._tokenize, texts))
        max_len = max(len(l) for l in texts)
        padded = np.ones((len(texts), max_len)) * self.word2id["<PAD>"]

        for i, text in enumerate(texts):
            padded[i, :len(text)] = text

        padded_tensor = torch.from_numpy(padded).type(torch.long).to(self.device)
        padded_tensor = padded_tensor.permute(1, 0) # Batch x Seq => Seq x Batch
        return padded_tensor

    def _preprocess_description(self, description):
        mobj = self.rgx.search(description)
        if mobj:
            kw = mobj.group(0)
            target = kw.replace('-', ' ')
            self.hifen_map[kw] = target
            self.hifen_rev_map[target] = kw
            return description.replace(kw, target)
        return description

    def entities_mapping(self, entities):
        res = []
        for e,t in entities:
            for k in self.hifen_rev_map.keys():
                if k in e:
                    e = e.replace(k, self.hifen_rev_map[k])
            res.append((e,t))
        return res

    def generate_entities(self, description, inventory=''):
        entities = extract_entities(description, inventory, model=self.ner)
        return self.entities_mapping(entities)

    def get_admissible_commands(self, description, inventory, entities, templates):
        if 'cookbook' in description and not self.reading:
            cmds = ['examine cookbook']
        else:
            cmds = self.langmodel.generate_all(entities, templates)
        if self.prepared and 'eat meal' not in cmds:
            cmds.append('eat meal')
        return cmds

    def drop_unnecessary_items(self, entity_types):
        recipe_entities_types = self.generate_entities(self.recipe)
        recipe_entities = [e for e, _ in recipe_entities_types]
        necessary_entities = []
        for entity in entity_types:
            if entity[1] == 'F':
                if entity[0] in recipe_entities:
                    necessary_entities.append(entity)
            else:
                if not entity[0] == 'cookbook':
                    necessary_entities.append(entity)
        return necessary_entities

    def _discount_rewards(self, last_values):
        returns, advantages = [], []
        R = last_values.data
        for t in reversed(range(len(self.transitions))):
            rewards, _, _, values = self.transitions[t]
            R = rewards + self.config['general']['discount_gamma'] * R
            adv = R - values
            returns.append(R)
            advantages.append(adv)

        return returns[::-1], advantages[::-1]

    def act(self, obs: str, score: int, done: bool, infos: Dict[str, Any]):
        if not self.reading:
            self.reading = 'and start reading' in obs
        self.recipe = self._get_recipe(obs)

        if not self.prepared:
            self.prepared = 'Adding the meal to your inventory' in obs

        inventory = infos['inventory']
        description = self._preprocess_description(infos['description'])

        # Let the state define <number of items in inventory><inventory text><recipe text><look text>
        # Tokenize and pad the state
        state_text = self.cp.convert(description, self.recipe, inventory, get_all_entities())
        state_tensor = self._process(state_text) # torch.Size([1, state_text_length])

        # Generate commands from observation with NER Bert
        entity_types = self.generate_entities(description, inventory) # ('cookbook', 'T'), ('south', 'W'), ('west', 'W'), ('red onion', 'F')

        # Drop recipe necessary items
        if self.reading:
            entity_types = self.drop_unnecessary_items(entity_types)

        commands = self.get_admissible_commands(description, inventory, entity_types, templates if self.custom_template else infos['command_templates'])
        commands_tensor = self._process(commands) # torch.Size([max_command_size, command_length])

        # Get our next action and value prediction.
        cmd_scores, prob, value, index = self.model(state_tensor, commands_tensor)

        action = commands[index]

        if self.mode == "test":
            if done:
                self.model.reset_hidden()
            return action

        self.no_train_step += 1

        if self.transitions:
            reward = score - self.last_score  # Reward is the gain/loss in score.
            self.last_score = score

            self.transitions[-1][0] = reward  # Update reward information.

        # self.stats["max"]["score"].append(score)
        if self.no_train_step % 10 == 0:

            # Update model
            returns, advantages = self._discount_rewards(value)

            loss = 0
            for transition, _return, advantage in zip(self.transitions, returns, advantages):
                reward, index, cmd_scores, values_ = transition

                advantage        = advantage.detach()
                probs            = F.softmax(cmd_scores, dim=-1)
                log_probs        = torch.log(probs)
                log_action_prob  = log_probs[index]
                policy_loss      = -log_action_prob * advantage
                value_loss       = (.5 * (values_ - _return) ** 2.)
                entropy          = (-log_probs * probs).mean()

                # add up the loss over time
                loss += policy_loss + 0.5 * value_loss - 0.1 * entropy

                self.stats["mean"]["reward"].append(reward)
                self.stats["mean"]["policy"].append(policy_loss.item())
                self.stats["mean"]["value"].append(value_loss.item())
                self.stats["mean"]["entropy"].append(entropy.item())
                self.stats["mean"]["confidence"].append(torch.exp(log_action_prob).item())

            self.model_updates += 1
            self.transitions = []

            if loss != 0:
                if self.no_train_step % 1000 == 0:
                    msg = "{:6d}. updated {}. ".format(self.no_train_step, self.model_updates)
                    msg += "  ".join("{}: {: 3.3f}".format(k, np.mean(v)) for k, v in self.stats["mean"].items())
                    msg += "  " + "  ".join("{}: {:2d}".format(k, np.max(v)) for k, v in self.stats["max"].items())
                    msg += "  vocab: {:3d}".format(len(self.id2word))
                    msg += "  loss: {}".format(loss)
                    print(msg)
                self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.model.parameters(), 40)
                self.optimizer.step()

                self.model.reset_hidden() # Do not edit this code because of pytorch error
        else:
            # Keep information about transitions for Truncated Backpropagation Through Time.
            self.transitions.append([None, index, cmd_scores, value.item()])  # Reward will be set on the next call

        if done:
            self.last_score = 0  # Will be starting a new episode. Reset the last score.

        return action

    def _get_recipe(self, observation, explicit_recipe=None):
        """
        Returns the recipe if possible. For HCP >=4 you can provide the info['extra.recipe'] as explicit recipe.
        Otherwise the observation is stored as the recipe if the last commmand was 'examine recipe' (=self.reading).
        """
        recipe = 'missing recipe'
        if self.recipe == 'missing recipe':
            if explicit_recipe is not None:
                recipe = explicit_recipe
            else:
                if self.reading:
                    recipe = '\nRecipe {}\n'.format(observation.split('\n\nRecipe ')[1].strip())
        else:
            recipe = self.recipe
        return recipe
