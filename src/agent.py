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
from ppo_memory import PPOMemory
from command_scorer import ActorNetwork, CriticNetwork
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
        self.update_frequency = self.config['training']['update_frequency']
        self.gamma = self.config['training']['gamma']
        self.gae_lambda = self.config['training']['gae_lambda']
        self.policy_clip = self.config['training']['policy_clip']

        self.id2word = ["<PAD>", "<UNK>"]
        self.word2id = {w: i for i, w in enumerate(self.id2word)}

        #　load trained NER Recognition Model
        self.ner_datapath = os.path.join(args.output, 'nermodel')
        self.ner = Ner(self.ner_datapath)
        self.custom_template = self.config['training']['custom_template']

        # Command Generator Model
        self.langmodel = CommandModel()
        self.cp = CompactPreprocessor()

        # Actor-Critic
        self.alpha = self.config['training']['optimizer']['alpha']
        self.input_dims = self.config['training']['optimizer']['input_dims']
        self.critic = CriticNetwork(self.device, self.input_dims, self.alpha)
        self.actor = ActorNetwork(self.device, self.config, self.critic)
        self.memory = PPOMemory(self.batch_size)

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
        self.actor.reset_hidden()
        self.last_score = 0
        self.state_text = ''

    def test(self):
        self.mode = "test"
        self.actor.reset_hidden()

    def remember(self, state, action, admissible_commands, probs, vals, reward, done):
        self.memory.store_memory(state, action, admissible_commands, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

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
        texts = re.sub("[^a-zA-Z0-9\- ]", " ", texts)
        texts = list(map(self._get_word_id, texts.split()))
        padded = np.ones((len(texts), 1)) * self.word2id["<PAD>"]

        for i, text in enumerate(texts):
            padded[i, :text] = text

        padded_tensor = torch.from_numpy(padded).type(torch.long).to(self.device)
        padded_tensor = padded_tensor.permute(1, 0) # Batch x Seq => Seq x Batch
        return padded_tensor

    def _process_command(self, texts):
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
        elif not self.reading:
            cmds = self.langmodel.get_direction_cmd(output=[], entities=entities)
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

    def store_state_text(self, obs: str, infos: Dict[str, Any]):
        if not self.reading:
            self.reading = 'and start reading' in obs
        self.recipe = self._get_recipe(obs)

        if not self.prepared:
            self.prepared = 'Adding the meal to your inventory' in obs

        inventory = infos['inventory']
        description = self._preprocess_description(infos['description'])

        # Let the state define <number of items in inventory><inventory text><recipe text><look text>
        # Tokenize and pad the state
        self.state_text = self.cp.convert(description, self.recipe, inventory, get_all_entities())

    def choose_action(self, obs: str, infos: Dict[str, Any]):
        inventory = infos['inventory']
        description = self._preprocess_description(infos['description'])

        if self.state_text == '':
            self.store_state_text(obs, infos)
        state_tensor = self._process(self.state_text) # torch.Size([1, state_text_length])

        # Generate commands from observation with NER Bert
        entity_types = self.generate_entities(description, inventory) # ('cookbook', 'T'), ('south', 'W'), ('west', 'W'), ('red onion', 'F')

        # Drop recipe necessary items
        if self.reading:
            entity_types = self.drop_unnecessary_items(entity_types)

        commands = self.get_admissible_commands(description, inventory, entity_types, templates if self.custom_template else infos['command_templates'])
        commands_tensor = self._process_command(commands) # torch.Size([max_command_size, command_length])

        # Get our next action and value prediction.
        dist, value = self.actor(state_tensor, commands_tensor)

        # sample from the distribution over commands
        index = dist.sample()
        # get probability command
        probs = dist.log_prob(index).item()

        action = commands[index]

        return action, probs, value, commands, index.item()

    def learn(self):
        for _ in range(self.nb_epochs):
            state_arr, action_arr, admissible_commands_arr,\
            old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.from_numpy(advantage).to(self.device)

            values = torch.from_numpy(values).to(self.device)
            for batch in batches:
                state_tensor = self._process(state_arr[batch][0])
                commands_tensor = self._process_command(admissible_commands_arr[batch][0])
                old_probs = torch.from_numpy(old_prob_arr[batch]).to(self.device)
                actions = torch.from_numpy(action_arr[batch]).to(self.device)

                dist, critic_value = self.actor(state_tensor, commands_tensor)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp() # importance ratio
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch[0]] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch[0]] # Cliping
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch[0]] + values[batch[0]]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                self.actor.reset_hidden()

        self.memory.clear_memory()

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
