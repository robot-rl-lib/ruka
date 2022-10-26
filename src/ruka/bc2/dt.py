import math
from typing import Callable, List, Optional
import collections

import numpy as np
import ruka.pytorch_util as ptu
import torch
import torch.nn as nn
from ruka.bc2.base import Action, BaseFeaturesExtractor, Observation
from ruka.bc2.utils import default_collate_fn, numpy_tree_to_torch
from ruka.models.mlp import Mlp
from torch.nn import functional as F


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, action_dim, block_size, **kwargs):
        self.block_size = block_size
        self.action_dim = action_dim
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                                     .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.action_dim, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)


        print("number of parameters: %e", sum(p.numel() for p in self.parameters()))


        # self.state_encoder = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
        #                          nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
        #                          nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
        #                          nn.Flatten(), nn.Linear(3136, config.n_embd), nn.Tanh())

        # self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        # self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        # nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, state_embeddings, action_embeddings, targets=None, rtgs=None, timesteps=None):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)

        # state_embeddings = self.state_encoder(states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous()) # (batch * block_size, n_embd)
        # state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd) # (batch, block_size, n_embd)
        
        if action_embeddings is not None and self.model_type == 'reward_conditioned': 
            raise NotImplementedError()
        elif action_embeddings is None and self.model_type == 'reward_conditioned': # only happens at very first timestep of evaluation
            raise NotImplementedError()
        elif action_embeddings is not None and self.model_type == 'naive':
            token_embeddings = torch.zeros((state_embeddings.shape[0], state_embeddings.shape[1]*2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = state_embeddings
            token_embeddings[:,1::2,:] = action_embeddings[:,-state_embeddings.shape[1] + int(targets is None):,:]
        elif action_embeddings is None and self.model_type == 'naive': # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        batch_size = state_embeddings.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd
        position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if action_embeddings is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1::3, :] # only keep predictions from state_embeddings
        elif action_embeddings is None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1:, :]
        elif action_embeddings is not None and self.model_type == 'naive':
            logits = logits[:, ::2, :] # only keep predictions from state_embeddings
        elif action_embeddings is None and self.model_type == 'naive':
            action_embeddings = logits # for completeness
        else:
            raise NotImplementedError()

        # # if we are given some desired targets also calculate the loss
        # loss = None
        # if targets is not None:
        #     loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits# , loss

class DTMDPPolicy(nn.Module):
    def __init__(
        self, 
        observation_encoder: nn.Module,
        actions_encoder: nn.Module,
        action_size: int,
        ):
        super().__init__()
        self._observation_encoder = observation_encoder
        self._actions_encoder = actions_encoder

        self._dt = GPT(
            GPTConfig(action_dim=action_size, block_size=8, n_layer=6, n_head=8, n_embd=128, model_type='naive', max_timestep=200)
        )

    def forward(
        self, 
        observation_sequences_stacked: dict, 
        action_sequences_stacked,
        ):
        encoded_obs = self._observation_encoder(observation_sequences_stacked)
        encoded_acts = self._actions_encoder(action_sequences_stacked)
        timesteps = observation_sequences_stacked['timestep'][:, -1:, :].long()
        actions = self._dt(encoded_obs, encoded_acts, timesteps=timesteps)[:, -1, :]
        return actions

    def get_action(self, obs_seq: List[Observation], act_seq):
        obs_stacked = self._default_collate_fn([obs_seq])
        act_stacked = self._default_collate_fn([act_seq])

        # obs_stacked = crop_augment(obs_stacked, ['depth', 'target_segmentation', 'gray'], 60)
        
        obs_stacked = numpy_tree_to_torch(obs_stacked)
        act_stacked = numpy_tree_to_torch(act_stacked)

        with torch.no_grad():
            actions = self.forward(obs_stacked, act_stacked).cpu().detach().numpy()
        return actions[0]

    @staticmethod
    def _default_collate_fn(sequence_list):
        return default_collate_fn(sequence_list)


class DTStatefulPolicy:
    def __init__(self, policy: DTMDPPolicy, seqlen=4, action_size=5):
        self.policy = policy
        self._seqlen = seqlen
        self.actions = collections.deque(maxlen=seqlen - 1)
        self.observations = collections.deque(maxlen=seqlen)
        self._zero_action = np.zeros(action_size)

    def get_action(self, observation):
        self.observations.append(observation)
        while len(self.observations) < self._seqlen:
            self.actions.append(self._zero_action)
            self.observations.append(observation)
        action = self.policy.get_action(self.observations, self.actions)
        self.actions.append(action)
        return action

    def reset(self):
        self.actions = collections.deque(maxlen=self._seqlen - 1)
        self.observations = collections.deque(maxlen=self._seqlen)
