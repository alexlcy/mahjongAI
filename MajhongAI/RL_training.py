import h5py
import glob
from tqdm import tqdm
from mahjong.ReinforcementLearning.experience import ExperienceBuffer
from mahjong.models.model import DiscardModel, KongModel, PongModel

import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper-parameters
lr = 0.01

def get_discounted_reward(r):
    discounted_r = (r - r.mean())/(r.std() + 1e-7)
    return discounted_r

def preprocess_data(s, a, r):
    s = torch.tensor(s, dtype=torch.float32, device=device)
    a = torch.tensor(a).squeeze(1).argmax(-1).long().to(device)
    r = torch.tensor(r, dtype=torch.float32, device=device)
    discounted_r = get_discounted_reward(r)
    return s, a, discounted_r

# Training
exp_paths = glob.glob('mahjong/data/*.h5')
exp_buffer = ExperienceBuffer()
model = DiscardModel(device).model
loss_fn = nn.CrossEntropyLoss(reduction='none')
optim = torch.optim.Adam(model.parameters(), lr=lr)

for exp in tqdm(exp_paths, desc=f"Training on experience buffer: "):
    states, rewards, actions = exp_buffer.read_experience(exp)
    states, actions, rewards = preprocess_data(states, actions, rewards)
    action_logits = model(states)
    
    optim.zero_grad()
    loss = rewards * loss_fn(action_logits, actions)
    loss = loss.mean()
    loss.backward()
    optim.step()