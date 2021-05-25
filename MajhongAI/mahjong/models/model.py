from .net import MJNet
import config as cfg

import torch

class BaseModel():
	def __init__(self, n_cls, weight_path=None, device=None, history_len=4):
		if device is None:
			self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		else:
			self.device = device
		self._load_models(n_cls, weight_path, history_len)

	def _load_models(self, n_cls, weight_path, history_len):
		self.model = MJNet(history_len, n_cls)
		if weight_path is not None:
			self.model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
			print(f'Loaded checkpoint {weight_path.split("/")[-1]}')
		
		self.model.to(self.device)

	def predict(self, inp):
		'''
		Run forward propagation for all models

		Args:
		- inp (torch.float32): array of size [bs, (hist+1)*39, 34, 1]

		Returns:
		- preds (torch.float32): array of size [bs, n_cls]
		'''
		self.model.eval()
		with torch.no_grad():
			preds = self.model(inp)

		return preds.cpu()


class PongModel(BaseModel):
	def __init__(self, device=None):
		super().__init__(2, cfg.pong_weight_path, device)

class KongModel(BaseModel):
	def __init__(self, device=None):
		super().__init__(2, cfg.kong_weight_path, device)

class DiscardModel(BaseModel):
	def __init__(self, device=None):
		super().__init__(34, cfg.discard_weight_path, device)