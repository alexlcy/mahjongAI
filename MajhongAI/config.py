# ==================================== Model weight ====================================
from datetime import datetime
pong_weight_path = './mahjong/models/weights/pong-ep3-val_acc_0.8889-val_f1_0.9078.pth'
kong_weight_path = './mahjong/models/weights/kong-ep55-val_acc_0.8862-val_f1_0.9218.pth'
discard_weight_path = './mahjong/models/weights/discard-ep17-val_acc_0.7356-val_f1_0.7356.pth'

log_dir = f'./runs/run{datetime.now().strftime("%Y-%m-%d-%H%M")}'

