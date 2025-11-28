import math
import os
import random
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torchvision.utils import make_grid


def save_img(img, img_path, mode='RGB'):
    img = torch.squeeze(img)
    img = torch.transpose(img, 0, 1)
    img = torch.transpose(img, 1, 2).cpu().numpy() * 255
    cv2.imwrite(img_path, img)


def seed_everything(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, epoch, model_name, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    checkpoint_file = os.path.join(outdir, model_name + '_' + 'epoch_' + str(epoch) + '.pth')
    torch.save(state, checkpoint_file)


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location=lambda storage, loc: storage.cuda(0))
    new_state_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('module'):
            name = key[7:]
        else:
            name = key
        new_state_dict[name] = value
    model.load_state_dict(new_state_dict)


def check_grads(model, log_file_path, epoch, step):
    # Tạo danh sách các thông báo để ghi một lần
    messages = []
    messages.append(f"\n[Epoch {epoch} - Step {step}] --- CHECKING GRADIENTS ---")
    
    has_nan = False
    has_zero = False
    has_exploding = False
    
    # Ngưỡng kiểm tra
    threshold_vanishing = 1e-7
    threshold_exploding = 10.0

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_max = param.grad.abs().max().item()
            
            # 1. Kiểm tra NaN
            if torch.isnan(param.grad).any():
                msg = f"NaN detected in: {name}"
                messages.append(msg)
                print(msg) # Vẫn in lỗi nghiêm trọng ra màn hình để biết ngay
                has_nan = True
            
            # 2. Kiểm tra Vanishing
            elif grad_max < threshold_vanishing:
                msg = f"Vanishing (Max < {threshold_vanishing}): {name} | Max: {grad_max:.2e}"
                messages.append(msg)
                has_zero = True
                
            # 3. Kiểm tra Exploding
            elif grad_max > threshold_exploding:
                msg = f"Exploding (Max > {threshold_exploding}): {name} | Max: {grad_max:.2f}"
                messages.append(msg)
                has_exploding = True
                
    if not has_nan and not has_zero and not has_exploding:
        messages.append("Gradients look healthy.")
    
    messages.append("--------------------------------------------------")

    # GHI VÀO FILE
    try:
        with open(log_file_path, mode='a', encoding='utf-8') as f:
            f.write('\n'.join(messages) + '\n')
    except Exception as e:
        print(f"Không thể ghi log gradient: {e}")