import torch
from torch.optim.lr_scheduler import  LambdaLR
import math
def get_lr_schedular(total_steps,warmup_steps,optimizer,schedular_type):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1,warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1,total_steps - warmup_steps))
            if schedular_type.lower() == "linear":
                return max(0.0, 1.0 - progress)
            elif schedular_type.lower() == "cosine":
                return 0.5 * (1.0 + math.cos(progress * math.pi))
            else:
                raise ValueError("unknown decay type")
    return LambdaLR(optimizer=optimizer,lr_lambda=lr_lambda)
