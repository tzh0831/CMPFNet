# import torch
# import math
from torch.optim.lr_scheduler import _LRScheduler
# from abc import ABCMeta, abstractmethod

# class BaseLR():
#     __metaclass__ = ABCMeta

#     @abstractmethod
#     def get_lr(self, cur_iter): pass

# class PolyLR(BaseLR):
#     def __init__(self, optimizer, start_lr, lr_power, total_iters):
#         self.start_lr = start_lr
#         self.lr_power = lr_power
#         self.total_iters = total_iters + 0.0
#         # super().__init__(optimizer)

#     def get_lr(self, cur_iter):
#         return self.start_lr * (
#                 (1 - float(cur_iter) / self.total_iters) ** self.lr_power)
    
# class WarmUpPolyLR(BaseLR):
#     def __init__(self, optimizer, start_lr, lr_power, total_iters, warmup_steps):
#         self.start_lr = start_lr
#         self.lr_power = lr_power
#         self.total_iters = total_iters + 0.0
#         self.warmup_steps = warmup_steps
#         # super().__init__(optimizer)

#     def get_lr(self, cur_iter):
#         if cur_iter < self.warmup_steps:
#             return self.start_lr * (cur_iter / self.warmup_steps)
#         else:
#             return self.start_lr * (
#                     (1 - float(cur_iter) / self.total_iters) ** self.lr_power)
    

class WarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_iter=300, warmup_ratio=5e-4, warmup='exp', last_epoch=-1) -> None:
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        return [ratio * lr for lr in self.base_lrs]

    def get_lr_ratio(self):
        return self.get_warmup_ratio() if self.last_epoch < self.warmup_iter else self.get_main_ratio()

    def get_main_ratio(self):
        raise NotImplementedError

    def get_warmup_ratio(self):
        assert self.warmup in ['linear', 'exp']
        alpha = self.last_epoch / self.warmup_iter

        return self.warmup_ratio + (1. - self.warmup_ratio) * alpha if self.warmup == 'linear' else self.warmup_ratio ** (1. - alpha)


class WarmupPolyLR(WarmupLR):
    def __init__(self, optimizer, power, max_iter, warmup_iter=300, warmup_ratio=5e-4, warmup='linear', last_epoch=-1) -> None:
        self.power = power
        self.max_iter = max_iter
        super().__init__(optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        alpha = real_iter / real_max_iter

        return (1 - alpha) ** self.power

def get_scheduler(optimizer, max_iter: int, power: int, warmup_iter: int, warmup_ratio: float):
    return WarmupPolyLR(optimizer, power, max_iter, warmup_iter, warmup_ratio, warmup='linear')
