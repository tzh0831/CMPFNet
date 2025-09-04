import torch
import re
from torch import distributed as dist

# 获取分布式训练环境中的当前进程rank和总进程world size的信息
def get_dist_info():
    if dist.is_available():
        # 分布式环境是否初始化
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        # 获取当前进程的等级rank
        rank = dist.get_rank()
        # 获取参与分布式训练的总进程数
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def load_state_dict(module, state_dict, strict=False, logger=None):
    # strict：是否在加载过程中对状态字典的键进行严格匹配
    # 存储状态字典中意外的键
    unexpected_keys = []
    # 存储模型中缺失的键
    all_missing_keys = []
    # 存储错误信息的列表
    err_msg = []
    # 获取名为_metadata的属性（通常存储有关状态字典的元数据，例如保存时模型的结构信息），若属性不存在则返回None
    metadata = getattr(state_dict, '_metadata', None)
    # 对状态字典进行浅拷贝
    state_dict = state_dict.copy()

    if metadata is not None:
        # 确保拷贝的状态字典保留保留原始状态字典的元数据信息
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # prefix：当前模块的路径前缀
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]
    # 收集错误信息
    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')
    # 获取分布式信息
    rank, _ = get_dist_info()
    # rank==0:主进程
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        else:
            print(err_msg)

# 加载预训练模型权重
def load_pretrain(model, filename, strict=False, revise_keys=[(r'^module\.', '')]):
    # revise_keys:定义正则表达式和替换规则，以修改状态字典中的键名
    checkpoint = torch.load(filename)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    # 模型的参数和缓冲区的字典
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    # 用于存储模型的状态
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    # p：正则表达式 r：替换字符串
    for p, r in revise_keys:
        # 对stact_dict中所有的键k应用正则表达式替换操作，将匹配p的部分替换为r
        state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}
    # load state_dict
    load_state_dict(model, state_dict, strict)
    return checkpoint