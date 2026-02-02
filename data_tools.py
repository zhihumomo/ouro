import math
import os
from typing import Callable, Optional, Tuple
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from dataclasses import dataclass
from itertools import chain
from torch.utils.data import Dataset

from datasets import DatasetDict, Dataset as DS


SQRT2 = math.sqrt(2)


# 字节分词器
class ByteTokenizer:
    """
    字节 Tokenizer, 无需训练, 全世界通用
    """
    def __init__(self):
        # 填充符
        self.pad_token_id = 256 
        # 终止符
        self.eos_token_id = 257

    def encode(self, text: str) -> list[int]:
        # 将文本转为 UTF-8 字节列表
        return list(text.encode('utf-8'))

    def decode(self, ids: list[int]) -> str:
        # 过滤特殊 token 并解码
        clean_ids = [i for i in ids if 0 <= i < 256]
        return bytes(clean_ids).decode('utf-8', errors='replace')
    
    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text)}
    

@dataclass
class OscillatorConfig:
    hidden_dim: int = 2         # 2维 (Re, Im)
    frequency: float = 0.05     # 角速度
    amplitude: float = 1.2      # 波动对 Logits 的影响强度
    noise_std: float = 0.15     # 状态噪声
    coupling: float = 0.05      # 输入信号对振荡器的扰动强度


@dataclass
class StateTransformerConfig:
    """
    StateTransformer 配置类
    """
    embed_dim: int
    layers: int
    heads: int
    dff: int

    # 状态与记忆
    states_len: int = 0 
    mem_len: int = 0  

    # Actor 专用
    num_anchors: int = 4
    use_cross_attn: bool = False   
   


class Config:
    """
    主模型配置类
    """
    def __init__(self, embed_dim: int, 
                 heads: int, sensor_layers: int, brain_layers: int, actor_layers: int,
                 wake_steps: int,
                 pretrain_steps: int,
                 sft_steps: int,
                 checkpoint_name: str,
                 data_base='path/datasets',
                 tokenizer=ByteTokenizer()
                 ) -> None:
        # Tokenizer
        self.tokenizer = tokenizer

        # 模型参数
        self.embed_dim = embed_dim
        self.states_len = self.embed_dim // 2

        self.chunk_size = embed_dim
        self.patch_size = embed_dim // 8
        self.byte_embed_dim = 258 
        self.byte_vocab_size = 258                      

        self.heads = heads

        self.wake_steps = wake_steps
        self.sleep_steps = max(self.wake_steps // 2, 1)

        self.bptt_span = 4

        self.cycle_len = self.wake_steps + self.sleep_steps
        self.osc_freq = 2 * math.pi / self.cycle_len  

        self.max_ponder_steps = self.wake_steps + self.sleep_steps

        # 神经振荡器
        self.oscillator_config = OscillatorConfig()

        # Sensor 配置
        self.sensor_config = StateTransformerConfig(
            embed_dim=self.embed_dim,                # 必须与 Brian 一致才能共享 State
            layers=sensor_layers,                    # 轻量级, 做简单的上下文混合
            heads=self.heads, 
            dff=self.embed_dim * 4,
            states_len=self.states_len,              # 与 Brian 共享相同的状态长度
            mem_len=self.states_len * 4              # 与 Brian 共享相同的记忆长度
        )

        # Brain 配置
        self.brain_config = StateTransformerConfig(
            embed_dim=self.embed_dim,                
            layers=brain_layers,                   
            heads=self.heads, 
            dff=self.embed_dim * 4,
            states_len=self.states_len,              
            mem_len=self.states_len * 4    
        )

        # Actor 配置, Actor 只是简单的 HormoneTransformer
        # 这里为了方便依然使用 StateTransformerConfig 配置类
        self.actor_config = StateTransformerConfig(
            embed_dim=self.embed_dim,                
            layers=actor_layers,                   
            heads=self.heads, 
            dff=self.embed_dim * 4,
            use_cross_attn=True,
            num_anchors=self.patch_size // 16
        )

        # 训练配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.warmip_steps = 1000
        
        self.grad_clip_norm = 1.0
        self.seed = 42

        # 数据集路径
        self.data_base = data_base
        self.response_start_token = '\nGridman: '

        # 预训练配置
        self.pretrain_train_file = f'{self.data_base}/pretrain_hq/pretrain_hq_train.jsonl'
        self.pretrain_val_file = f'{self.data_base}/pretrain_hq/pretrain_hq_val.jsonl'
        self.pretrain_steps = pretrain_steps 

        # SFT 配置
        self.sft_train_file = f'{self.data_base}/sft_mini_512/sft_mini_512_train.jsonl'
        self.sft_val_file = f'{self.data_base}/sft_mini_512/sft_mini_512_val.jsonl'
        self.sft_steps = sft_steps
        
        # 日志配置
        self.logging_steps = 1000
        self.val_batches = 50     

        # 保存配置
        self.save_steps = 5000 
        self.checkpoint_dir = './checkpoints'
        self.checkpoint_name = checkpoint_name

        # 学习率配置
        self.base_lr = 3e-4
        self.sft_lr = 3e-6
        self.lr = self.lr_func

    def lr_func(self, update_count=0, is_sft=False):
        """
        返回标准余弦退火(Cosine Annealing)策略下, 当前 update_count 对应的理论学习率
        用于在 Resume 时校准优化器的学习率
        """
        # 确定该阶段的超参数
        if is_sft:
            max_lr = self.sft_lr
            # 计算 SFT 阶段的总更新次数 (Total Updates)
            total_updates = self.sft_steps // self.bptt_span
        else:
            max_lr = self.base_lr
            # 计算 Pretrain 阶段的总更新次数
            total_updates = self.pretrain_steps // self.bptt_span
            
        # 设定最小学习率 
        min_lr = max_lr * 0.01
        
        # 获取 warmup 步数
        warmup_steps = self.warmip_steps 

        # Warmup 阶段
        if update_count < warmup_steps:
            # 避免除以 0
            return max_lr

        # 超过总步数保持最小学习率, 防止崩溃
        if update_count >= total_updates:
            return float(min_lr)

        progress = (update_count - warmup_steps) / (total_updates - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        current_lr = min_lr + (max_lr - min_lr) * cosine_decay
        
        return current_lr


MINI_CONFIG = Config(384, 6, 3, 9, 3, 4, 400000, 240000, 'gridman_mini')
SMALL_CONFIG = Config(512, 8, 4, 12, 4, 4, 400000, 300000, 'gridman_s')
MEDIUM_CONFIG = Config(768, 12, 6, 24, 4, 4, 1200000, 300000, 'gridman_m')
LAGRE_CONFIG = Config(1024, 16, 8, 36, 6, 4, 500000, 300000, 'gridman_l')

RUNNING_CONFIG = LAGRE_CONFIG
    

def preprocess_sft_dataset(dataset: DS | DatasetDict, config: Config, num_proc=20):
    block_byte_size = (config.chunk_size + 1) * config.patch_size 
    
    pad_token_id = config.tokenizer.pad_token_id
    eos_token_id = config.tokenizer.eos_token_id
    ignore_index = -100
    
    USER_PREFIX = "\nUser: "       
    ASST_PREFIX = "\nGridman: " 
    
    def format_and_mask(example):
        if 'conversations' in example:
            conversations = example['conversations']
        else:
            return {"input_ids": [], "labels": []}

        full_ids = []
        labels = []
        
        for msg in conversations:
            role = msg['role']
            content = msg['content']
            
            if role == 'user':
                text_chunk = USER_PREFIX + content
                ids = config.tokenizer.encode(text_chunk)
                full_ids.extend(ids)
                labels.extend([ignore_index] * len(ids))
                
            elif role == 'assistant':
                prefix_ids = config.tokenizer.encode(ASST_PREFIX)
                full_ids.extend(prefix_ids)
                labels.extend([ignore_index] * len(prefix_ids))
                
                content_ids = config.tokenizer.encode(content)
                full_ids.extend(content_ids)
                labels.extend(content_ids)
        
        full_ids.append(eos_token_id)
        labels.append(eos_token_id)
            
        return {
            "input_ids": full_ids,
            "labels": labels
        }

    # 动态检测列名
    if isinstance(dataset, (dict, DatasetDict)):
        column_names = next(iter(dataset.values())).column_names
    else:
        column_names = dataset.column_names
    
    tokenized_datasets = dataset.map(
        format_and_mask, 
        num_proc=num_proc, 
        remove_columns=column_names
    )

    # 分组
    def group_texts(examples: dict):
        concatenated = {k: list(chain(*examples[k])) for k in ["input_ids", "labels"]}
        total_length = len(concatenated["input_ids"])
        
        remainder = total_length % block_byte_size
        if remainder != 0:
            pad_len = block_byte_size - remainder
            concatenated["input_ids"] += [pad_token_id] * pad_len
            concatenated["labels"] += [ignore_index] * pad_len
            total_length += pad_len
        
        result = {
            "input_ids": [concatenated["input_ids"][i : i + block_byte_size] 
                          for i in range(0, total_length, block_byte_size)],
            "labels":    [concatenated["labels"][i : i + block_byte_size] 
                          for i in range(0, total_length, block_byte_size)]
        }
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc)
    return lm_datasets


def preprocess_and_group_bytes(dataset: DS | DatasetDict, config: Config, num_proc=20):
    # 总字节长度 = (Patch数 + 1) * Patch大小
    block_byte_size = (config.chunk_size + 1) * config.patch_size 
    
    pad_token_id = config.tokenizer.pad_token_id
    
    def tokenize_function(examples):
        return {"input_ids": [config.tokenizer.encode(t.replace('<|im_end|>', '')) + [config.tokenizer.eos_token_id] for t in examples["text"]]}
    
    if isinstance(dataset, (dict, DatasetDict)):
        column_names = next(iter(dataset.values())).column_names
    else:
        column_names = dataset.column_names

    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=column_names, 
        num_proc=num_proc
    )
    
    def group_texts(examples: dict):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        
        # 填充 (Padding)
        remainder = total_length % block_byte_size
        if remainder != 0:
            # 计算需要填充的长度
            pad_len = block_byte_size - remainder
            # 对 input_ids 进行填充
            concatenated["input_ids"] += [pad_token_id] * pad_len
            
            # 更新总长度
            total_length += pad_len
        
        result = {
            k: [t[i : i + block_byte_size] for i in range(0, total_length, block_byte_size)]
            for k, t in concatenated.items()
        }
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc)
    return lm_datasets


class BytePatchDataset(Dataset):
    def __init__(self, dataset: DS, patch_size: int):
        self.dataset = dataset
        self.patch_size = patch_size
        self.pad_id = 256  # 定义 PAD ID

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        full_input = torch.tensor(data['input_ids'], dtype=torch.long)
        num_patches = full_input.size(0) // self.patch_size
        
        reshaped_input = full_input[:num_patches*self.patch_size].view(num_patches, self.patch_size)
        input_patches = reshaped_input[:-1] 
        
        if 'labels' in data:
            # SFT 
            full_label = torch.tensor(data['labels'], dtype=torch.long)
            reshaped_label = full_label[:num_patches*self.patch_size].view(num_patches, self.patch_size)
            target_patches = reshaped_label[1:]
        else:
            # Pretrain 
            target_patches = reshaped_input[1:].clone()
            target_patches[target_patches == self.pad_id] = -100
        
        return input_patches, target_patches


class OuroDataLoader:
    """
    数据加载器
    """
    def __init__(self, raw_datasets: DS | DatasetDict, config: Config, 
                 split: str = 'train', global_step: int = 0, is_sft: bool = False,
                 num_workers=20, num_proc=20
                 ):
        
        self.config = config
        self.is_train = (split == 'train')
        
        print(f"📦 Initializing OuroDataLoader for [{split}] (SFT={is_sft})...")

        # 预处理策略路由
        if is_sft:
            processed_dict = preprocess_sft_dataset(raw_datasets, config, num_proc)
        else:
            processed_dict = preprocess_and_group_bytes(raw_datasets, config, num_proc)
        
        hf_dataset = processed_dict[split]
        total_len = len(hf_dataset)
        
        # 仅针对训练集快进
        if self.is_train and global_step > 0:
            # 节律计算
            num_cycles = global_step // config.cycle_len
            remainder = global_step % config.cycle_len
            
            # 计算实际消耗的 Batch 数
            consumed_batches = num_cycles * config.wake_steps + min(remainder, config.wake_steps)
            
            # 处理 Epoch 循环
            start_idx = consumed_batches % total_len
            
            print(f"⏩ Fast-forwarding: Step {global_step} => Consumed {consumed_batches} batches.")
            print(f"✂️  Slicing dataset from index {start_idx} to {total_len} (Skipped {consumed_batches} items total).")
            
            # 切片
            hf_dataset = hf_dataset.select(range(start_idx, total_len))
        
        self.patch_dataset = BytePatchDataset(hf_dataset, config.patch_size)
        
        self.loader = DataLoader(
            self.patch_dataset,
            batch_size=1,
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=self.is_train
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)
    

def states_clone(states: Tuple[Optional[torch.Tensor], ...], need_clone=True):
    res = []
    for state in states:
        if state is not None:
            state: torch.Tensor
            if need_clone:
                res.append(state.clone().detach())
            else:
                res.append(state.detach())
        else:
            res.append(None)
    return tuple(res)
    

def save_checkpoint(
    step: int, 
    update_count: int, 
    model: nn.Module, 
    states: tuple[torch.Tensor, ...], 
    path: str,
    override_model_dict=None
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 将 Tuple 状态转为 CPU
    states_cpu = tuple(s.detach().cpu() for s in states)
    
    # 修正后的权重字典, 否则获取当前模型的
    model_dict = override_model_dict if override_model_dict is not None else model.state_dict()
    
    checkpoint_dict = {
        'step': step,
        'update_count': update_count,
        'model': model_dict,
        'states': states_cpu,
        # 保存随机数状态
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }

    torch.save(checkpoint_dict, path)
    print(f'💾 Saved checkpoint to {path} (Step {step})')


def load_checkpoint(
    path: str, 
    model: nn.Module, 
    optimizer: torch.optim.AdamW, 
    device: torch.device,
    lr: Callable[[int, bool], float]
):
    if not os.path.exists(path): 
         # 将 target_lr 应用到优化器
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr(0)
            param_group['initial_lr'] = lr(0)
        return 0, 0, None
        
    print(f"📂 Loading checkpoint from {path}")
    ckpt: dict = torch.load(path, map_location=device)
    
    # 加载模型
    model.load_state_dict(ckpt['model'])

    # 获取步数, 更新次数
    steps: int = ckpt['step']
    update_count: int = ckpt['update_count']

    # 将 target_lr 应用到新优化器
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr(update_count)
        param_group['initial_lr'] = lr(update_count)
    
    # 恢复随机状态
    if 'rng_state' in ckpt:
        torch.set_rng_state(ckpt['rng_state'].cpu().byte())
    if 'cuda_rng_state' in ckpt and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu().byte() for s in ckpt['cuda_rng_state']])
        except: pass

    _states = tuple(s.to(device) for s in ckpt['states'])
    return steps, update_count, _states