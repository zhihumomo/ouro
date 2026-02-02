import math
import os

os.environ["HF_DATASETS_CACHE"] = "path/cache"

from typing import Optional, Tuple
import torch
import time
import sys 

from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset

from data_tools import RUNNING_CONFIG, ByteTokenizer,  OuroDataLoader

from ouro import Ouro
from data_tools import save_checkpoint, load_checkpoint, states_clone
from tools.watch import TrainingVisualizer

from datetime import datetime


torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True


def print_model_parameters(model: Ouro):
    """打印模型参数统计信息"""
    total_params = 0
    trainable_params = 0
    buffer_params = 0
    
    # 遍历所有参数
    for _, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    # 遍历所有buffer
    for _, buffer in model.named_buffers():
        num_buffer = buffer.numel()
        buffer_params += num_buffer
    
    # 打印汇总信息
    print("\n" + "="*60)
    print(f"Gridman 参数统计: {(total_params+buffer_params)/1e6:.2f} M")
    print(f"        状态大小: {2.5 * model.config.embed_dim**2/1e6:.2f} M")
    print("="*60 + "\n")


def output_error_data(
        step: int, 
        current_span_avg_loss: float, 
        running_avg_loss: float,
        span_input_buffer: int,
        tokenizer: ByteTokenizer,
        error_data_dir='error_data'
        ):
    """
    输出 error_data
    """
    os.makedirs(error_data_dir, exist_ok=True)

    timestamp = int(time.time())
    err_file = os.path.join(error_data_dir, f"error_step_{step}_{timestamp}.txt")

    with open(err_file, "w", encoding="utf-8", errors='replace') as f:
        f.write(f"Step: {step}\n")
        f.write(f"Loss: {current_span_avg_loss}\n")
        f.write(f"Running Avg Loss: {running_avg_loss}\n")
        f.write(f"BPTT Span Length: {len(span_input_buffer)}\n\n")
        f.write("=== Input Data Dump (Full Batch Reconstructed) ===\n")
        
        # batch_input shape: [Batch, Seq, Patch]
        current_batch_size = span_input_buffer[0].size(0)
        
        for b_idx in range(current_batch_size):
            f.write(f"\n>>> [Batch Sample Index: {b_idx}] <<<\n")
            f.write("-" * 20 + "\n")
            
            # 收集该样本在整个 BPTT Span 内的所有字节
            full_span_bytes = []
            for step_tensor in span_input_buffer:
                # step_tensor: [B, S, P] -> 取出第 b_idx 个样本 -> [S, P]
                # 展平为字节序列
                flat_bytes = step_tensor[b_idx].view(-1).tolist()
                full_span_bytes.extend(flat_bytes)
            
            # 解码
            decoded_text = tokenizer.decode(full_span_bytes)
            f.write(decoded_text)
            
            f.write("\n" + "=" * 40 + "\n")



def evaluate(model: Ouro, val_loader: OuroDataLoader, current_states: Tuple[Optional[torch.Tensor], ...], config=RUNNING_CONFIG):
    model.eval()
    
    # 状态克隆
    val_states = states_clone(current_states)
    
    # 备份隐式 Buffer
    backup_hormone = model.hypothalamus.current_hormone.clone().detach()
    backup_self = model.self_encoder.current_self.clone().detach()
    
    total_loss = 0.0
    total_task_loss = 0.0
    steps = 0
    
    with torch.no_grad():
        for i, (input_patches, target_patches) in enumerate(val_loader):
            if i >= config.val_batches: 
                break
            input_patches: torch.Tensor
            target_patches: torch.Tensor

            input_patches = input_patches.to(config.device)
            target_patches = target_patches.to(config.device)
            
            # 前向传播
            result = model(
                input_patches, 
                states=val_states,  
                target_patches=target_patches, 
                is_sleeping_phase=False
            )
            
            loss: torch.Tensor
            task_loss: torch.Tensor

            loss, _, next_states, _, task_loss = result
            
            # 状态演化
            val_states = next_states
            
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            steps += 1
            
    avg_loss = total_loss / (steps + 1e-9)
    avg_task_loss = total_task_loss / (steps + 1e-9)
    
    # 将 buffer 水平恢复到验证前的状态
    model.hypothalamus.current_hormone.copy_(backup_hormone)
    model.self_encoder.current_self.copy_(backup_self)
    
    # 切回训练模式
    model.train() 
    return avg_loss, avg_task_loss


def main(stage='sft', config=RUNNING_CONFIG):
    # 基础设置和变量
    torch.manual_seed(config.seed)
    print(f"Using config.device: {config.device} | Gridman Sleep-Awake Architecture")

    visualizer = TrainingVisualizer()

    if stage == 'pretrain':
        print("🚀 Mode: PRE-TRAINING")
        train_file = config.pretrain_train_file
        val_file = config.pretrain_val_file
        max_steps = config.pretrain_steps
        ckpt_name = f'{config.checkpoint_name}.pt'
        is_sft = False
    else:
        print("🚀 Mode: SUPERVISED FINE-TUNING (SFT)")
        train_file = config.sft_train_file
        val_file = config.sft_val_file
        max_steps = config.sft_steps
        ckpt_name = f'{config.checkpoint_name}_sft.pt'
        is_sft = True

    # 模型初始化
    model = Ouro(config).to(config.device)
    print_model_parameters(model)
    model = torch.compile(model, mode='default') 

    model: Ouro
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0, fused=True) # 学习率在下面的 load_checkpoint 中重置
    
    total_updates = max_steps // config.bptt_span

    ckpt_path = os.path.join(config.checkpoint_dir, ckpt_name)
    pretrain_ckpt_path = os.path.join(config.checkpoint_dir, f'{config.checkpoint_name}.pt')

    warmip_steps = config.warmip_steps if not os.path.exists(ckpt_path) else 250

    start_step, update_counter, states = load_checkpoint(
        ckpt_path, model, optimizer, config.device, lambda update_counts: config.lr(update_counts, is_sft) 
    )

    remaining_updates = total_updates - update_counter

    if remaining_updates <= 0:
        print('Training already finished!')
        return None
        
    steps_for_scheduler = remaining_updates + warmip_steps

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmip_steps, 
        num_training_steps=steps_for_scheduler
    )

    # 数据准备
    tokenizer = ByteTokenizer()

    data_files = {"train": train_file, "validation": val_file}
    raw_datasets = load_dataset("json", data_files=data_files)

    train_loader = OuroDataLoader(
        raw_datasets, config,
        split='train', 
        global_step=start_step, 
        is_sft=is_sft
    )
    
    val_loader = OuroDataLoader(
        raw_datasets, config, 
        split='validation', 
        is_sft=is_sft
    )
    
    data_iter = iter(train_loader)

    print(f"🔄 Resumed from step {start_step}")
    
    if is_sft and start_step == 0: # SFT 初始化
        if os.path.exists(pretrain_ckpt_path):
            print("📥 Loading Pre-trained weights for SFT...")
            _, _, states = load_checkpoint(pretrain_ckpt_path, model, optimizer, config.device, lambda update_counts: config.lr(update_counts, True))
        else:
            print("⚠️ Warning: No pre-trained weights found!")

        update_counter = 0

        # 使用新的优化器
        optimizer = torch.optim.AdamW(model.parameters(), config.lr(is_sft=True), fused=True)
    
    if states is None: 
        states = model.get_init_states()

    model.train()

    current_cycle_pos = model.cycle_counter.item() % config.cycle_len
    if current_cycle_pos < config.wake_steps:
        accumulated_steps = current_cycle_pos
    else:
        accumulated_steps = 0
    
    total_loss_tensor = 0.0
    span_loss_scalar = 0.0
    span_task_loss_scalar = 0.0
    current_span_tasks = 0

    logging_loss_accumulator = 0.0
    logging_task_loss_accumulator = 0.0

    logging_span_count = 0

    running_avg_loss = None  # 指数移动平均 Loss
    loss_alpha = 0.95        # 平滑系数
    
    # 用于回滚的备份
    backup_states = None     # 备份 (mem, state, osc)
    backup_hormone = None    # 备份激素
    backup_self = None
    span_input_buffer = []   # 备份当前 Span 的输入文本 (用于 Dump)
    
    print("🚀 Training Start (With Anomaly Detection)...")
    
    # 第一次循环备份
    if backup_states is None:
        backup_states = states_clone(states)
        backup_hormone = model.hypothalamus.current_hormone.clone().detach()
        backup_self = model.self_encoder.current_self.clone().detach()

    for step in range(start_step + 1, max_steps + 1):
        current_cycle_pos = model.cycle_counter.item() % config.cycle_len
        is_sleep_mode = current_cycle_pos >= config.wake_steps

        input_patches: Optional[torch.Tensor] = None
        target_patches: Optional[torch.Tensor] = None
        
        if not is_sleep_mode:
            try:
                input_patches, target_patches = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                input_patches, target_patches = next(data_iter)
            
            input_patches = input_patches.to(config.device, non_blocking=True)
            target_patches = target_patches.to(config.device, non_blocking=True)

            # 将输入加入缓冲,用于错误转储
            span_input_buffer.append(input_patches.detach().cpu())

        dtype = torch.bfloat16
        
        # 前向传播
        with torch.amp.autocast('cuda', dtype=dtype):
            result = model(
                input_patches, 
                states=states,  
                target_patches=target_patches, 
                is_sleeping_phase=is_sleep_mode
            )
            loss, _, next_states, is_sleeping_internal, task_loss = result

        assert is_sleep_mode == is_sleeping_internal
        
        # 清醒模式
        if not is_sleep_mode:
            loss_scaled = loss / config.bptt_span
            total_loss_tensor += loss_scaled
            span_loss_scalar += loss.item()
            span_task_loss_scalar += task_loss.item()
            current_span_tasks += 1
            accumulated_steps += 1

            visualizer.update(step, loss.item(), task_loss.item())
            
            # 检查是否到达 BPTT 结尾
            if accumulated_steps % config.bptt_span == 0:
                current_span_avg_loss = span_loss_scalar / config.bptt_span
                current_span_avg_task_loss = span_task_loss_scalar / config.bptt_span

                is_nan = torch.isnan(torch.tensor(current_span_avg_loss)) or torch.isinf(torch.tensor(current_span_avg_loss))
                
                if is_nan:
                    print(f"\n⚠️ [Anomaly Detected] Step {step}: Loss nan")
                    print("🔄 Rolling back states and skipping update...\n")
                    
                    # 导出错误数据
                    output_error_data(step, current_span_avg_loss,
                                    running_avg_loss, span_input_buffer, tokenizer)
                    
                    dirty_state_dict = model.state_dict()
                    
                    # 确定 key 的前缀 (compile 会增加 _orig_mod 前缀)
                    prefix = "_orig_mod." if any(k.startswith("_orig_mod.") for k in dirty_state_dict.keys()) else ""
                    
                    hormone_key = f"{prefix}hypothalamus.current_hormone"
                    self_key = f"{prefix}self_encoder.current_self"
                    cycle_key = f"{prefix}cycle_counter"
                    
                    # 使用备份数据写回
                    dirty_state_dict[hormone_key] = backup_hormone
                    dirty_state_dict[self_key] = backup_self
                    dirty_state_dict[cycle_key] = dirty_state_dict[cycle_key] - config.bptt_span

                    save_checkpoint(
                        step=step,
                        update_count=update_counter,
                        model=model, 
                        states=backup_states,
                        path=ckpt_path,
                        override_model_dict=dirty_state_dict 
                    )
                    
                    print("✅ Emergency checkpoint saved. Skipping bad batch.")
                    print("🔄 Calling run.sh to restart training process...")
                    
                    sys.exit(1)
                else:
                    total_loss_tensor.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    update_counter += 1

                    logging_loss_accumulator += current_span_avg_loss
                    logging_task_loss_accumulator += current_span_avg_task_loss
                    logging_span_count += 1
                    
                    # 正常截断与状态传递
                    states = states_clone(next_states, need_clone=False)
                    model.detach_internal_states()

                    # 更新 Running Avg Loss
                    if running_avg_loss is None or math.isnan(running_avg_loss):
                        running_avg_loss = current_span_avg_loss
                    else:
                        running_avg_loss = loss_alpha * running_avg_loss + (1 - loss_alpha) * current_span_avg_loss
                    
                    # 日志与评估
                    if update_counter % config.logging_steps == 0:
                        val_loss, val_task_loss = evaluate(model, val_loader, states)
                        avg_logging_loss = logging_loss_accumulator / max(1, logging_span_count)
                        avg_logging_task_loss = logging_task_loss_accumulator / max(1, logging_span_count)
                        
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        visualizer.plot()
                        
                        print(f"--{current_time} "
                              f"Step {step} (Upd {update_counter}) "
                              f"Loss: {avg_logging_loss:.4f} (Avg: {running_avg_loss:.4f}) | "
                              f"Task Loss: {avg_logging_task_loss:.4f} | "
                              f"Val Loss: {val_loss:.4f} | "
                              f"Val Task Loss: {val_task_loss:.4f}")
                              
                        # 重置日志区间平均 loss
                        logging_loss_accumulator = 0.0
                        logging_task_loss_accumulator = 0.0
                        logging_span_count = 0
                    
                    # 重置 
                    total_loss_tensor = 0.0 
                    span_loss_scalar = 0.0
                    span_task_loss_scalar = 0.0
                    current_span_tasks = 0
                    span_input_buffer = [] # 清空缓冲

                # 为下一个 Span 建立新的快照
                backup_states = states_clone(states)
                backup_hormone = model.hypothalamus.current_hormone.clone().detach()

            else:
                # BPTT Span 中间步骤,直接传递状态
                states = next_states

        else:
            # 睡眠模式处理
            states = next_states
           

        # 保存检查点
        if step % config.save_steps == 0:
            save_checkpoint(step, update_counter, model, states, ckpt_path)


if __name__ == "__main__":
    main()