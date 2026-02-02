import torch
import os
import codecs
from ouro import Ouro
from data_tools import RUNNING_CONFIG, ByteTokenizer 


def chat(model_path: str = "checkpoints/gridman_s_sft.pt", 
         device_str: str = "cuda", 
         temperature: float = 0.38, 
         top_k: int = 10, 
         max_new_bytes: int = 2048):
    # 环境准备
    config = RUNNING_CONFIG
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    tokenizer = ByteTokenizer()
    
    # 特殊字节定义
    EOS_ID = config.tokenizer.eos_token_id
    PAD_ID = config.tokenizer.pad_token_id
    PATCH_SIZE = config.patch_size

    # 模型加载
    print(f"⚡ Gridman Chat Mode | Temp: {temperature} | TopK: {top_k} | Device: {device}")
    model = Ouro(config).to(device)
    
    if os.path.exists(model_path):
        print(f"📂 Loading weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        # 处理可能的 DDP 包装前缀
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print("⚠️ Warning: No checkpoint found, using random weights.")

    model.eval()
    
    _states: tuple[torch.Tensor, ...] = checkpoint['states']
    states = tuple(s.to(device) for s in _states)

    print("\n" + "="*50)
    print("Gridman 启动完毕. 输入 'exit' 退出. ")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("User > ")
            if not user_input: continue
            if user_input.lower() in ['exit', 'quit']: break
        except EOFError:
            break
        
        print("Gridman > ", end="", flush=True)

        # 将用户输入编码为字节 ID
        input_ids = tokenizer.encode(user_input)
        total_len = len(input_ids)
        
        # 准备增量解码器处理 UTF-8 流
        decoder = codecs.getincrementaldecoder("utf-8")(errors='replace')
        
        # 初始化上一轮的输出 Patch 为全 0 (静默态) 以启动递归
        current_input_patch = torch.full((1, 1, PATCH_SIZE), PAD_ID, dtype=torch.long, device=device)
        last_real_byte = PAD_ID
        
        # 前缀注入
        i = 0
        while i < total_len:
            chunk_len = min(PATCH_SIZE, total_len - i)
            user_chunk = input_ids[i : i + chunk_len]
            
            # 构造强制前缀和起始字节
            prefix_tensor = torch.tensor([user_chunk], dtype=torch.long, device=device) # [1, L]
            sos_tensor = torch.tensor([[last_real_byte]], dtype=torch.long, device=device) # [1, 1]
            
            with torch.no_grad():
                _, logits, next_states, _, _ = model(
                    input_patches=current_input_patch, 
                    target_patches=None, 
                    states=states,
                    override_last_tokens=sos_tensor,
                    force_prefix=prefix_tensor,
                    temperature=temperature,
                    top_k=top_k
                )
                states = next_states
            
            # 获取生成的 Patch
            next_patch_ids = torch.argmax(logits[:, 0, :, :], dim=-1) # [1, P]
            patch_list = next_patch_ids[0].cpu().tolist()
            
            # 如果这是一个不完整的末尾 Patch, 打印模型自动补全的部分
            if i + chunk_len >= total_len and chunk_len < PATCH_SIZE:
                generated_part = patch_list[chunk_len:]
                valid_bytes = bytes([b for b in generated_part if b < 256])
                print(decoder.decode(valid_bytes, final=False), end="", flush=True)

            # 更新循环状态
            current_input_patch = next_patch_ids.unsqueeze(1) # [1, 1, P]
            last_real_byte = patch_list[-1]
            i += chunk_len

        # 自由生成 
        generated_count = 0
        stop_generation = False
        
        while generated_count < max_new_bytes and not stop_generation:
            sos_tensor = torch.tensor([[last_real_byte]], dtype=torch.long, device=device)
            
            with torch.no_grad():
                _, logits, next_states, _, _ = model(
                    input_patches=current_input_patch, 
                    target_patches=None, 
                    states=states,
                    override_last_tokens=sos_tensor,
                    force_prefix=None,
                    temperature=temperature,
                    top_k=top_k
                )
                states = next_states

            # 提取生成的 token
            next_patch_ids = torch.argmax(logits[:, 0, :, :], dim=-1)
            patch_list = next_patch_ids[0].cpu().tolist()
            
            # 检查 EOS
            output_patch = []
            for b in patch_list:
                if b == EOS_ID:
                    stop_generation = True
                    break
                if b < 256: # 仅处理有效字节
                    output_patch.append(b)
            
            # 解码并实时打印
            print(decoder.decode(bytes(output_patch), final=stop_generation), end="", flush=True)
            
            # 更新状态
            current_input_patch = next_patch_ids.unsqueeze(1)
            last_real_byte = patch_list[-1]
            generated_count += len(patch_list)
            
            if stop_generation:
                break
        
        print("") # 换行处理下一轮对话


if __name__ == "__main__":
    chat()
