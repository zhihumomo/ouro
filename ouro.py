import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional
from torch.utils.checkpoint import checkpoint
from data_tools import Config, StateTransformerConfig, SQRT2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, timescale: float = 100000.0):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(timescale) / d_model))
        self.div_term: torch.Tensor
        self.register_buffer('div_term', div_term)

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() == 1:
            indices = indices.unsqueeze(0).expand(x.size(0), -1)
            
        pos_expanded = indices.unsqueeze(-1).float()
        
        # 计算相位
        phase = pos_expanded * self.div_term
        
        # 构造 PE 向量并注入
        pe = torch.zeros_like(x)
        pe[:, :, 0::2] = torch.sin(phase)
        pe[:, :, 1::2] = torch.cos(phase)
        
        return x + pe


class NeuralOscillator(nn.Module):
    """
    神经节律器
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        oscillator_config = self.config.oscillator_config

        self.noise = oscillator_config.noise_std
        self.amp = oscillator_config.amplitude

    def forward(self, prev_state: torch.Tensor, current_cycle_len: int):
        # fp32
        with torch.amp.autocast('cuda', enabled=False):
            prev_state_fp32 = prev_state.float()
            
            # 计算当前步进的角速度
            omega = 2 * math.pi / current_cycle_len
            
            cos_w = math.cos(omega)
            sin_w = math.sin(omega)
            
            # 动态构建矩阵 [2, 2]
            rot_matrix = torch.tensor([
                [cos_w, -sin_w],
                [sin_w,  cos_w]
            ], device=prev_state.device, dtype=torch.float32)
            
            # 旋转
            state_rot = prev_state_fp32 @ rot_matrix
            
            # 噪声
            noise = torch.randn_like(state_rot) * self.noise
            state_rot = state_rot + noise
            
            # 归一化
            norm = torch.norm(state_rot, dim=1, keepdim=True) + 1e-6
            next_state_fp32 = state_rot / norm

        return next_state_fp32.to(prev_state.dtype)

    def get_init_state(self):
        # 初始化
        theta = torch.rand(1, device=self.config.device, dtype=torch.float32) * 2 * math.pi
        x = torch.cos(theta)
        y = torch.sin(theta)
        return torch.stack([x, y], dim=1)
    

class Hippocampus(nn.Module):
    """
    海马体: 负责睡眠阶段梦境生成和记忆更新
    """
    def __init__(self, config: Config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.dream_seq_len = config.embed_dim
        
        # Inception 组件, 梦境生成 
        self.osc_proj = nn.Linear(2, self.embed_dim)
        self.dream_anchors = nn.Parameter(torch.randn(1, self.dream_seq_len, self.embed_dim))
        self.dream_attention = nn.MultiheadAttention(self.embed_dim, num_heads=config.brain_config.heads, batch_first=True, dropout=0.1)

        self.dream_state_norm = nn.LayerNorm(self.embed_dim)
        self.dream_mem_norm = nn.LayerNorm(self.embed_dim)
        self.dream_norm = nn.LayerNorm(self.embed_dim)
        
        # Consolidate 组件
        # State -> Mem 
        self.mem_input_norm = nn.LayerNorm(self.embed_dim)
        self.mem_attn = nn.MultiheadAttention(self.embed_dim, num_heads=config.heads, batch_first=True)
        self.mem_gate = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.Sigmoid()
        )
        self.mem_update_norm = nn.LayerNorm(self.embed_dim)

        # Mem -> State
        self.state_input_norm = nn.LayerNorm(self.embed_dim)
        self.state_attn = nn.MultiheadAttention(self.embed_dim, num_heads=config.heads, batch_first=True)
        self.state_gate = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.Sigmoid()
        )
        self.state_update_norm = nn.LayerNorm(self.embed_dim)
        
        self.noise_scale = 0.05

    def inception(self, mem: torch.Tensor, state: torch.Tensor, osc_state: torch.Tensor) -> torch.Tensor:
        """
        梦境生成: 根据当前节律和记忆生成伪输入序列
        """
        B = mem.size(0)

        # 归一化
        mem = self.dream_mem_norm(mem)
        state = self.dream_state_norm(state)

        context = torch.cat([mem, state], dim=1) 
        
        rhythm_tone = self.osc_proj(osc_state).unsqueeze(1)
        dream_queries = self.dream_anchors.expand(B, -1, -1) + rhythm_tone
        
        dream_seq, _ = self.dream_attention(query=dream_queries, key=context, value=context, need_weights=False)
        
        noise = torch.randn_like(dream_seq) * self.noise_scale
        dream_seq = self.dream_norm(dream_seq + noise)
        
        return F.normalize(input=dream_seq, p=2, dim=-1, eps=1e-5)

    def consolidate(self, mem: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        记忆固化: 睡眠时调用，双向更新 Mem 和 State
        """
        # 归一化
        mem = self.mem_input_norm(mem)
        state = self.state_input_norm(state)

        # Mem 更新: Q=Mem, K=State, V=State
        delta_mem, _ = self.mem_attn(query=mem, key=state, value=state, need_weights=False)
        
        gate_in_m = torch.cat([mem, delta_mem], dim=-1)
        z_m = self.mem_gate(gate_in_m)
        
        # mem 更新
        new_mem = self.mem_update_norm((1 - z_m) * mem + z_m * delta_mem)
        
        # State 更新: Q=State, K=New_Mem, V=New_Mem
        delta_state, _ = self.state_attn(query=state, key=new_mem, value=new_mem, need_weights=False)
        
        # 注入噪声
        noise = torch.randn_like(delta_state) * self.noise_scale
        
        gate_in_s = torch.cat([state, delta_state], dim=-1)
        z_s = self.state_gate(gate_in_s)
        
        new_state = self.state_update_norm((1 - z_s) * state + z_s * (delta_state + noise))

        new_state = F.normalize(new_state, p=2, dim=-1, eps=1e-5)
        new_mem = F.normalize(new_mem, p=2, dim=-1, eps=1e-5)
        return new_mem, new_state
    

class Hypothalamus(nn.Module):
    """
    下丘脑: 根据状态分泌激素实现状态对整个网络的控制
    """
    def __init__(self, config: Config):
        super().__init__()
        # 维度: 状态/梦境内容的维度 (D) + 节律器维度 (2)
        total_input_dim = config.embed_dim + 2
        self.hormone_dim = config.embed_dim // 4

        self.context_norm = nn.LayerNorm(config.embed_dim)
        
        # 激素网络
        self.network = nn.Sequential(
            nn.Linear(total_input_dim, total_input_dim),
            nn.LayerNorm(total_input_dim),
            nn.SiLU(), 
            nn.Linear(total_input_dim, self.hormone_dim),
            nn.Tanh(),  
            nn.Linear(self.hormone_dim, self.hormone_dim),
            nn.LayerNorm(self.hormone_dim),
            nn.Tanh() 
        )

        self.buffer_norm = nn.LayerNorm(self.hormone_dim)
        
        # 激素缓存
        self.current_hormone: torch.Tensor
        self.register_buffer("current_hormone", torch.zeros(1, 1, self.hormone_dim))

    def get_hormone(self, state_input: torch.Tensor, osc_state: torch.Tensor) -> torch.Tensor:
        # 状态压缩
        if state_input.dim() == 3:
            context = state_input.mean(dim=1) 
        else:
            context = state_input

        # 归一化
        context = self.context_norm(context)
        
        # 节律
        combined_input = torch.cat([context, osc_state], dim=1) 
        
        # 计算激素调节量
        _hormone_delta: torch.Tensor = self.network(combined_input)
        hormone_delta = _hormone_delta.unsqueeze(1)

        hormone_delta = F.normalize(input=hormone_delta, p=2, dim=-1, eps=1e-5)
        
        # 当前的激素水平 = 过去水平 * 0.95 + 新调节量 * 0.05
        new_hormone = self.buffer_norm(0.9 * self.current_hormone + 0.1 * hormone_delta)
        
        # 更新内部缓存
        self.current_hormone = F.normalize(input=new_hormone, p=2, dim=-1, eps=1e-5)
            
        return self.current_hormone
    

class HormoneReceptor(nn.Module):
    """
    激素受体: 通用层包装器, 使得输出受到激素调节
    """
    def __init__(self, backbone: nn.Module, config: StateTransformerConfig):
        super().__init__()
        self.backbone = backbone
    
        self.dim = config.embed_dim
        self.hormone_dim = self.dim // 4

        # 动态推断输入输出维度
        if isinstance(backbone, nn.Linear):
            self.in_dim = backbone.in_features
            self.out_dim = backbone.out_features
        else:
            self.in_dim = config.embed_dim
            self.out_dim = config.embed_dim
            
        # Query 生成器
        self.compress_factor = self.in_dim // self.dim
        self.query_norm = nn.LayerNorm(self.dim)

        # Output 投影器 
        self.expand_factor = self.out_dim // self.dim
        self.proj_gate = nn.Parameter(torch.randn(self.expand_factor, self.dim))
        self.proj_norm = nn.LayerNorm(self.out_dim)

        # 激素转化网络
        self.hormone_network = nn.Sequential(
            nn.Linear(self.hormone_dim, self.hormone_dim),
            nn.SiLU(), 
            nn.Linear(self.hormone_dim, self.dim),
            nn.LayerNorm(self.dim)
        )

    def _generate_query(self, x: torch.Tensor, hormone: torch.Tensor) -> torch.Tensor:
        """
        从输入 x 生成查询向量
        """
        B, S, _ = x.shape
        x_folded = x.view(B, S, self.compress_factor, self.dim)
        
        # 加权求和 
        query = x_folded.sum(dim=2) 

        hormone = self.hormone_network(hormone)
            
        return self.query_norm(query + hormone)

    def _project_output(self, query: torch.Tensor) -> torch.Tensor:
        """
        将 query 投影回输出空间
        """
        B, S, _ = query.shape
        out_expanded = query.unsqueeze(2).expand(B, S, self.expand_factor, self.dim) * self.proj_gate
       
        injection = out_expanded.reshape(B, S, -1) 
        active = torch.tanh(self.proj_norm(injection))
            
        return active

    def forward(self, x: torch.Tensor, hormone: torch.Tensor) -> torch.Tensor:
        # 主干计算
        y_logic = self.backbone(x) 
        
        # 生成 Query
        query = self._generate_query(x, hormone)
        
        # 投影回输出 
        active = self._project_output(query)
        y = y_logic * active
      
        return y


class HormoneTransformerLayer(nn.Module):
    """
    支持激素注入的 Transformer Layer
    """
    def __init__(self, config: StateTransformerConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.use_cross_attn = config.use_cross_attn 
        
        # Attention
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = nn.MultiheadAttention(config.embed_dim, config.heads, batch_first=True, dropout=0.1)

        # 交叉注意力
        if self.use_cross_attn:
            self.norm_cross = nn.LayerNorm(config.embed_dim)
            self.cross_attn = nn.MultiheadAttention(
                    embed_dim=config.embed_dim, 
                    num_heads=config.heads, 
                    kdim=config.embed_dim,  
                    vdim=config.embed_dim,  
                    batch_first=True, 
                    dropout=0.1
                )
        
        # FFN & Norm 使用 HormoneReceptor 包裹
        self.norm2 = HormoneReceptor(nn.LayerNorm(config.embed_dim), config)
        
        self.linear1 = HormoneReceptor(nn.Linear(config.embed_dim, config.dff), config)
        self.linear2 = HormoneReceptor(nn.Linear(config.dff, config.embed_dim), config)
        
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, hormone: torch.Tensor, src_mask=None, context: Optional[torch.Tensor]=None):
        # Attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=src_mask, need_weights=False)
        x = residual + self.dropout(x)

        # 交叉注意力
        if self.use_cross_attn and context is not None:
            residual = x
            x = self.norm_cross(x)
            x, _ = self.cross_attn(query=x, key=context, value=context, need_weights=False)
            x = residual + self.dropout(x)
        
        # FFN 
        residual = x
        
        # 注入激素到 Norm2
        x = self.norm2(x, hormone)
        
        # 注入激素到 FFN
        x = self.linear1(x, hormone)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x, hormone)
        
        return residual + x
    

class SelfEncoder(nn.Module):
    """
    显式的自我表征, 视作稳定的人格锚点
    """
    def __init__(self, config: Config):
        super().__init__()
        self.self_dim = config.embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.self_dim, self.self_dim // 2),
            nn.GELU(),
            nn.Linear(self.self_dim // 2, self.self_dim),
            nn.Tanh()
        )

        self.norm = nn.LayerNorm(self.self_dim)

        # 持久化
        self.current_self: torch.Tensor
        self.register_buffer(
            "current_self",
            torch.zeros(1, 1, self.self_dim)
        )

    def forward(self, state: torch.Tensor, need_update_self=False):
        """
        state: [B, S, D]
        """
        if need_update_self:
            # 压缩 state → 全局自我描述
            state_summary = state.mean(dim=1)
            delta_self = self.encoder(state_summary).unsqueeze(1)

            # 更新
            new_self = 0.95 * self.current_self + 0.05 * delta_self
            self.current_self = self.norm(new_self)

        return self.current_self


class StateTransformer(nn.Module):
    """
    结构: [Mem | State_Read | Input | State_Write]
    序列总长度 4 * embed_dim, Input 最大长度 embed_dim
    """
    def __init__(self, config: StateTransformerConfig, self_encoder: SelfEncoder):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim

        # 自我表征
        self.self_encoder = self_encoder
        
        # 输入归一化 
        self.input_norm = nn.LayerNorm(config.embed_dim)
        self.state_norm = nn.LayerNorm(config.embed_dim) 
        self.mem_norm = nn.LayerNorm(config.embed_dim)
        
        # 位置编码
        self.max_logical_pos = config.states_len * 8
        
        self.pos_encoder = PositionalEncoding(config.embed_dim)
        
        # 状态槽位 Embedding, 区分同一个 State 在 Read 和 Write 位置的不同角色
        self.read_slot_emb = nn.Parameter(torch.randn(1, config.states_len, config.embed_dim))
        self.write_slot_emb = nn.Parameter(torch.randn(1, config.states_len, config.embed_dim))
        nn.init.normal_(self.read_slot_emb, std=0.02)
        nn.init.normal_(self.write_slot_emb, std=0.02)
        
        self.layers = nn.ModuleList([
            HormoneTransformerLayer(config) 
            for _ in range(config.layers)
        ])

        # 输出归一化
        self.text_output_norm = HormoneReceptor(nn.LayerNorm(config.embed_dim), config)
        self.state_output_norm = HormoneReceptor(nn.LayerNorm(config.embed_dim), config)
        
    def _create_sandwich_mask(self, total_len, input_len, device):
        """
        创建掩码:
        Prefix (Mem + Read): 内部全互联
        Input: Causal (看 Prefix + 之前的 Input)
        Write: Full Context (看前面所有)
        """
        end_read = self.config.mem_len + self.config.states_len
        end_input = end_read + input_len
        
        # 基础 Causal Mask
        mask = torch.ones((total_len, total_len), device=device, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)
        
        # Prefix 
        mask[:end_read, :end_read] = False
        
        # Write 
        mask[end_input:, :] = False 
        
        return mask

    def forward(self, x_chunk: torch.Tensor, prev_memory: torch.Tensor, prev_state: torch.Tensor, hormone: torch.Tensor, need_update_self=False) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        
        _, inputs_len, _ = x_chunk.shape
        device = x_chunk.device
        
        # 归一化
        x_input_norm = self.input_norm(x_chunk)
        mem_norm = self.mem_norm(prev_memory)
        state_norm = self.state_norm(prev_state)
        
        # Mem
        part_mem = mem_norm
        
        # State Read 
        part_read = state_norm + self.read_slot_emb
        
        # Input
        part_input = x_input_norm
        
        # State Write
        self_encode: torch.Tensor = self.self_encoder(state_norm, need_update_self)
        part_write = state_norm + self.write_slot_emb + self_encode.expand(-1, self.config.states_len, -1)
        
        # 拼接
        combined_input = torch.cat([part_mem, part_read, part_input, part_write], dim=1)
        
        # 位置索引
        
        # Mem: [-M ... -1]
        idx_mem = torch.arange(-self.config.mem_len, 0, dtype=torch.float, device=device)
        # Read: [0 ... 0]
        idx_read = torch.zeros(self.config.states_len, dtype=torch.float, device=device)
        # Input: [1 ... L]
        idx_input = torch.arange(1, inputs_len + 1, dtype=torch.float, device=device)
        # Write: [Fixed ... Fixed]
        idx_write = torch.full((self.config.states_len,), self.max_logical_pos, dtype=torch.float, device=device)
        
        all_indices = torch.cat([idx_mem, idx_read, idx_input, idx_write])
        
        # 注入位置编码与 Transformer 计算 
        hidden = self.pos_encoder(combined_input, all_indices)
        
        total_len = hidden.size(1)
        mask = self._create_sandwich_mask(total_len, inputs_len, device)
        
        for layer in self.layers:
            hidden = checkpoint(layer, hidden, hormone, mask, use_reentrant=False)
            
        # 输出
        start_input = self.config.mem_len + self.config.states_len
        end_input = start_input + inputs_len
        
        # 截取 Text Output
        text_output_raw = hidden[:, start_input : end_input, :]
        
        # 截取 New State
        new_state_raw = hidden[:, end_input:, :]
        
        # 输出归一化
        text_output = self.text_output_norm(text_output_raw, hormone)
        text_output = F.normalize(text_output, p=2, dim=-1, eps=1e-5)
        
        new_state = self.state_output_norm(new_state_raw, hormone)
        new_state = F.normalize(new_state, p=2, dim=-1, eps=1e-5)
        
        # Mem 在此前向过程中不产生新值，保持只读
        return text_output, new_state
    

class Compressor(nn.Module):
    """
    压缩器: 将字节流压缩为向量
    """
    def __init__(self, patch_size: int, byte_emb_dim: int, model_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.byte_emb_dim = byte_emb_dim
        
        # 捕捉局部字节组合
        self.resonator = nn.Sequential(
            nn.Conv1d(
                in_channels=byte_emb_dim, 
                out_channels=model_dim * 2, # 升维
                kernel_size=8,              # 感受野
                padding=2,
                groups=1                    # 全通道混合
            ),
            nn.GELU(),
            nn.BatchNorm1d(model_dim * 2)   
        )

        # 压缩回原维度
        self.condenser = nn.Sequential(
            nn.Conv1d(model_dim * 2, model_dim, kernel_size=1),
            nn.GELU()
        )

        # 位置编码
        self.pos_emb = nn.Parameter(torch.randn(1, model_dim, patch_size)) 
        nn.init.normal_(self.pos_emb, std=0.02)

        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, kernel_size=1), 
            nn.GELU(),
        )

        # Start + Max + End -> model_dim
        self.boundary_fusion = nn.Sequential(
            nn.Linear(model_dim * 3, model_dim * 2), 
            nn.LayerNorm(model_dim * 2),             
            nn.GELU(),                               
            nn.Linear(model_dim * 2, model_dim)      
        )

        nn.init.xavier_uniform_(self.boundary_fusion[-1].weight, gain=0.1)

        # 输出投影
        self.proj_norm = nn.LayerNorm(model_dim)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        B, S, _ = x_flat.shape
        
        # [B*S, Byte_Emb, Patch_Size]
        x = x_flat.view(B * S, self.patch_size, self.byte_emb_dim).transpose(1, 2)
        
        x = self.resonator(x) 
        x = self.condenser(x)
        L = x.size(2)
        
        feat_start = x[:, :, 0]

        pos = self.pos_emb[:, :, :L]
        x_pos = x + pos
        x_pos = self.fusion_mlp(x_pos)

        feat_body = x_pos.mean(dim=-1)

        feat_end = x[:, :, -1]
        x_combined = torch.cat([feat_start, feat_body, feat_end], dim=-1)

        x = self.boundary_fusion(x_combined)
        
        # 归一化
        x = self.proj_norm(x)
        return x.view(B, S, -1)


class Sensor(nn.Module):
    """
    Sensor: 感受器, 将输入转化成状态扰动
    """
    def __init__(self, config: Config, self_encoder: SelfEncoder):
        super().__init__()
        self.pad_token_id = config.tokenizer.pad_token_id
        self.patch_size = config.patch_size
        self.byte_emb_dim = config.byte_embed_dim
        self.model_dim = config.embed_dim 
        
        self.byte_embedding = nn.Embedding(config.byte_vocab_size, self.byte_emb_dim)

        # 基础压缩组件
        self.compressor = Compressor(
            patch_size=self.patch_size,
            byte_emb_dim=self.byte_emb_dim,
            model_dim=self.model_dim
        )
        
        # 使用共享记忆的 StateTransformer 作为上下文混合器
        self.context_mixer = StateTransformer(config.sensor_config, self_encoder)
        
        # 初始化
        nn.init.normal_(self.byte_embedding.weight, std=0.02)

    def forward(self, byte_patches: torch.Tensor, prev_mem: torch.Tensor, prev_state: torch.Tensor, hormone: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = byte_patches.shape
        
        x_emb: torch.Tensor = self.byte_embedding(byte_patches)
        mask = (byte_patches != self.pad_token_id).float().unsqueeze(-1)
        x_emb = x_emb * mask

        x_flat = x_emb.view(B, S, -1)
        x_latent = self.compressor(x_flat) # 无语义, 纯特征
        x_latent = F.normalize(x_latent, p=2, dim=-1)
        
        # 使用 StateTransformer 注入语义, 你感知到的是由你决定的
        features, _ = self.context_mixer(x_latent, prev_mem, prev_state, hormone)
        
        # 返回增强后的特征流, Sensor 不改变记忆
        return features
    

class Decompressor(nn.Module):
    """
    解压器: 将 Brain 的 1 个向量扩展为 K 个高维锚点
    """
    def __init__(self, input_dim: int, num_anchors: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_anchors = num_anchors
        
        self.expander = nn.Linear(input_dim, input_dim * num_anchors)
        
        # 初始锚点 
        self.sos_anchor = nn.Parameter(torch.randn(1, 1, num_anchors, input_dim))
        nn.init.normal_(self.sos_anchor, std=0.02)

    def forward(self, thought_stream: torch.Tensor, prev_thought: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = thought_stream.shape
        K = self.num_anchors
        
        current_anchors = self.expander(thought_stream).view(B, S, K, D)
        
        if prev_thought is not None:
            prev_anchors_start = self.expander(prev_thought[:, -1:, :]).view(B, 1, K, D)
        else:
            prev_anchors_start = self.sos_anchor.expand(B, 1, -1, -1)
            
        prev_stream = torch.cat([
            prev_anchors_start,
            current_anchors[:, :-1, :, :]
        ], dim=1)
        
        window_context = torch.cat([prev_stream, current_anchors], dim=2)
        
        context_flat = window_context.view(B * S, 2 * K, D)
        
        return context_flat


class Actor(nn.Module):
    """
    生成字节流
    """
    def __init__(self, config: Config):
        super().__init__()
        self.patch_size = config.patch_size
        self.vocab_size = config.byte_vocab_size
        self.dim = config.actor_config.embed_dim

        self.num_anchors = config.actor_config.num_anchors
        self.decompressor = Decompressor(config.embed_dim, self.num_anchors)

        self.hormone_proj = nn.Linear(config.embed_dim//4, self.dim//4)
        
        self.byte_embedding = nn.Embedding(self.vocab_size, self.dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.patch_size, self.dim))
        
        self.decoder_layer = nn.ModuleList([
            HormoneTransformerLayer(config.actor_config)
            for _ in range(config.actor_config.layers)
        ])

        self.final_norm = HormoneReceptor(nn.LayerNorm(self.dim), config.actor_config)
        self.head = nn.Linear(self.dim, self.vocab_size)
        self.global_sos = nn.Parameter(torch.randn(1, 1, 1, self.dim))

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, thought_stream: torch.Tensor, 
                target_patches: Optional[torch.Tensor] = None, 
                hormone: Optional[torch.Tensor] = None, 
                prev_thought: Optional[torch.Tensor] = None,
                prev_last_tokens: Optional[torch.Tensor] = None,
                force_prefix: Optional[torch.Tensor] = None,
                temperature: float = 1.0,
                top_k: int = 0) -> torch.Tensor:
        B, S, _ = thought_stream.shape
        P = self.patch_size
        device = thought_stream.device

        context_flat = self.decompressor(thought_stream, prev_thought)
        
        # 起始 Embedding (SOS)
        if prev_last_tokens is not None:
            sos_emb = self.byte_embedding(prev_last_tokens).unsqueeze(2)
        else:
            sos_emb = self.global_sos.expand(B, S, 1, -1)

        # 展开激素
        if hormone is not None:
            hormone = self.hormone_proj(hormone) 
            H_dim = self.decoder_layer[0].norm2.hormone_dim
            hormone_flat = hormone.expand(-1, S, -1).reshape(B*S, 1, H_dim)
        else:
            hormone_flat = None

        # Teacher Forcing 训练
        if target_patches is not None:
            clean_targets = target_patches.clone()
            clean_targets[clean_targets == -100] = 0 
            target_emb = self.byte_embedding(clean_targets)

            decoder_input = torch.cat([sos_emb, target_emb[:, :, :-1, :]], dim=2)
            decoder_input = decoder_input + self.pos_embedding
            
            flat_input = decoder_input.view(B * S, P, -1)
            mask = self._generate_square_subsequent_mask(P, device).to(dtype=flat_input.dtype)
            
            x = flat_input
            for layer in self.decoder_layer:
                x = layer(x, hormone_flat, src_mask=mask, context=context_flat)

            out_feat = self.final_norm(x, hormone_flat)
            return self.head(out_feat.view(B, S, P, -1))

        # 推理 
        else:
            # SOS
            current_seq_emb = sos_emb + self.pos_embedding[:,:,0:1,:]
            generated_ids = []
            
            for t in range(P):
                # 当前序列长度 L
                L = current_seq_emb.size(2)
                flat_emb = current_seq_emb.view(B * S, L, -1)
                mask = self._generate_square_subsequent_mask(L, device).to(dtype=flat_emb.dtype)
    
                x = flat_emb
                for layer in self.decoder_layer:
                    x = layer(x, hormone_flat, src_mask=mask, context=context_flat)
                
                # 取最后一个时间步的输出
                last_feat = self.final_norm(x[:, -1:, :], hormone_flat)
                logits_step = self.head(last_feat).squeeze(1) # [B*S, Vocab]
                
                # 前缀强制或采样
                if force_prefix is not None and t < force_prefix.size(1):
                    # 强行选择前缀指定的 Token
                    next_token = force_prefix[:, t].reshape(B * S)
                else:
                    # 采样逻辑
                    if temperature != 1.0:
                        logits_step = logits_step / max(temperature, 1e-6)
                    
                    if top_k > 0:
                        v, _ = torch.topk(logits_step, min(top_k, self.vocab_size))
                        logits_step[logits_step < v[:, [-1]]] = -float('Inf')
                    
                    probs = F.softmax(logits_step, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
                generated_ids.append(next_token.view(B, S, 1))
                
                # 准备下一步的输入
                if t < P - 1:
                    next_emb = self.byte_embedding(next_token).view(B, S, 1, -1)
                    next_input = next_emb + self.pos_embedding[:, :, t+1:t+2, :]
                    current_seq_emb = torch.cat([current_seq_emb, next_input], dim=2)
            
            # 汇总结果并构造伪 Logits 兼容 argmax
            stacked_ids = torch.cat(generated_ids, dim=2) # [B, S, P]
            fake_logits = torch.full((B, S, P, self.vocab_size), -100.0, device=device)
            fake_logits.scatter_(3, stacked_ids.unsqueeze(-1), 100.0)
            
            return fake_logits


class Brain(nn.Module):
    def __init__(self, config: Config, self_encoder: SelfEncoder):
        super().__init__()
        self.core = StateTransformer(config.brain_config, self_encoder)
        
        # 输出概率
        self.prob_proj = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Linear(config.embed_dim // 2, config.embed_dim // 4),
            nn.GELU(),
            nn.Linear(config.embed_dim // 4, 1)
        )

    def forward(self, thoughts_stream: torch.Tensor, prev_mem: torch.Tensor, prev_state: torch.Tensor, hormone: torch.Tensor):
        features, new_state = self.core(thoughts_stream, prev_mem, prev_state, hormone, True)
        prob = torch.mean(self.prob_proj(new_state))
        return features, prob, new_state
    

class Ouro(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.oscillator = NeuralOscillator(config)

        self.cycle_counter: torch.Tensor
        self.register_buffer('cycle_counter', torch.tensor(0, dtype=torch.long))

        self.hypothalamus = Hypothalamus(config)
        self.hippocampus = Hippocampus(config)

        self.self_encoder = SelfEncoder(config)
        
        self.sensor = Sensor(config, self.self_encoder)
        self.brain = Brain(config, self.self_encoder)
        self.actor = Actor(config)
        
        self.loss_fct = nn.CrossEntropyLoss()

    def get_init_states(self):
        def _get(cfg: StateTransformerConfig):
            return (torch.zeros(1, cfg.mem_len, cfg.embed_dim, device=self.config.device),
                    torch.zeros(1, cfg.states_len, cfg.embed_dim, device=self.config.device))
        
        m_inn, s_inn = _get(self.config.brain_config)
        t_inn = None
        
        # 节律器状态 
        osc_state = self.oscillator.get_init_state()
        
        return (m_inn, s_inn, t_inn, osc_state)
    
    def detach_internal_states(self):
        """
        在 BPTT 边界手动截断内部的隐式递归状态
        """
        # 截断全局激素
        self.hypothalamus.current_hormone = self.hypothalamus.current_hormone.detach()

        # 截断全局自我
        self.self_encoder.current_self = self.self_encoder.current_self.detach()
    
    def forward(self, input_patches: Optional[torch.Tensor], target_patches: Optional[torch.Tensor]=None, 
                states: Optional[Tuple[torch.Tensor, ...]]=None, is_sleeping_phase: bool = False,
                override_last_tokens: Optional[torch.Tensor] = None,
                force_prefix: Optional[torch.Tensor] = None,
                temperature: float = 1.0,
                top_k: int = 0
                ) -> tuple[torch.Tensor, Optional[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor], bool]: 
        
        # 状态初始化
        if states is None and input_patches is not None:
             states = self.get_init_states()
        
        brain_mem, brain_state, brain_thought,  osc_state = states
        device = brain_mem.device
        
        # 节律更新
        next_osc_state = self.oscillator(osc_state, self.config.cycle_len)
        self.cycle_counter = self.cycle_counter + 1

        is_sleeping = is_sleeping_phase
        if input_patches is None:
            is_sleeping = True
            
        # 睡眠
        if is_sleeping:
            dream_input = self.hippocampus.inception(brain_mem, brain_state, next_osc_state)
            hormone = self.hypothalamus.get_hormone(dream_input, next_osc_state)
            _, _, brain_state = self.brain(dream_input, brain_mem, brain_state, hormone)

            next_brain_mem, next_brain_state = self.hippocampus.consolidate(brain_mem, brain_state)
            zero_loss = torch.tensor(0.0, device=device, dtype=next_osc_state.dtype)

            return zero_loss, None, (next_brain_mem, next_brain_state, brain_state, next_osc_state), True, zero_loss
        
        # 清醒
        else:
            hormone = self.hypothalamus.get_hormone(brain_state, next_osc_state)
            
            # Sensor
            latent_input = self.sensor(input_patches, brain_mem, brain_state, hormone)
            
            # Brain
            final_thought, prob_loss, next_brain_state = self.brain(latent_input, brain_mem, brain_state, hormone)
            next_brain_mem = brain_mem
            
            # 处理 SOS token
            prev_last_tokens = override_last_tokens if override_last_tokens is not None else input_patches[:, :, -1]

            # Actor
            logits = self.actor(
                final_thought, 
                prev_thought=brain_thought,
                target_patches=target_patches, 
                hormone=hormone, 
                prev_last_tokens=prev_last_tokens,
                force_prefix=force_prefix,
                temperature=temperature,
                top_k=top_k
            )
            
            # Loss
            task_loss = torch.tensor(0.0, device=device)
            logits_loss = torch.tensor(0.0, device=device)
            if target_patches is not None:
                flat_logits = logits.view(-1, self.config.byte_vocab_size)
                flat_targets = target_patches.view(-1)
                logits_loss = self.loss_fct(flat_logits, flat_targets)

                task_loss = (1/SQRT2 * prob_loss - SQRT2 * logits_loss) ** 2 + 0.5 * prob_loss ** 2
                
            return task_loss, logits, (next_brain_mem, next_brain_state, final_thought, next_osc_state), False, logits_loss