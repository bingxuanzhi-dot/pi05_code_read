import builtins
import logging
import math
import collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Unpack
from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKINg or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma
    from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    GemmaForCausalLM = None
    PaliGemmaForConditionalGeneration = None

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import DEFAULT_IMAGE_SIZE, PI05Config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import(
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)

class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None

def get_safe_dtype(target_dtype, device_type):
    # 为 给定的 device type 选择一个 safe dtype
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype

'''
输入：
    time:
    dimension:
    min_period:
    max_period:
输出：

把每个样本的标量时间 t(比如扩散的 time step)变成一个 长度为 dimension 的正余弦位置编码向量
'''
def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    # dimension 必须是偶数
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    
    # time 必须是一维
    if time.ndim != 1
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
    
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device) # 为了后面生成周期
    period = min_period * (max_period / min_period) ** fraction # 得到了一堆不同的周期（不同频率），覆盖从高频（小周期）到低频（大周期）。

    scaling_factor = 1.0 / period * 2 * math.pi # w = 2pi / T [ dimension // 2 ]
    sin_input = scaling_factor[None, :] * time[:, None] # 这里会广播，[1, dimension // 2] * [B, 1] -> [B, dimension // 2]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1) # 把正余弦信号拼在一起，最后得到 [B, dimension]

# 为 batch_size 中的每个样本采样去噪时间步骤，根据给定的参数 alpha 和 beta
def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))

'''
如果一个序列的 attention mask 是 [1 1 1 1 1 1]:
    这个序列是因果注意力
如果一个序列的 attention mask 是 [0 0 0 1 1 1]:
    这个序列的前三个 token 之间是双向注意力, 后三个是因果注意力。
    即 0, 1, 2 三个 token 之间可以相互看到, 但是看不到3, 4, 5; 3, 4, 5 都可以看到之间的 token, 但是看不到之后的 token
如果一个序列的 attention mask 是 [1 0 1 0 1 0 0 1 0 0]
    这个序列是分块的, 块内是双向注意力, 块间是因果注意力

torch.cumsum 函数计算前缀和：[1 0 1 0] -> [1 1 2 2] 这就意味着 0, 1 token之间是双向注意力, 2,3 之间是因果注意力
att_2d_masks: 仅考虑 attention mask 的注意力矩阵
                      1 1 2 2                                  1 1 1 1
[1 1 2 2]             1 1 2 2            [1 1 2 2]             1 1 1 1
cumsum[:, None, :]    1 1 2 2            cumsum[:, :, None]    2 2 2 2
                      1 1 2 2                                  2 2 2 2

                                            1 1 0 0
cumsum[:, None, :] <= cumsum[:, :, None]    1 1 0 0
                                            1 1 1 1
                                            1 1 1 1 

pad_2d_masks: 观察哪些 token 是 padding, padding 的 token 无法关注别人, 也无法被别人关注
考虑原始的 pad_mask : [1 0 1 1]
                         1 0 1 1                                     1 1 1 1
[1 0 1 1]                1 0 1 1            [1 0 1 1]                0 0 0 0
pad_masks[:, None, :]    1 0 1 1            pad_masks[:, :, None]    1 1 1 1
                         1 0 1 1                                     1 1 1 1
                                                1 0 1 1
pad_masks[:, None, :] * pad_masks[:, :, None]   0 0 0 0
                                                1 0 1 1
                                                1 0 1 1

att_2d_masks & pad_2d_masks: 把两种 masks 取交集，构造最后每个 token 对每个 token 的注意力是否存在


输入：
    pad_masks: [B, N_prefix + N_suffix]
    att_masks: [B, N_prefix + N_suffix]
输出：
    att_2d_masks & pad_2d_masks: [B, N_prefix + N_suffix, N_prefix + N_suffix]
'''
def mask_att_2d_masks(pad_masks, att_masks):
    
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_maskss.ndim != 3:
        raise ValueError(pad_masks.ndim)
    
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]

    return att_2d_masks & pad_2d_masks

'''
padding 张量的一个操作，支持 [B, T, D] 和 [B, D] 的输入
如果输入张量的最后一维 ≥ new_dim, 则直接返回原张量
否则返回用 0 padding到指定长度之后的结果
'''
def pad_vector(vector, new_dim):
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1])) # 这里默认 padding 的数是0，但是这里的参数 0 表示左侧不填充

'''
把输入图片等比例缩放到能目标 (height, width) 大小, 然后用常数值在四周补边到目标大小
输入：
    image: [B, H, W, C] 或 [B, C, H, W]
    height: 目标 Height
    width: 目标 Width
    mode: 插值模式, 可选项有 bilinear, nearest 等等
'''
def resize_with_pad_torch(
    image: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> Tensor:
    if images.shape[-1] <= 4 # channel 一般维度为3, 满足这个 if 表示数据是格式是 [B, H, W, C]
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0) # 如果没有 batch 维度则手动添加
        images = images.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
    else:
        channels_last = False
        if image_dim() == 3:
            images = images.unsqueeze(0)
    
    batch_size, channels, cur_height, cur_width = images.shape # 把 B C H W 具体的值提取出来

'''
整体作用：在同一层 layer_idx 上，把两路输入(inputs_embeds[0]/[1])各自计算 Q/K/V 后，
将两路的 Q/K/V 在序列长度维拼接成一个更长序列，做一次“联合 self-attention”,
然后把 attention 输出按原两路长度切回，分别走各自的 proj + 门控残差 + post-attn norm + MLP + 门控残差，
最终返回两路在该层更新后的 hidden states(outputs_embeds[0]/[1])。
'''
def compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, gemma_expert
):
    models = [paligemma.language_model, gemma_expert.model]  # 两路模型主干：0=paligemma语言模型，1=动作专家Gemma主干
    query_states = []  # 存两路的 Q（后面会拼接）
    key_states = []  # 存两路的 K（后面会拼接）
    value_states = []  # 存两路的 V（后面会拼接）
    gates = []  # 存两路 input_layernorm 返回的 gate，用于 attention 后的第一次门控残差
    for i, hidden_states in enumerate(inputs_embeds):  # 遍历两路输入特征：i=0/1，hidden_states shape 通常是 [B, Li, D]
        layer = models[i].layers[layer_idx]  # 取出第 i 路模型的第 layer_idx 个 Transformer block
        hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # 输入归一化（支持 AdaRMS 条件化），并得到 gate
        gates.append(gate)  # 保存该路 gate（用于第一次门控残差）
        input_shape = hidden_states.shape[:-1]  # 取 [B, Li]（去掉最后的 hidden 维）
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)  # 构造 reshape 目标：(B, Li, num_heads, head_dim)，num_heads 用 -1 自动推断
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # Q: 线性投影 -> reshape -> 转成 [B, num_heads, Li, head_dim]
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # K: 同上（注意：若是 GQA/MQA，这里需确保形状设计兼容）
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # V: 同上（注意：若是 GQA/MQA，这里需确保形状设计兼容）
        query_states.append(query_state)  # 收集该路 Q
        key_states.append(key_state)  # 收集该路 K
        value_states.append(value_state)  # 收集该路 V
    # Concatenate and process attention  # 下面开始做“联合 attention”：把两路在序列长度维拼成一个更长序列
    query_states = torch.cat(query_states, dim=2)  # 在序列长度维 dim=2 拼接两路 Q：形状变为 [B, heads, L0+L1, head_dim]
    key_states = torch.cat(key_states, dim=2)  # 在序列长度维 dim=2 拼接两路 K：形状变为 [B, heads, L0+L1, head_dim]
    value_states = torch.cat(value_states, dim=2)  # 在序列长度维 dim=2 拼接两路 V：形状变为 [B, heads, L0+L1, head_dim]
    dummy_tensor = torch.zeros(  # 构造一个 dummy 张量，用于 rotary_emb 生成对应长度/ dtype 的 cos/sin
        query_states.shape[0],  # B：batch size
        query_states.shape[2],  # L_total：拼接后的总序列长度
        query_states.shape[-1],  # head_dim：每个 head 的维度
        device=query_states.device,  # 与 Q 同设备
        dtype=query_states.dtype,  # 与 Q 同 dtype
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)  # 根据 position_ids 生成 RoPE 的 cos/sin（使用 paligemma 的 rotary_emb）
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(  # 将 RoPE 应用于拼接后的 Q/K（V 不旋转）
        query_states, key_states, cos, sin, unsqueeze_dim=1  # unsqueeze_dim=1 便于在 heads 维 broadcast
    )
    batch_size = query_states.shape[0]  # B：batch size
    scaling = paligemma.language_model.layers[layer_idx].self_attn.scaling  # attention 缩放因子（常见为 1/sqrt(head_dim) 或实现内定义值）
    # Attention computation  # 开始计算注意力
    att_output, _ = modeling_gemma.eager_attention_forward(  # 使用 eager attention 实现计算一次“联合 self-attn”
        paligemma.language_model.layers[layer_idx].self_attn,  # 使用 paligemma 的 self_attn 模块作为 attention 的实现/配置来源
        query_states,  # 拼接后的 Q：[B, heads, L_total, head_dim]
        key_states,  # 拼接后的 K：[B, heads, L_total, head_dim]
        value_states,  # 拼接后的 V：[B, heads, L_total, head_dim]
        attention_mask,  # 注意力 mask（需匹配 L_total，决定因果/遮挡/两路互看规则）
        scaling,  # 缩放因子
    )
    # Get head_dim from the current layer, not from the model  # 这里再次从当前层取 head_dim
    head_dim = paligemma.language_model.layers[layer_idx].self_attn.head_dim  # 当前层的 head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)  # 将注意力输出 reshape 成 [B, L_total, hidden_size]
    # Process layer outputs  # 将联合 attention 的输出切回两路，并分别走各自的 o_proj + 残差 + MLP
    outputs_embeds = []  # 存两路该层的最终输出 hidden states
    start_pos = 0  # 拼接序列中当前路的起始位置（用于切片）
    for i, hidden_states in enumerate(inputs_embeds):  # 再遍历两路：用原始 hidden_states 做残差基底
        layer = models[i].layers[layer_idx]  # 取该路的当前层
        end_pos = start_pos + hidden_states.shape[1]  # 该路序列在拼接输出中的结束位置：start_pos + Li
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:  # 若 attention 输出 dtype 与该路 o_proj 权重 dtype 不一致
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)  # 将 attention 输出 cast 到与 o_proj 权重一致的 dtype，避免 matmul 类型不匹配
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])  # 切出该路的 attention 输出片段并过该路自己的 o_proj，回到 hidden_size 维
        # first residual  # 第一次残差（attention 残差）
        out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # 门控残差：将 attention 分支与残差分支融合（gate 来自 input_layernorm）
        after_first_residual = out_emb.clone()  # 复制一份作为第二次残差的基底（避免后续覆盖/引用问题）
        out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])  # post-attn 归一化（支持条件化），并返回用于第二次残差的 gate
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16  # 若 MLP 权重是 bf16，则将输入也转 bf16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:  # 检查 MLP 的 up_proj 权重 dtype
            out_emb = out_emb.to(dtype=torch.bfloat16)  # 将 MLP 输入 cast 成 bf16，匹配权重 dtype
        out_emb = layer.mlp(out_emb)  # 过该路的 MLP/FFN 子层
        # second residual  # 第二次残差（MLP 残差）
        out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # 门控残差：将 MLP 分支与残差基底融合（gate 来自 post_attention_layernorm）
        outputs_embeds.append(out_emb)  # 保存该路该层输出
        start_pos = end_pos  # 更新 start_pos，为下一路切片做准备（路0结束后 start_pos=L0）
    return outputs_embeds  # 返回两路的该层输出：[out0, out1]，形状分别为 [B, L0, D] 和 [B, L1, D]

    '''
    假设 cur_height = cur_width = 224
    height = 8   height = 16
    ratio = max(224/8, 224/16) = 28
    resize_height = 8
    resize_width = 8
    '''
    ratio = max(cur_width / width, cur_height / height)
    resize_height = int(cur_height / ratio)
    resize_width = int(cur_width / ratio)

    resize_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    ) # 利用插值的方法做下采样，得到的 resize_images 的维度为 [resized_height, resized_width] (8，8), 还没达到指定维度 [height, width]

    # 不同 dtype 的图片有不同的“合理像素范围”，插值后可能会越界或类型不对，所以要处理
    if image.dtype == torch.uint8: # 常见的 0~255 整数图像
        resized_images = torch.round(resize_images).clamp(0, 255).to(torch.uint8) # 小数化整并把范围严格限制在[0, 255]
    elif images.dtype == torch.float32: # float32 图像必须在 [-1,1]
        resize_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")
    
    '''
    把缩放后的图居中放入目标框中，看看上下左右分别需要补多少像素，如果差值是奇数，多出来的 1 像素放到“后边”(bottom / right)
    divmod(x, 2)
    pad_h0 = x // 2
    remainder_h = x % 2
    上述例子我们需要把 (8, 8) 补充到 (8, 16)
    pad_w0 = 4, remainder_w = 0
    pad_w1 = pad_w0 + remainder_w = 4
    即左右各补 4 个像素
    '''
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resize_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # padding 直到 resize_images 达到指定维度 [height, width]
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),
        mode="constant",
        value=constant_value,
    )
    if channels_last:
        padded_images = padded_images.premute(0, 2, 3, 1)

    return padded_images

class GemmaConfig:
    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width # hidden dimension
        self.depth = depth # num_layers
        self.mlp_dim = mlp_dim # mlp 投影到的维度
        self.num_heads = num_heads # 注意力里 Query 头的数量
        self.num_kv_heads = num_kv_heads # KV的头数，这里数量为1，表示所有 query 公用一个 KV, 即 MQA
        self.head_dim = head_dim # 每个 attention head 的维度
    
    def get_gemma_config(variant: str) -> GemmaConfig:
        if variant == "gemma_300m":
            return GemmaConfig(
                width=1024,
                depth=18,
                mlp_dim=4096,
                num_heads=8,
                num_kv_heads=1,
                head_dim=256,
            )
        else variant == "gemma_2b":
            return GemmaConfig(
                width=2048,
                depth=18,
                mlp_dim=16_384,
                num_heads=8,
                num_kv_heads=1,
                head_dim=256,
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")

class PaliGemmaWithExpertModel(
    nn.Module
):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_config_hf = CONFIG_MAPPING["paligemma"]() # 获取默认配置实例，后面都是在修改这个配置对象的属性
        vlm_config_hf._vocab_size = 257152  # 词表规模
        vlm_config_hf.image_token_index = 257152 # 图像 token 在词表里对应的 token id
        vlm_config_hf.text_config.hidden_size = vlm_config.width # 隐藏层维度
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim # mlp 的维度
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads # query head数量
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim # 每个 attention head维度
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth # transformer block深度
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads # KV 头数, 决定 MHA、GQA还是 MQA
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh" # MLP 中的激活函数
        vlm_config_hf.text_config.torch_dtype = "float32" # 权重默认数据类型
        vlm_config_hf.text_config.vocab_size = 257152 # 文本词表大小
        vlm_config_hf.text_config.use_adarms = use_adarms[0] # 文本端是否启用 AdaRMS
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None # adarms_cond_dim 的维度
        vlm_config_hf.vision_config.image_size = image_size # 图像大小
        vlm_config_hf.vision_config.intermediate_size = 4304 # ViT 中的隐藏层维度
        vlm_config_hf.vision_config.projection_dim = 2048 # 视觉特征投影后的维度，也就是文本那边的隐维度
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast" # 投影函数中的 MLP 操作
        vlm_config_hf.vision_config.torch_dtype = "float32" # 视觉部分的数据类型

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim, # 每个 attention head 的维度
            hidden_size=action_expert_config.width, # 模型主干维度
            intermediate_size=action_expert_config.mlp_dim, # MLP 的中间层维度
            num_attention_heads=action_expert_config.num_heads, # Q heads 数
            num_hidden_layers=action_expert_config.depth, # transformer block 层数
            num_key_value_heads=action_expert_config.num_kv_heads, # KV 头数, 决定 MHA、GQA还是 MQA
            vocab_size=257152, # 词表大小
            hidden_activation="gelu_pytorch_tanh", # MLP 激活函数类型
            torch_dtype="float32", # action_expert 部分数据类型
            use_adarms=use_adarms[1], # Action Expert 端是否启用 AdaRMS
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None, # AdaRMS 条件输入的维度
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf) # vlm 实例
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf) # action expert 实例
        self.gemma_expert.model.embed_tokens = None # 把 Gemma 主干的 token embedding 禁用

        self.to_bfloat16_for_selected_params(precision) # 按精度策略把部分参数转成 bf16
        self._set_requires_grad() # 设置哪些参数需要梯度
    
    '''
    把整个模型切到 bf16 或 fp32, 但强制把某些“数值敏感/容易不稳定”的参数保留为 fp32。
    先把整个模型统一转换 dtype, 然后指定“需要保留 fp32”的参数名片段
    '''
    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")
        
        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)
    
    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
            for param in self.paligemma.vision_tower.parameters():
                param.requires_grad = False
        
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False
    
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()
    
    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)
    
    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)
    
    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None: # 只跑 prefix 的过程
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=pisition_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        
        elif inputs_embeds[0] is None: # 只跑 suffix 的过程
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        
        else: # prefix 和 suffix 的过程都跑
            models = [self.paligemma.language.model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                
            def compute_final_norms(inputs_embeds, adarms_cond): # 最后的 RMSNorm
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds
            
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)
            
            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None
        
        return [prefix_output, suffix_output], prefix_past_key_values

class PI05Pytorch(nn.Module):
    def __init__(self, config: PI05Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )
        
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config, # 主模型的配置
            action_expert_config, # 动作专家的配置
            use_adarms=[False, True], # prefix 不启用 adarms, suffix 启用 adarms
            precision=config.dtype, # 模型内部计算精度
            image_size=config.image_resolution[0], # 视觉部分输入图像的尺寸
            freeze_vision_encoder=config.freeze_vision_encoder, # 是否冻结视觉编码器参数
            train_expert_only=config.train_expert_only, # 是否只训练 expert
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        self.gradient_checkpointing_enabled = False

        '''
        如果配置里开启了 compile_model, 则需要把模型里的关键函数用 torch.compile 编译加速
        这里的关键函数指代 self.sample_actions 和 self.forward
        
        '''
        if config.compile_model:
            torch.set_float32_matmul_precision("high") # 更偏向速度
            self.sample_actions = torch.compile(self.sample_actions, mode=complie_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

        '''
        检测你当前环境里的 transformers / siglip 是否处在一个“正确安装/正确替换”的状态。如果不对，就抛出一个统一的错误 ValueError(msg)，告诉用户怎么修。
        '''
        msg = """"An incorrect transformer version is used, please create an issue on https://github.com/huggingface/lerobot/issues"""
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        
        except ImportError:
            raise ValueError(msg) from None
    
    '''
    函数开关, 决定模型内部相关模块是否都打开梯度检查点
    '''
    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True #整个模型
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True # Paligemma的语言部分
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True # Paligemma的视觉部分
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpoint = True # action expert部分
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False # Paligemma的语言部分
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False # Paligemma的视觉部分
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False # action expert部分
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")
    
    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    '''
    同一入口：
    如果开启了 gradient checkpointing 并且当前在训练模式，就用 PyTorch 的 checkpoint() 来跑 func; 否则就正常直接调用 func。
    use_reentrant=False: 不再用“重入式 autograd.Function”
    preserve_rng_state=False: 重跑前向时, 不保证随机性操作的一致性
    '''
    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    '''
    把 2D bool attention mask(形状 [B, N_prefix + N_suffix, N_prefix + N_suffix])变成 Transformers 常用的 4D additive attention mask(形状 [B, 1, N_prefix + N_suffix, N_prefix + N_suffix])
    并把它转换成“允许的位置加 0, 不允许的位置加一个很大的负数”, 从而在 softmax 前把不允许的位置压成 0 概率。
    这里因为最后 softmax 要计算注意力分数，如果不允许注意力的地方加上这个负数，那它经过 softmax 之后的值近乎为 0, 也就是没有注意力。
    为什么是 4D ? 因为会涉及到 head, 这个新添加的维度会自动广播到 head 维度上
    '''
    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    # 生成一个指定形状的高斯白噪声张量
    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
    
    '''
    为每个 batch 中的样本采样一个去噪的时间步 t, 所有最后返回的维度为 [B]
    假设这个时间步满足 beta 分布: t ~ Beta(α,β), 这个分布严格在 (0, 1)
    默认的超参数 α = 1.5, β = 1.0, 采集的数据偏向1, 及完全的噪声
    对采集到的 t 做线性缩放和平移, 防止取到原始的数据取到 0 附近的值
    '''
    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)
    
    '''
    输入：
        images: list([B, C, H, W])
        img_masks: list([B])
        tokens: [B, L]
        masks: [B, L]
    输出：
        embs: [B, N_prefix, H], 存储每个 batch 中所有 prefix token 的特征表示
        pad_masks: [B, N_prefix], 表示每个 token 是否是 padding 的
        att_masks: [B, N_prefix], 后续计算注意力时候会涉及，这里先知道一下维度即可
    '''
    def embed_prefix(
        self, images, img_masks, tokens, masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        embs = []
        pad_masks = []
        att_masks = []

        # 处理图片：
        for img, img_mask in zip(images, img_masks, strict=True): # 逐路相机处理。strict=True 表示两者长度必须一致，否则报错
            def image_embed_function(img): # 计算图像 embedding(SigLIP), [B, C, H, W] -> [B, N_img, H]
                return self.paligemma_with_expert.embed_image(img)
            
            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb) # 把当前这路相机的 token 存储到最后的 embs 中
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs)) # 把当前这路相机 token 是否有效存储下来 [B] -> [B, N_img]
            att_masks += [0] * num_img_embs # prefix 之间可以全注意力(看到后面计算注意力就理解了)，所以 att_masks 填充时候就是 [0] * num_img_embs
        
        # 处理文本：
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)
        
        lang_emb = self._apply_checkpoint(lang_embed_func, tokens) # [B, L] -> [B, N_lang, H]
        embs.append(lang_emb) # 把文本 token 加入到 embs 中，现在里面存储了所有的视觉 token 和 文本 token
        pad_masks.append(masks) # 把文本 masks 加入到 pad_masks，现在里面存储了所有的视觉 masks 和 文本 masks

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        # 合并
        emb = torch.cat(embs, dim=1) # [[B, N_img, H],[B, N_img, H],...[B, N_lang, H]] -> [B, N_prefix, H]
        pad_masks = torch.cat(pad_masks, dim=1) # 同理
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    '''
    输入：
        noisy_actions: [B, chunk_size, D]
        timestep: [B]
    输出：
        embs: [B, chunk_size, H], 带噪动作块的特征表示
        pad_masks: [B, chunk_size], 全 1 tensor, 表示动作全都不是 padding 的
        att_masks: [B, chunk_size], 对于每个样本都一致， [1, 0, 0, 0,...]
        adarms_cond: [B, H], adarms 的条件向量，这里是 timestep 进行 embedding 之后的结果
    '''
    def embed_suffix(self, noisy_actions, timestep):
        embs = []
        pad_masks = []
        att_masks = []

        # 这里相当于把 timestep转换成 embedding 
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features, # action_expert_config.width
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype) # [B] -> [B, H]

        def action_proj_func(noisy_actions):
            return self.action_in_proj(noise_actions)
        
        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions) # [B, chunk_size, D] -> [B, chunk_size, H]

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x) # SiLU(x) = x · sigmoid(x)
            x = self.time_mlp_out(x)
            return F.silu(x)
        
        time_emb = self._apply_checkpoint(time_mlp_func, time_emb) # [B, H] -> [B, H]
        action_time_emb = action_emb # 命名统一问题，但是这里 action_emb 显然没有时间信息
        adarms_cond = time_emb # time embedding 作为 adarms 的 condition，符合 pi05 的实现方式

        embs.append(action_time_emb) # 现在的embs：list[[B, chunk_size, H]]
        bsize, action_time_dim = action_time_emb.shape[:2] # bsize 和 chunk_size
        action_time_masks = torch.one(bsize, action_time_dim, dtype=torch.bool, device=timestep.device) # 每个 action 都有效
        pad_masks.append(action_time_mask)

        # attn_masks:[[1],[0],[0].....] 共计 chunk_size 个元素
        att_masks += [1] + ([0] * (self.config.chunk_size - 1)) # action 之间是因果注意力，这个看到后面注意力那一块就明白了

        embs = torch.cat(embs, dim=1) # 这里其实列表中只有一个元素，这看起来像是之前 pi0 的代码改过来的, list[[B, chunk_size, H]] -> [B, chunk_size, H]
        pad_masks = torch.cat(pad_masks, dim=1) # 同理
        att_masks =  torch.tensor(att_masks, dtype=embs.type, device=embs.device) # [chunk_size]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks)) # [B, chunk_size]

        return embs, pad_masks, att_masks, adarms_cond

    '''
    输入:
    (1) images, 前面一部分是真实存在的图像, 后面一部分是 padding 的
    (2) img_masks, 图像掩码，前面一部分是全 1 的 tensor, 后面一部分是全 0 的 tensor
    (3) tokens: 文本 token id, 这里包括指令和state token 化之后的结果
    (4) masks: 文本掩码
    (5) actions: GT actions
    (6) noise: 带噪动作块
    (7) time: 带噪动作块对应的时间步
    输出：
    预测的去噪向量场 u_t, v_t 之间的 mse_loss
    '''
    def forward(self, images, img_masks, tokens, masks, actions, noise=None, time=None) -> Tensor:
        # 如果没给 noise 就自己从高斯白噪声里采样一个
        if noise is None:
            noise = self.sample_noise(action.shape, actions.device) # [B, chunk_size, D]
        
        # 如果没给 time 就自己采样一个接近 1 的值
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device) # [B]
        
        time_expanded = time[:, None, None] # 把一维 tensor 扩展成三维 [B, 1, 1]，为了后续形状匹配使用

        '''
        把真实动作 actions 和 噪声 noise 做线性插值, 构造训练输入 x_t, 同时构造目标 u_t
        这是基于 flow-matching 的方法
        '''
        x_t = time_expanded * noise + (1 - time_expanded) * actions # [B, chunk_size, D]
        u_t = noise - actions # [B, chunk_size, D]

        '''
        prefix_embs: [B, N_prefix, H]
        prefix_pad_masks: [B, N_prefix]
        prefix_att_masks: [B, N_prefix]
        suffix_embs: [B, chunk_size, H] / [B, N_suffix, H]
        suffix_pad_masks: [B, chunk_size] / [B, N_suffix]
        suffix_att_masks: [B, chunk_size] / [B, N_suffix]
        adarms_cond: [B, H]
        '''
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if(
            self.paligemma_with_expert.paligemma.language_model.layer[0].self_attn.q_proj.weight.dtype == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1) # [B, N_prefix + N_suffix]
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1) # [B, N_prefix + N_suffix]

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks) # [B, N_prefix + N_suffix, N_prefix + N_suffix]

        '''
        给拼接后的整段 token 序列(prefix+suffix)生成“位置编号”，并且让位置编号只对有效 token 递增, padding token 不占位置。 
        这样 Gemma/PaliGemma 在做位置编码(RoPE/position embedding)时, 能正确对齐真实 token 的相对位置。
        '''
        position_ids = torch.cumsum(pad_masks, dim=1) - 1 # [B, N_prefix + N_suffix]

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert_forward(
                attention_mask=att_2d_masks_4d, # attention_mask: [B, 1, N_prefix + N_suffix, N_prefix + N_suffix]
                position_ids=position_ids, # [B, N_prefix + N_suffix]
                past_key_values=None, # 不使用 KV Cache
                input_embeds=[prefix_embs, suffix_embs], # 输入的 token embedding
                use_cache=False, # 模型不返回 Cache
                adarms_cond=[None, adarms_cond] # prefix 部分不使用 adarms_cond, suffix 部分使用 adarms_cond
            )
            return suffix_out
        
        suffix_out =  self.__apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :] # 从输出结果中取出预测的动作块 [B, N_prefix + N_suffix, H]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)
        
        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")
    
    @torch.no_grad() # 装饰器，表示在这个函数期间关闭梯度跟踪，等于在外面包一层 with torch.no_grad():
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None, # 去噪步数，默认十步
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        # 进行推理过程的的前向并计算动作
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        
        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None: # 没有带噪动作块就采样一个
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )
            noise = self.sample_noise(actions_shape, device)
        
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prifix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        '''
        在 Transformers 里, attention 往往有多种实现/后端可选:
        "eager"：最朴素、最直接的 PyTorch 注意力实现(显式做矩阵乘 + softmax)，不走特殊加速路径
        其他常见后端："sdpa"(PyTorch 的 scaled_dot_product_attention)、"flash_attention_2"、"flex_attention" 等(更快/更省显存，但可能有限制)
        '''
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        # 拿到 prefix token 之间的 KV Cache
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps # 你这里可以看到 dt 是 -0.1，所以去噪时时间步是从 1 -> 0

        x_t = noise # 原始带噪动作块, [B, chunk_size, D]

        for step in range(num_steps):
            time = 1.0 + step * dt # 这里时间就直接是从 1.0 开始，而不是从一个 beta 分布里面进行采样
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize) #[B]

            # 具体一步去噪过程的函数(获得向量场)
            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )
            
            if self._rtc_enabled(): # 如果启用 Real-Time Control
                inference_delay = kwargs.get("inference_delay") # 获取推理时延参数
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over") # 获取上一段 chunk 残留(未执行)
                execution_horizon = kwargs.get("execution_horizon") # 获取实际执行窗口大小

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            
            else:
                v_t = denoise_step_partial_call(x_t)
            
            x_t = x_t + dt * v_t # 去噪过程，去除带噪动作块中的一部分噪声

            # 如果RTC对象存在, 并且打开了debug模式, 就记录本步信息
            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)
        
        return x_t # 返回去噪之后的干净动作块，[B, chunk_size, D]
    
    # 具体的去噪一步过程，这个过程需要重复执行 num_steps 次
    def denoise_step(
        self,
        prefix_pad_masks, # prefix token 是否 pad 对应的 masks, [B, N_prefix]
        past_key_values, # prefix token的 KV cache
        x_t, # 带噪动作块 [B, chunk_size, D]
        timestep, # 当前去噪时间步 [B]
    ):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1] # chunk_size
        batch_size = prefix_pad_masks.shape[0] # batch_size
        prefix_len = prefix_pad_masks.shape[1] # N_prefix

        '''
        这里我们想要得到完整的 full_att_2d_masks([B, N_suffix, N_prefix + N_suffix])
        为什么不是([B, N_prefix + N_suffix, N_prefix + N_suffix])??
        因为这里使用了 KV Cache, 因为在所有的去噪步中 prefix 都是不变的, suffix是改变的, 使用 KV Cache 更快。
        为什么训练不用？
        因为训练就是单步预测向量场, 然后构造 mse_loss 训练, 这里一次推理都是要十步。
        所以你这里可以理解成 Q 是 suffix token, KV 来自 prefix token 和 suffix token
        现在已经有的：
        (1) prefix_pad_masks: [B, N_prefix]
        (2) suffix_pad_masks: [B, N_suffix]
        (3) suffix_att_masks: [B, N_suffix]
        怎么做? 可以依次执行以下三步：
        (1) 先得到 suffix token 对 prefix token的注意力, 因为 prefix 对 suffix 来说都是可见的, 所以决定注意力的就是这个地方是否被 padding
        (2) 得到 suffix 对 suffix 的注意力, 这需要综合考虑注意力方式和 padding 的情况
        (3) 最后把两个注意力叠在一起：[B, N_suffix, N_prefix] + [B, N_suffix, N_suffix] = [B, N_suffix, N_prefix + N_suffix]
        '''
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len) # 广播得到 suffix 对 prefix 的注意力是否有效
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks) # 这个去翻上面吧，一个原理
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2) # [B, N_suffix, N_prefix + N_suffix]

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None] #  统计 prefix 中的有效 token 数量，作为 suffix 的起始偏移量
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1 # 生成 suffix 段的 position_ids，并且让它们的编号从 “prefix 有效 token 数” 之后接着开始。

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks) # [B, 1， N_suffix, N_prefix + N_suffix]
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        '''
        KV cache 中存储了 prefix token 在每一层的 K_prefix 和 V_prefix
        K_all = [K_prefix; K_suffix]    V_all = [V_prefix; V_suffix], 这其实就相当于得到了完整的注意力矩阵
        '''
        output_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            input_embeds=[None, suffix],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = output_embeds[1] # 从 output_embeds 中提取 suffix_out 的部分 (prefix, suffix)
        suffix_out = suffix_out[:, -self.config.chunk_size :] # [B, chunk_size, H]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out) # [B, chunk_size, D], 注意这里是向量场

class PI05Policy(PreTrainedPolicy):

    config_class = PI05Config
    name = "pi05"

    def __init__(
        self,
        config: PI05Config,
        **kwargs,
    ):
        super().__init__(config) # 把 PI05Config 这个配置对象传给父类 PreTrainedPolicy 的构造函数。
        config.validate_features() # 确保输入规范一致
        self.config = config

        # Initialize the core PI05 model
        self.init_rtc_processor()
        self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        self.model.to(config.device)

        # 把 PI05Policy 里面“跟推理过程有关的内部缓存状态”清空，让它从一个“干净的初始状态”开始工作，避免把上一条轨迹的内部缓存带到下一条轨迹里。
        self.reset()
    
    @classmethod #类方法
    # 从一个“预训练模型路径/名字”加载权重 + 配置，然后构造并返回一个 PI05Policy 实例
    def from_pretrained(
        cls: builtins.type[T], # 表示这是一个类，类的类型为T
        pretrained_name_or_path: str | Path, # 加载模型的来源
        config: PreTrainedConfig | None = None, # config
        force_download: bool = False, # 是否下载相关参数
        resume_download: bool | None = None, # 是否断点续传
        proxies: dict | None = None, # 走代理下载
        token: str | None = None, # HF相关token
        cache_dir: str | Path | None = None, # 下载内容指定缓存目录
        local_files_only: bool = False, # 是否只从本地加载
        revision: str | None = None, # 指定模型版本
        strict: bool = True, # 权重key是否需要完全匹配
        **kwargs,
    ) -> T:

        print(
            "The PI05 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure ofr compatibility. \n"
            "Original implementation: https:// github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")
        
        # 如果没有提供 config, 则创建一个默认 config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        
        # 初始化模型，但是这时候并不加载权重
        # 检测 dataset_stats 是否在 kwargs 中提供
        model = cls(config, **kwargs) # 相当于 model = PI05Policy(config, **kwargs), 在执行构造函数__init__(self, config, **kwargs)

        # 接下来手动读取 checkpoint 的参数字典（state_dict）
        try:
            # 从 pytorch_model.bin 或者 model.safetensors文件中导入数据
            try:

                print(f"Loading model from: {pretrained_name_or_path}")

                from transformers.utils import cached_file
                
                # 在 pretrained_name_or_path 这个模型来源里，找到名为 model.safetensors 的文件，并把它下载/定位到本地，返回它的实际路径
                resolved_file = cache_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    use_auth_token=kwargs.get("use_auth_token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )

                from safetensors.torch import load_file
                
                # 把 model.safetensors 读出来，得到参数字典
                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model
            '''
            先把 checkpoint 的参数名字"翻译/修复"为当前版本能认的格式
            避免直接加载时报 missing_keys / unexpected_keys 或结构不匹配崩掉。
            输出的是一个新的 state_dict 字典。
            '''
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            '''
            把修复后的 fixed_state_dict 的 key 再统一加上 model. 前缀
            得到 remapped_state_dict, 然后用 model.load_state_dict() 真正把权重加载进当前 policy, 并打印缺失/多余参数情况。
            '''
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                    if remap_count <= 10:
                        print(f"Remapped: {key} -> {new_key}")
                
                else:
                    remapped_state_dict[key] = value
            
            if(remap_count > 0):
                print(f"Remapped {remap_count} state dict keys")
            
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")
            
            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        exception Exception as e:
            print(f"Warning: Could not remap state dict keys: {e}")
        
        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):
        # 将载入的 stact dict 匹配现有模型
        
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            '''
            这里是判断是否启用了 AdaRMSNorm
            如果启用了，就直接抛弃 RMSNorm 的 weight
            pi0.5采用 AdaRMSNorm, t 在这里作为生成 scale 的条件向量; pi0采用的是 RMSNorm
            下面两个分别匹配模型内部和输出后端的 norm
            '''
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue
            
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue
            
            if "patch_embedding" in key:
                logging.warning(f"Vision embedding key might need handling: {key}")
            
            fixed_state_dict[new_key] = value
        
        return fixed_state_dict
    
    def get_optim_params(self) -> dict:
        return self.parameters()
    
    def reset(self):
        '''
        直接创建一个空队列来起到清空的作用
        _action_queue:专门给动作用的快捷缓存队列(直观、简单)
        _queues: 更通用的“多队列容器”, 为以后扩展或兼容框架接口准备
        '''
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    '''
    在 PI0.5 这种一次预测一段动作 chunk 的策略里（比如一次输出未来 chunk_size 步动作），现实机器人执行时经常会遇到一个问题：
    真实系统里动作不是“立刻生效”的
    原因可能是：
        相机/传感器有延迟
        模型推理需要时间
        你可能还在执行上一次 chunk 的剩余动作
    这会导致：
        chunk 之间衔接不平滑
        延迟导致的抖动、重复执行、或者未来动作被覆盖
    '''
    def init_rtc_processor(self):
        # 如果 config 中 RTC 能启用，在这里就初始化一个 RTC processor
        self.rtc_processor = None

        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None) # 如果这时候还没有self.model，则返回None值；否则返回取到的值
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor
    
    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled
    
    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        '''
        把 batch 里的多路相机图像统一处理成模型能吃的格式，并给每一路相机生成一个“是否存在”的 mask
        Lerobot 中的 Images 的形状为[B, C, H, W], 并且归一化到[0, 1]
        PaliGemma 要求 Images 的形状为[B, C, H, W], 并且归一化到[-1, 1]
        最终输出的 images 和 img_masks 应该是这样的：
        images:
        [视角1特征图, 视角2特征图,...缺失视角1特征图(全-1)...]
        img_masks
        [全 1 tensor, 全 1 tensor,...全 0 tensor...]
        '''
        
        images = []
        img_masks = []

        device = next(self.parameters()).device # 取出第一个参数的 device 并赋值

        '''
        self.config.image_features = [
            "image_front",
            "image_wrist",
            "image_left",
        ]
        batch = {
            "image_front": Tensor(...),
            "state": Tensor(...),
            ...
        }
        present_img_keys 表示在模型期望的图像字段里，挑出 batch 里实际存在的那些
        missing_img_keys 表示在模型期望的图像字段里，挑出 batch 里缺失的那些
        实际训练按照 config 中的字段为准, batch 中如果缺少了需要补
        '''
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )
        
        # 处理 batch 中有的image features
        for key in present_img_keys:
            img = batch[key]

            if img.device != device:
                img = img.to(device)
            
            if img.dtype != torch.float32:
                img = img.to(torch.float32)
            
            is_channels_first = img.shape[1] == 3

            '''
            这里可能会疑惑为什么一开始是[B, C, H, W], 后来要变成[B, H, W, C], 最后又变成[B, C, H, W]呢？
            因为 resize_with_pad_torch这个操作需要维度为 [B, H, W, C]
            '''
            if is_channels_first:
                # 将数据从[B, C, H, W] 变成[B, H, W, C]
                img = img.permute(0, 2, 3, 1)
            
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # 把数据从 [0, 1] 转变为 [-1, 1]
            img = img * 2.0 - 1.0

            if is_channel_first:
                # 还原数据格式 [B, H, W, C] -> [B, C, H, W]
                img = img.permute(0, 3, 1, 2)
            
            images.append(img) # images是个列表，每个元素代表这个batch一个视角的数据
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device) # 存在图像的 mask 都是 1, 即这个视角是有效的
            img_masks.append(mask)

        # 缺几个相机就补几个
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1 # 造一个"全 -1"的假图像
            mask = torch.zeros_like(mask) # mask 全 0：表示这路相机无效
            images.append(img)
            img_masks.append(mask)
        
        return images, img_masks

    def prepare_action(self, batch):
        # 这里就对 action 进行了一个 padding 的操作
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        # 从给定的环境观测中选择单一动作
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        if len(self._action_queue) == 0: # 这个是执行队列
            actions = self.predict_action_chunk(batch)[:, :self.config.n_action_steps] # n_action_steps 是实际的执行的步数
            self._action_queue.extend(actions.transpose(0, 1)) # 把第一个元素拆开放入队列中，即队列中每个元素都是 [B, D] 形状
        
        return self._action_queue.popleft() # 弹出最早动作, [B, A]
    
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        # 从给定环境观测中预测 action chunk
        self.eval()

        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)

        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        '''
        前向过程, 把 batch 数据输入给模型, 然后计算训练 loss
        输入数据是一个字典, key 是 str 类型的, value 是 Tensor类型的
        reduction 这个参数表示如何把一堆值聚合成一个标量, 这里选择直接求平均的方式
        输出数据是一个元组, 第一个元素是 tensor 类型的变量, 第二个元素是 dict 类型的变量
        '''

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        action = self.prepare_action(batch)

        '''
        输入images, img_masks, tokens, masks, actions, 然后可以 action 每个维度的 loss [B, chunk_size, D]
        由于动作是 padding 到32维的维度不适合做计算, 在这里需要做截断
        '''
        losses = self.model.forward(images, img_masks, tokens, masks, actions)
        original_action_dim = self.config.output_feature[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {
            "loss_per_dim": loss.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        '''
        如果 reduction 的模式设置为 none, 则返回对于每个样本返回一个loss, 即最后是 [B]
        如果 reduction 的模式设置为其他, 则对所有样本的所有 action chunk 的每一个维度返回一个标量值, 即最后是一个数
        '''
        if reduction == "none":
            per_sample_loss = losses.mean(dim(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            loss = loss.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

