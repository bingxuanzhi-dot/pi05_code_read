from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import pad_vector
from lerobot.processor import (
    AddBatchDimensionProcessStep,
    DeviceProcessStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_SATET,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):

    max_state_dim: int = 32
    task_key: str = "task"

    '''
    如果一个类实现了：
    def __call__(self, x):
    ...

    那么这个类的对象可以像函数一样被调用：
    obj = SomeClass()
    obj(x)   # 这行等价于 obj.__call__(x)
    '''
    def __call__(self.transition: EnvTransition) -> EnvTransition:
        '''
        输入参数 transition: 一次环境交互的数据(通常包含 observation / action / reward / 额外信息)
        返回值也是 EnvTransition: 处理后的 transition。
        把机器人状态(连续向量)变成 “0~255 的离散符号序列”，和 task 文本一起构造成 给 PI05(π0.5)这种 VLA 模型用的 prompt。
        '''

        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE) # 从transition中取出state
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key) # 从transition中取出文本指令
        if tasks is None:
            raise ValueError("No task found in complementary data")
    
        state = deepcopy(state)

        state = pad_vactor(state, self.max_state_dim) # 原始state的维度可能不是 max_state_dim，需要统一到这个维度

        # 这时候的state已经被分位数归一化到[-1,1]了
        state_np = state.cpu().numpy()
        
        ''' 
        把连续值映射成整数token, 在[-1,1]之间生成257个点, 去掉最后一个点, 保留256个bins起点,
        np.digitize(state_np, bins=...)对state_np中的每个元素, 返回它的bin编号(这时候是1,...256), 然后把它变成(0,...255)
        最后state变成了一个32维整数token, 比如State: 12 88 130 250 ...(共32个)
        '''
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256+1)[:-1]) - 1

        full_prompts = [] # 存储每个样本的 prompt 字符串
        for i, task in enumerate(tasks): # 遍历 batch 中的每条任务文本
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ") # 对文本进行规范化处理，去掉首尾空格，把下划线("_")和标识"\n"替换成空格

            '''
            取出第 i 个样本的离散 state token 序列: discretized_states[i]
            map(str, ...)：把每个整数 token 转成字符串
            " ".join(...)：用空格拼起来形成一个长字符串，比如：[12, 8, 255] → "12 8 255"
            最终得到 state_str: 数值 token 序列的文本版本
            '''
            state_str = " ".join(map(str, discretized_states[i]))

            '''
            构造最终 prompt 字符串，格式是
            Task: <任务文本>, State: <离散state序列>;
            Action:
            '''
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction"
            full_prompts.append(full_prompt)

        '''
        把任务文本升级成“任务+状态+动作占位符”的完整 prompt。
        原本可能是：["pick up ...", "open ..."]
        现在变成：["Task: ... State: ... Action:", ...]
        '''
        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts

        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        #这里不修改特征定义
        return features

def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    '''
    为模型构建pre-processor和post-processor
    pre-processor为模型准备输入数据:
    1. 重命名 features 来匹配预训练的配置；
    2. 添加 batch 维度
    3. 利用数据集的统计数据来归一化输入和输出特征；
    4. 在任务描述添加换行符(action位置)
    5. 把 text prompt 利用 PaliGemma tokenizer进行tokenizing
    6. 将所有数据移动到确切的设备上

    post-processor处理模型的输出:
    1. 把输出从归一化空间还原回原始尺度(反归一化），这里就需要保存的统计数据帮助
    2. 把数据移动到CPU上

    Args:
        config: policy config
        dataset_stats: 存放归一化信息的统计数据的字典
        preprocessor_kwargs: pre-processor pipeline的额外参数
        postprocessor_kwargs: post-processor pipeline的额外参数

    Returns:
        一个元组, 这个元组包含 pre-procrssor 和 post-processor pipeline
    '''

    # input_steps是一个列表，包含于预处理的每一步，每一步都是一个ProcessorStep
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessStep(rename_map={}), #对应1，这里是空映射，实际已经对齐
        AddBatchDimensionProcessorStep(), #对应2
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ), #对应3
        Pi05PrepareStateTokenizerProcessorStep(max_action_dim=config.max_state_dim), #对应4
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ), #对应5，把 prompt 变成 input_ids
        DeviceProcessStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, state=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict(str, Any), dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition, #把PolicyAction转换成transition(dict)结构，方便需要这种数据格式的处理
            to_output=transition_to_policy_action, #把transition(dict)转换成PolicyAction结构，方便需要这种数据格式的处理
        ),
    )

