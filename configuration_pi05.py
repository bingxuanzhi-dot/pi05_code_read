from dataclass import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWconfig
from lerobot.optim.schedulers import ConsineDecayWithWarmSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224

@PreTrainedConfig.register_subclass("pi05") #把下面这个 PI05Config 类注册进 Transformers 的配置系统里，注册名叫 "pi05"
@dataclass #自动生成 __init__() 构造函数，根据写的字段（带类型和默认值）自动初始化
class PI05Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32" # Options: "bfloat16", "float32"

    n_obs_steps: int = 1 # 每次决策输入的观测帧数
    chunk_size: int = 50 # action_horizon, 预测的action steps数量
    n_action_steps: int = 50 # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters
    num_inference_steps: int = 10 # 去噪迭代步数

    # 需要对时间步 t 进行采样，t ~ Beta(α,β)，𝛼=1.5, 𝛽=1.0会导致分布更加偏向于1(纯噪声)
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    # 需要把采样出来的 t 映射到(0.001，1.0)，防止出现一开始就采样到 t ≈ 0 的情况，t' = offset + scale * t
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001

    # 最短周期和最长周期，在后面对 timestep 做 embedding 的时候发挥作用
    min_period: float = 4e-3
    max_period: float = 4.0

    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )

    empty_cameras: int = 0

    tokenizer_max_length: int = 200

    ''' 
    给不同类型的数据指定不同的归一化方式：
    图像：不做归一化
    STATE和Action: 分位数归一化。因为不同维度量纲差距很大，同时运行时可能存在异常抖动
    对每一维找分别统计 q_low(比如1%分位数) 和 q_high(比如99%分位数)
    然后把中间的部分映射到稳定区间: x_norm = 2 × (x - q_low) / (q_high - q_low) - 1
    对于那2%的数据, 需要使用clip截断: x_norm = clip(x_norm, -1, 1)
    '''
    normalization_mapping: dict{str, NormalizationMode} = field(
        default_factory = lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    # Training settings
    '''
    gradient_checkpointing: 梯度检查点, 选择性不保存某些中间激活, 反向传播时重新计算一次forward换取显存
    compile: 是否启用 torch.compile 加速, 它会把模型的 forward 过程做图优化和 kernel fusion, max_autotune是其最激进的性能优化模型。
    compile虽然可以带来加速, 但是不适合动态shape(需要反复编译, 速度很慢)
    '''
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max_autotune"
    device: str | None = None

    # Finetuning settings
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    # optimizer settings:
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    ''' 
    @dataclass 会先自动生成 __init__() 把所有字段赋值完，然后立刻自动调用 __post_init__() 让你做额外处理。
    因为 dataclass 的字段赋值是“傻瓜式”的：只负责把参数存进去，不会帮你做校验、推导字段、自动修正等逻辑。
    '''
    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        
        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
    
    '''
    确保输入规范一致
    模型结构要求有固定数量的相机输入，但数据集可能没有那么多相机 → 用“空相机”占位。
    如果输入里没有 STATE，就补一个默认的 STATE feature
    如果输出里没有 ACTION，就补一个默认的 ACTION feature
    '''
    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution).
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features[OBS_STATE] = state_feature
        
        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features[ACTION] = action_feature
    
    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )
    
    def get_scheduler_preset(self):
        return ConsineDecayWithWarmSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )
    
    '''
    @property把方法伪装成属性来使用

    class A:
    def action_dim(self):
        return 50
    a = A()
    a.action_dim() 必须加括号

    class A:
        @property
        def action_dim(self):
            return 50
    a = A()
    a.action_dim 不用括号，看起来像变量
    '''

    # 观测给原始值，不是增强值
    @property
    def observation_delta_indices(self) -> None:
        return None
    
    # 输出的动作不是绝对动作，而是delta形式
    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))
    
    # reward 不用 delta(奖励就是原始标量序列，不做差分）
    @property
    def reward_delta_indices(self) -> None:
        return None