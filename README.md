这个仓库对lerobot中pi05的pytorch实现过程进行了详细标注
建议阅读顺序：
1.configuration_pi05.py
2.processor_pi05.py
3.modeling_pi05.py
  -> class PI05Policy(PreTrainedPolicy)
  -> class PI05Pytorch(nn.Module)
  -> class PaliGemmaWithExpertModel()
  关键的几个函数: 
  (1) def _prepare_attention_masks_4d(self, att_2d_masks)
  (2) def embed_prefix(self, images, img_masks, tokens, masks)
  (3) def embed_suffix(self, noisy_actions, timestep)
  (4) def sample_actions(...)、def denoise_step(...)
  (5) 所有 forward(...)
