这个仓库对 lerobot 中 pi05 的 pytorch 实现过程进行了详细标注。

建议阅读顺序：
1. `configuration_pi05.py`
2. `processor_pi05.py`
3. `modeling_pi05.py`
   - `PI05Policy(PreTrainedPolicy)`
   - `PI05Pytorch(nn.Module)`
   - `PaliGemmaWithExpertModel()`

关键函数：
1. `_prepare_attention_masks_4d(self, att_2d_masks)`
2. `embed_prefix(self, images, img_masks, tokens, masks)`
3. `embed_suffix(self, noisy_actions, timestep)`
4. `sample_actions(...)`、`denoise_step(...)`
5. 所有 `forward(...)`
