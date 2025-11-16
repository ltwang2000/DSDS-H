# draw_high_layer_cross_modal.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from fairseq import checkpoint_utils, utils

# ========= 手动配置 =========
CKPT      = "checkpoints/my_transformer_19/checkpoint_best.pt"  # 模型路径
DATA      = "data-bin/en-de"
SPLIT     = "test2016"
LAYER_ID  = 4          # 选一个“高层”的 encoder 层，比如 3 / 4 / 5
SAMPLE_IDX = 115         # 数据集中第几个样本
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========= 1. 加载模型和数据 =========
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [CKPT],
    arg_overrides={"data": DATA},
)
model = models[0].to(DEVICE)
model.eval()

task.load_dataset(SPLIT)
dataset = task.dataset(SPLIT)

# 用 batch iterator 找到包含 SAMPLE_IDX 的那一 batch
itr = task.get_batch_iterator(
    dataset=dataset,
    max_tokens=4096,
    max_sentences=16,
    max_positions=utils.resolve_max_positions(
        task.max_positions(), model.max_positions()
    ),
    ignore_invalid_inputs=True,
    seed=1,
    num_workers=1,
).next_epoch_itr(shuffle=False)

batch_samples = None
for samples in itr:
    ids = samples["id"]
    if (ids == SAMPLE_IDX).any():
        batch_samples = samples
        break

if batch_samples is None:
    raise RuntimeError(f"Cannot find sample index {SAMPLE_IDX} in {SPLIT}")

batch_samples = utils.move_to_cuda(batch_samples) if DEVICE == "cuda" else batch_samples
net_input = batch_samples["net_input"]

src_tokens        = net_input["src_tokens"]          # [B, T_src]
src_lengths       = net_input["src_lengths"]
img_features_list = net_input["img_features_list"]   # 视觉特征

# ========= 2. 打开高层 encoder 的可视化开关 =========
enc_layer = model.encoder.layers[LAYER_ID]
enc_layer.save_vis = True
enc_layer.vis_cache = {}  # 清空旧的缓存

# ========= 3. 正常跑 encoder，一次前向即可把注意力写入 vis_cache =========
with torch.no_grad():
    _ = model.encoder(
        src_tokens=src_tokens,
        src_lengths=src_lengths,
        img_features_list=img_features_list,
        return_all_hiddens=False,
    )

if "high_cross_attn" not in enc_layer.vis_cache:
    raise RuntimeError(
        "high_cross_attn not found in vis_cache. "
        "请确认 LAYER_ID 为高层（layer_idx>=2），且 forward 中已按说明记录注意力。"
    )

attn_all = enc_layer.vis_cache["high_cross_attn"]   # [B, T, K]

# ========= 4. 取出该 batch 中 SAMPLE_IDX 对应的那条样本 =========
ids = batch_samples["id"]
b_idx = (ids == SAMPLE_IDX).nonzero(as_tuple=False)[0].item()

attn = attn_all[b_idx]          # [T, K]
src_tokens_b = src_tokens[b_idx]

src_dict = task.source_dictionary
src_str = src_dict.string(src_tokens_b).split()
T = len(src_str)
attn = attn[:T]                 # [T, K]

print("[INFO] src tokens:", " ".join(src_str))
print("[INFO] attn shape:", attn.shape)

# 行归一化，方便画图
attn_np = attn.cpu().numpy()
attn_np = attn_np / (attn_np.max() + 1e-8)

T, K = attn_np.shape

# ========= 5. 画跨模态对齐热图 =========
fig, ax = plt.subplots(figsize=(8, 4))

im = ax.imshow(attn_np, aspect="auto", cmap="Blues")

ax.set_xlabel("Visual index (Top-k grid + region)")
ax.set_ylabel("Text tokens (encoding step t)")

ax.set_yticks(np.arange(T))
ax.set_yticklabels(src_str, fontsize=7)

ax.set_xticks(np.arange(K))
ax.set_xticklabels([str(i) for i in range(K)], fontsize=5, rotation=45, ha="right")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Cross-modal attention weight", fontsize=8)

ax.set_title(f"High-layer cross-modal alignment (layer {LAYER_ID}, sample {SAMPLE_IDX})", fontsize=10)

fig.tight_layout()
out_fig = f"high_layer_cross_modal_L{LAYER_ID}_sample{SAMPLE_IDX}.png"
fig.savefig(out_fig, dpi=300)
plt.close(fig)

print("saved:", out_fig)
