import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from fairseq import checkpoint_utils, utils

# ================= 手动配置 =================
CKPT  = "checkpoints/my_transformer_19/checkpoint_best.pt"
DATA  = "data-bin/en-de"
SPLIT = "test2016"

SAMPLE_IDX   = 10      # 第几条样本
LAYER_ID     = 4      # 用哪个 encoder 层做可视化（建议用高层）
TARGET_STEP  = 12      # 想看的解码步 t（从 0 开始）
GRID_SIDE    = 14     # grid patch 列数行数，ViT 通常 14×14 = 196

IMG_ROOT     = "flickr30k/test_2016_flickr"  # 图片根目录
USE_CUDA     = torch.cuda.is_available()

# =======================================================
# 1. 加载模型和任务
# =======================================================
print("==== loading checkpoint ====")
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [CKPT],
    arg_overrides={"data": DATA},
)
model = models[0]
model.eval()
if USE_CUDA:
    model.cuda()

src_dict = task.source_dictionary
tgt_dict = task.target_dictionary

# =======================================================
# 2. 取出一个 batch_size=1 的样本
# =======================================================
task.load_dataset(SPLIT)
dataset = task.dataset(SPLIT)
raw_item = dataset[SAMPLE_IDX]

print(f"[INFO] raw sample idx={SAMPLE_IDX}")
print("  src ids:", raw_item["source"])
print("  tgt ids:", raw_item["target"])

itr = task.get_batch_iterator(
    dataset=dataset,
    max_tokens=4096,
    max_sentences=1,
    max_positions=utils.resolve_max_positions(
        task.max_positions(),
        model.max_positions(),
    ),
    ignore_invalid_inputs=True,
    seed=1,
    num_workers=1,
).next_epoch_itr(shuffle=False)

sample = None
for batch in itr:
    if batch["id"][0].item() == SAMPLE_IDX:
        sample = batch
        break

if sample is None:
    raise RuntimeError(f"Cannot find sample with id={SAMPLE_IDX}")

if USE_CUDA:
    sample = utils.move_to_cuda(sample)

net_input = sample["net_input"]
src_tokens         = net_input["src_tokens"]
src_lengths        = net_input["src_lengths"]
prev_output_tokens = net_input["prev_output_tokens"]
img_features_list  = net_input["img_features_list"]

pad_tgt = tgt_dict.pad()
tgt_ids = prev_output_tokens[0]
tgt_valid_idx = [i for i, tok in enumerate(tgt_ids) if tok != pad_tgt]
tgt_ids_valid = tgt_ids[tgt_valid_idx]

def tokens_to_words(token_ids, dictionary):
    return [dictionary.string([int(t)]).strip() for t in token_ids]

tgt_words = tokens_to_words(tgt_ids_valid, tgt_dict)
print("[INFO] target tokens:", " ".join(tgt_words))

if TARGET_STEP >= len(tgt_words):
    raise ValueError(f"TARGET_STEP={TARGET_STEP} >= valid target length={len(tgt_words)}")

print(f"[INFO] visualize step t={TARGET_STEP}, token='{tgt_words[TARGET_STEP]}'")

# =======================================================
# 3. 可视化用的辅助函数：Top-k + dense 注意力
# =======================================================
def token_aware_select_for_vis(text_feats, visual_feats, visual_mask,
                               topk_dict, default_k, layer_idx):
    """
    与 MyTransformerEncoderLayer.token_aware_select 逻辑保持一致，
    但额外返回：
      - dense_full : 不用 Top-k、对所有 patch 的 softmax 注意力 [B,T,N]
      - dense_topk : 只对选择的 K 个 patch 归一化，再还原到 N 维 [B,T,N]
    """
    B, T, D = text_feats.shape
    _, N_visual, _ = visual_feats.shape

    # 动态 K
    k = default_k
    for l, topk in sorted(topk_dict.items()):
        if layer_idx >= l:
            k = topk
    k = min(k, N_visual)

    # 文本-视觉相似度
    sim_matrix = torch.matmul(text_feats, visual_feats.transpose(1, 2)) / (D ** 0.25)  # [B,T,N]
    if visual_mask is not None:
        sim_matrix = sim_matrix.masked_fill(visual_mask.unsqueeze(1), float('-inf'))

    # dense: 不用 Top-k，直接对 N_visual 做 softmax
    dense_full = F.softmax(sim_matrix, dim=-1)  # [B,T,N]

    # Top-k：先在 N 维上选 patch
    visual_scores, _ = sim_matrix.max(dim=1)              # [B,N]
    topk_scores, topk_idx = visual_scores.topk(k, dim=-1) # [B,K]
    idx_exp = topk_idx.unsqueeze(-1)                      # [B,K,1]
    selected = torch.gather(visual_feats, dim=1,
                            index=idx_exp.expand(-1, -1, D))   # [B,K,D]

    # 文本-Topk 注意力
    attn_logits = torch.matmul(text_feats, selected.transpose(1, 2)) / math.sqrt(D)  # [B,T,K]
    attn_topk = F.softmax(attn_logits, dim=-1)  # [B,T,K]

    dense_topk = torch.zeros_like(dense_full)   # [B,T,N]
    for b in range(B):
        for k_i in range(k):
            idx = topk_idx[b, k_i]
            dense_topk[b, :, idx] += attn_topk[b, :, k_i]

    return dense_full, dense_topk, topk_idx


# =======================================================
# 4. 在 encoder 第 LAYER_ID 层挂 hook，拿到 grid 的 heatmap
# =======================================================
encoder_layer = model.encoder.layers[LAYER_ID]

vis_store = {
    "grid_full": None,   # [T, Ng]
    "grid_topk": None,   # [T, Ng]
}

def encoder_hook(module, inputs, output):
    """
    forward(self, x, grid_feats, region_feats,
            encoder_padding_mask, grid_img_mask, region_img_mask,
            batch_len, layer_idx)
    """
    x, grid_feats, region_feats, encoder_padding_mask, \
        grid_img_mask, region_img_mask, batch_len, layer_idx = inputs

    B = x.size(1)
    assert B == 1, "当前脚本只支持 batch_size=1 可视化"

    x_b = x.transpose(0, 1)  # [B,T,D]

    with torch.no_grad():
        dense_full, dense_topk, _ = token_aware_select_for_vis(
            x_b,
            grid_feats,
            grid_img_mask,
            module.topk_grid_dict,
            default_k=GRID_SIDE * GRID_SIDE,
            layer_idx=int(layer_idx),
        )

    # 只取 batch 0
    vis_store["grid_full"] = dense_full[0].detach().cpu().numpy()   # [T,Ng]
    vis_store["grid_topk"] = dense_topk[0].detach().cpu().numpy()   # [T,Ng]

handle = encoder_layer.register_forward_hook(encoder_hook)

# 跑一遍 encoder，触发 hook
with torch.no_grad():
    _ = model.encoder(
        src_tokens=src_tokens,
        src_lengths=src_lengths,
        img_features_list=img_features_list,
        return_all_hiddens=False,
    )

handle.remove()

grid_full = vis_store["grid_full"]   # [T,Ng]
grid_topk = vis_store["grid_topk"]   # [T,Ng]

if grid_full is None or grid_topk is None:
    raise RuntimeError("hook 没有拿到 grid heatmap，请检查 LAYER_ID 是否正确。")

T, Ng = grid_full.shape
assert GRID_SIDE * GRID_SIDE == Ng, f"Ng={Ng} 与 GRID_SIDE={GRID_SIDE} 不匹配"

# 选定 target step t
if TARGET_STEP >= T:
    raise ValueError(f"TGT_STEP={TARGET_STEP} >= T={T}")

heat_full_t  = grid_full[TARGET_STEP]   # [Ng]
heat_topk_t  = grid_topk[TARGET_STEP]   # [Ng]

# 归一化
heat_full_t /= (heat_full_t.max() + 1e-8)
heat_topk_t /= (heat_topk_t.max() + 1e-8)

# reshape 成 2D grid
heat_full_2d = heat_full_t.reshape(GRID_SIDE, GRID_SIDE)
heat_topk_2d = heat_topk_t.reshape(GRID_SIDE, GRID_SIDE)

# =======================================================
# 5. 读取对应的原图，并插值热图到同尺寸
# =======================================================
# 这里需要你根据自己的 dataset 结构拿到 img_id。
# 下面假设 raw_item 里有 'img_id' 字段，如果是别的名字，改一下即可：
# ========= 原来的部分 =========
# img_id = int(raw_item.get("img_id", raw_item["id"]))  # 尝试 img_id，否则用 id
# img_path = os.path.join(IMG_ROOT, f"{img_id}.jpg")

# ========= 改成下面这样 =========

IMG_LIST_FILE = "flickr30k/test_2016_flickr.txt"  # <-- 换成你真实的列表文件路径

# 读入所有测试图片文件名（每行一个）
with open(IMG_LIST_FILE, "r") as f:
    img_names = [line.strip() for line in f if line.strip()]

# 用 SAMPLE_IDX 在列表里查名字
if SAMPLE_IDX >= len(img_names):
    raise ValueError(f"SAMPLE_IDX={SAMPLE_IDX} >= num_images={len(img_names)}")

img_name = img_names[SAMPLE_IDX]
# 列表里如果已经带 .jpg 就直接拼；没带的话补上
if not (img_name.endswith(".jpg") or img_name.endswith(".jpeg") or img_name.endswith(".png")):
    img_name = img_name + ".jpg"

img_path = os.path.join(IMG_ROOT, img_name)

print("[INFO] image file from list:", img_name)
print("[INFO] image path:", img_path)


if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")

img = Image.open(img_path).convert("RGB")
W, H = img.size

def upsample_heat(h2d):
    h_img = Image.fromarray((h2d * 255).astype(np.uint8)).resize(
        (W, H), resample=Image.BILINEAR
    )
    return np.array(h_img).astype(np.float32) / 255.0

heat_full_up = upsample_heat(heat_full_2d)
heat_topk_up = upsample_heat(heat_topk_2d)

out_name = f"encoder_sample{SAMPLE_IDX}_t{TARGET_STEP}_overlay.png"


# 整行分成 3 列：左图、右图、colorbar
fig = plt.figure(figsize=(9, 4))  # 可以再调大一点宽度
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.05)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
cax = fig.add_subplot(gs[0, 2])   # 专门放 colorbar 的窄轴

# 左：原始 dense 注意力
ax1.imshow(img)
ax1.imshow(heat_full_up, cmap="jet", alpha=0.5)
ax1.axis("off")
ax1.set_title(
    f"Original (no Top-k), t={TARGET_STEP}\n token='{tgt_words[TARGET_STEP]}'",
    fontsize=8,
)

# 右：Top-k 注意力
ax2.imshow(img)
im2 = ax2.imshow(heat_topk_up, cmap="jet", alpha=0.5)
ax2.axis("off")
ax2.set_title(
    f"Enhanced (Top-k), t={TARGET_STEP}\n token='{tgt_words[TARGET_STEP]}'",
    fontsize=8,
)

# colorbar 放在第三列的 cax 上，不会挤压图片
cbar = fig.colorbar(im2, cax=cax)
cbar.set_label("Normalized visual weight", fontsize=8)

# 不必 tight_layout，用 GridSpec 已经控制好布局
fig.savefig(out_name, dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved:", out_name)
