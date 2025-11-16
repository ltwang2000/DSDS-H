# analyze_topk_heatmaps.py
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from fairseq import checkpoint_utils, utils

# ================= 手动配置 =================
CKPT  = "checkpoints/my_transformer_19/checkpoint_best.pt"  # 改成你的 ckpt
DATA  = "data-bin/en-de"                                  # data-bin 路径
SPLIT = "test2016"                                        # 测试集 split 名

SAMPLE_IDX    = 0   # 想看的样本 index（0,1,2,...）
LOW_LAYER_ID  = 0   # 作为“低层”的 encoder 层索引（layer_idx < 2 的路径）
HIGH_LAYER_ID = 4   # 作为“高层”的 encoder 层索引（layer_idx >= 2 的路径）

USE_CUDA = torch.cuda.is_available()

# ====== 1. 加载模型和 task ======
print("loading checkpoint...")
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

# ====== 2. 取出一个样本（batch size = 1） ======
task.load_dataset(SPLIT)
dataset = task.dataset(SPLIT)
raw_item = dataset[SAMPLE_IDX]

print(f"[INFO] raw sample idx={SAMPLE_IDX}")
print("  src ids:", raw_item["source"])
print("  tgt ids:", raw_item["target"])

itr = task.get_batch_iterator(
    dataset=dataset,
    max_tokens=4096,
    max_sentences=1,  # 每个 batch 1 条，方便可视化
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
src_tokens         = net_input["src_tokens"]          # [1, Ts]
src_lengths        = net_input["src_lengths"]
prev_output_tokens = net_input["prev_output_tokens"]  # [1, Tt] 目标端 token（用来做 y 轴）
img_features_list  = net_input["img_features_list"]

print(f"[INFO] found batch id={sample['id'][0].item()}")

# ====== 3. “分析版”的 Top-k 选择函数（只用于可视化） ======
def token_aware_select_for_vis(text_feats, visual_feats, visual_mask,
                               topk_dict, default_k, layer_idx):
    """
    复刻 MyTransformerEncoderLayer.token_aware_select，
    但额外返回：
      - topk_idx     [B, K]
      - attn_weight  [B, T, K]：文本 token 对选中 K 个视觉向量的注意力
      - dense_heat   [B, T, N_visual]：把 K 个注意力分布还原到 N_visual 维度的“热图”
    text_feats:   [B, T, D]
    visual_feats: [B, N_visual, D]
    visual_mask:  [B, N_visual]，True = mask
    """
    B, T, D = text_feats.shape
    _, N_visual, _ = visual_feats.shape

    # 1) 动态 K
    k = default_k
    for l, topk in sorted(topk_dict.items()):
        if layer_idx >= l:
            k = topk
    k = min(k, N_visual)

    # 2) 文本-视觉相似度（与你 encoder 中一致）
    sim_matrix = torch.matmul(text_feats, visual_feats.transpose(1, 2)) / (D ** 0.25)
    if visual_mask is not None:
        sim_matrix = sim_matrix.masked_fill(visual_mask.unsqueeze(1), float('-inf'))

    # 3) 文本引导的 top-k（先对 T 维做 max pooling 到 N_visual）
    visual_scores, _ = sim_matrix.max(dim=1)               # [B, N_visual]
    topk_scores, topk_idx = visual_scores.topk(k, dim=-1)  # [B, K]

    # 4) 取出 top-k 视觉向量
    idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, D)         # [B, K, D]
    selected = torch.gather(visual_feats, dim=1, index=idx_exp)  # [B, K, D]

    # 5) 文本-图像注意力（与你底层 forward 中第二步一致）
    attn_logits = torch.matmul(text_feats, selected.transpose(1, 2)) / math.sqrt(D)  # [B, T, K]
    attn_weight = F.softmax(attn_logits, dim=-1)  # [B, T, K]

    # 6) 把注意力填回 N_visual 维
    dense_heat = torch.zeros(B, T, N_visual, device=text_feats.device)
    for b in range(B):
        for k_i in range(k):
            idx = topk_idx[b, k_i]
            dense_heat[b, :, idx] += attn_weight[b, :, k_i]

    return selected, topk_idx, attn_weight, dense_heat


# ====== 4. 在 encoder 上挂 hook，捕获低层 / 高层的 Top-k 热图 ======

# 你的 Top-k 代码在 MyTransformerEncoderLayer 里，
# 所以直接 hook encoder 层本身即可：
low_block  = model.encoder.layers[LOW_LAYER_ID]
high_block = model.encoder.layers[HIGH_LAYER_ID]

vis_store = {
    "low_region_heat": None,    # [T, N_region]
    "high_grid_heat": None,     # [T, N_grid]
    "high_region_heat": None,   # [T, N_region]
    "tgt_tokens": None,         # [T]
}

def low_hook(module, inputs, output):
    """
    对应 layer_idx < 2 的路径：只用 region_feats。
    forward 签名：
      forward(self, x, grid_feats, region_feats,
              encoder_padding_mask, grid_img_mask, region_img_mask,
              batch_len, layer_idx)
    """
    x, grid_feats, region_feats, encoder_padding_mask, \
        grid_img_mask, region_img_mask, batch_len, layer_idx = inputs

    B = x.size(1)
    assert B == 1, "当前脚本只支持 batch_size=1 可视化"

    x_b = x.transpose(0, 1)       # [B, T, D]
    region_mask = region_img_mask # [B, N_region]

    with torch.no_grad():
        _, _, _, dense_heat = token_aware_select_for_vis(
            x_b, region_feats, region_mask,
            module.topk_region_dict,    # 直接用模块里的 topk_region_dict
            default_k=36,
            layer_idx=int(layer_idx),
        )

    vis_store["low_region_heat"] = dense_heat[0].detach().cpu().numpy()  # [T, N_region]


def high_hook(module, inputs, output):
    """
    对应 layer_idx >= 2 的路径：grid + region。
    """
    x, grid_feats, region_feats, encoder_padding_mask, \
        grid_img_mask, region_img_mask, batch_len, layer_idx = inputs

    B = x.size(1)
    assert B == 1, "当前脚本只支持 batch_size=1 可视化"

    x_b = x.transpose(0, 1)  # [B, T, D]

    with torch.no_grad():
        # grid 部分
        _, _, _, dense_grid = token_aware_select_for_vis(
            x_b, grid_feats, grid_img_mask,
            module.topk_grid_dict,
            default_k=196,
            layer_idx=int(layer_idx),
        )
        # region 部分
        _, _, _, dense_region = token_aware_select_for_vis(
            x_b, region_feats, region_img_mask,
            module.topk_region_dict,
            default_k=36,
            layer_idx=int(layer_idx),
        )

    vis_store["high_grid_heat"]   = dense_grid[0].detach().cpu().numpy()   # [T, N_grid]
    vis_store["high_region_heat"] = dense_region[0].detach().cpu().numpy() # [T, N_region]


low_handle  = low_block.register_forward_hook(low_hook)
high_handle = high_block.register_forward_hook(high_hook)

# ====== 5. 跑一遍 encoder，触发 hook 收集 Top-k 热图 ======
with torch.no_grad():
    _encoder_out = model.encoder(
        src_tokens=src_tokens,
        src_lengths=src_lengths,
        img_features_list=img_features_list,
        return_all_hiddens=False,
    )

# hook 用完就移除
low_handle.remove()
high_handle.remove()

# ====== 6. 准备 y 轴的目标 token（用 gold target，即 prev_output_tokens） ======
pad_tgt = tgt_dict.pad()
tgt_tokens_1 = prev_output_tokens[0]
tgt_valid_idx = [i for i, tok in enumerate(tgt_tokens_1) if tok != pad_tgt]
vis_store["tgt_tokens"] = tgt_tokens_1[tgt_valid_idx].cpu().numpy()

def tokens_to_words(token_ids, dictionary):
    return [dictionary.string([int(t)]).strip() for t in token_ids]

tgt_words = tokens_to_words(vis_store["tgt_tokens"], tgt_dict)

# ====== 7. 画热图：target token × patch index ======
os.makedirs("topk_vis", exist_ok=True)

def plot_heat(heat, title, x_label, out_name):
    """
    heat: [T, N]
    """
    if heat is None:
        print(f"[WARN] {title} is None, skip.")
        return
    T, N = heat.shape
    # 只保留有效 tgt 步数
    if len(tgt_words) < T:
        heat = heat[:len(tgt_words), :]

    plt.figure(figsize=(10, 4))
    plt.imshow(heat, aspect="auto", cmap="YlGnBu", origin="upper")
    plt.yticks(range(len(tgt_words)), tgt_words, fontsize=8)
    plt.xlabel(x_label)
    plt.ylabel("Target tokens (decoding step t)")
    plt.title(title)
    plt.colorbar(label="Normalized visual weight")
    plt.tight_layout()
    out_path = os.path.join("topk_vis", out_name)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("saved:", out_path)

plot_heat(
    vis_store["low_region_heat"],
    title=f"Low layer (region only, Top-k), sample {SAMPLE_IDX}",
    x_label="Region index",
    out_name=f"sample{SAMPLE_IDX}_low_region.png",
)

plot_heat(
    vis_store["high_grid_heat"],
    title=f"High layer (grid, Top-k), sample {SAMPLE_IDX}",
    x_label="Grid patch index",
    out_name=f"sample{SAMPLE_IDX}_high_grid.png",
)

plot_heat(
    vis_store["high_region_heat"],
    title=f"High layer (region, Top-k), sample {SAMPLE_IDX}",
    x_label="Region index",
    out_name=f"sample{SAMPLE_IDX}_high_region.png",
)

print("done.")
