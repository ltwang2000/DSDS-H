# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor
from fairseq.modules import TransformerEncoderLayer
import torch
import torch.nn.functional as F
import math


class MyTransformerEncoderLayer(TransformerEncoderLayer):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    # MyTransformerEncoderLayer 的 __init__ 方法
    def __init__(self, args):
        super().__init__(args)
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

        # --- FFN 部分 ---
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        # --- Cross-Modal 部分 ---
        self.grid_proj = nn.Linear(768, self.embed_dim)
        self.region_proj = nn.Linear(2048, self.embed_dim)
        self.visual_feature_layer_norm = LayerNorm(self.embed_dim)  # 用于稳定拼接后的特征

        # b. 分层 Top-K 选择的参数(源)
        self.topk_grid_dict = {0: 64, 2: 32, 4: 16}
        self.topk_region_dict = {0: 12, 3: 6}

        #消融不收缩
        #self.topk_grid_dict = {0: 196}
        #self.topk_region_dict = {0: 36}

        #消融激进
        #self.topk_grid_dict = {0:32, 2:16, 4:8}
        #self.topk_region_dict = {0:8, 3:4}

        #更细粒度：
        #self.topk_grid_dict = {0:64, 1:48, 2:32, 3:24, 4:16, 5:12}
        #self.topk_region_dict = {0:12, 1:10, 2:8, 3:7, 4:6, 5:5}

        #消融反向
        #self.topk_grid_dict = {0:16, 2:32, 4:64}
        #self.topk_region_dict = {0:6, 3:12}

        # c. 视觉内部上下文建模模块
        self.visual_self_attn = self.build_attention_module(self.embed_dim, args)
        self.visual_self_attn_layer_norm = LayerNorm(self.embed_dim)

        # d. 最终图文交叉注意力模块
        self.cross_attn = self.build_attention_module(self.embed_dim, args, is_cross_attention=True)
        self.cross_attn_layer_norm = LayerNorm(self.embed_dim)

        # e. 为您的分层逻辑定义的专属模块
        #    - 底层融合模块
        self.gate_linear_low = nn.Linear(self.embed_dim * 2, self.embed_dim)  # 输入: text + region_context

        #    - 高层融合模块
        self.grid_attention_high = self.build_attention_module(self.embed_dim, args)  # 高层独立的 grid attention
        self.region_attention_high = self.build_attention_module(self.embed_dim, args)  # 高层独立的 region attention
        self.scale_attention = nn.Linear(self.embed_dim, 1)  # 动态尺度加权
        self.gate_linear = self.gate_linear_low
        self.mix_gate = nn.Linear(self.embed_dim * 2, self.embed_dim)
        # 3) forward 里用到的 img_proj（把跨模态注意力输出投到 D）
        self.img_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.visual_dropout_module = FairseqDropout(
            float(getattr(args, "visual_dropout", 0.2)),  # 默认0.2，可通过命令行配置
            module_name=self.__class__.__name__
        )

        #单路消融
        #self.visual_ablation = "region_only"

        #可视化
        # === 可视化开关（默认关闭，对训练无影响） ===
        self.save_vis = False        # 外部脚本置 True 时才会存注意力
        self.vis_cache = {}          # 用来存一次前向的可视化中间结果

    def build_attention_module(self, embed_dim, args, is_cross_attention=False):
        """一个辅助函数，用于清晰地构建各种注意力模块"""
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,  # 统一使用编码器的头数
            dropout=args.attention_dropout,
            encoder_decoder_attention=is_cross_attention,
        )
        # 在您的 MyTransformerEncoderLayer 类中

    # 在您的 MyTransformerEncoderLayer 类中

    def token_aware_select(self, text_feats, visual_feats, visual_mask, topk_dict, default_k, layer_idx):
        D = text_feats.size(-1)
        B, N_visual, _ = visual_feats.shape

        # 动态选择Top-K
        k = default_k
        for l, topk in sorted(topk_dict.items()):
            if layer_idx >= l:
                k = topk
        k = min(k, N_visual)

        # 改进的相似度计算（加入温度系数和阈值过滤）
        sim_matrix = torch.matmul(text_feats, visual_feats.transpose(1, 2)) / (D ** 0.25)  # 温度系数0.25
        if visual_mask is not None:
            sim_matrix.masked_fill_(visual_mask.unsqueeze(1), float('-inf'))

        # 文本引导的Top-K选择
        visual_scores, _ = sim_matrix.max(dim=1)
        topk_scores, topk_idx = visual_scores.topk(k, dim=-1)
        selected = torch.gather(visual_feats, dim=1, index=topk_idx.unsqueeze(-1).expand(-1, -1, D))
        return selected

    def forward(self, x, grid_feats, region_feats, encoder_padding_mask, grid_img_mask, region_img_mask, batch_len,
                layer_idx):
        T, B, D = x.size()
        residual = x

        # === 1. 文本自注意力 ===
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=encoder_padding_mask)
        x = self.dropout_module(x)
        x = residual + x
        x_b = x.transpose(0, 1)  # (B, T, D)

        # === 2. 分层级多模态融合 ===
        if layer_idx < 2:
            # === 底层：仅使用 region_feats，融合文本语义 ===
            # 1. 选取 top-k region 特征
            selected_region = self.token_aware_select(
                x_b, region_feats, region_img_mask, self.topk_region_dict, default_k=36, layer_idx=layer_idx
            )
            # 2. 文本-图像注意力引导 region 融合（跨模态）
            x_b = x.transpose(0, 1)  # (B, T, D)
            attn_weight = torch.matmul(x_b, selected_region.transpose(1, 2)) / (D ** 0.5)  # (B, T, K)
            attn_weight = torch.softmax(attn_weight, dim=-1)
            filtered_region = torch.bmm(attn_weight, selected_region)  # (B, T, D)

            # 3. 门控融合文本与 region 特征
            gate_input = torch.cat([x_b, filtered_region], dim=-1)
            gate = torch.sigmoid(self.gate_linear(gate_input))  # (B, T, D)
            x = gate * x_b + (1 - gate) * filtered_region
            x = x.transpose(0, 1)  # 回到 (T, B, D)

        else:
            # === 高层：融合 grid + region，加入自注意力建模 ===
            selected_grid = self.token_aware_select(
                x_b, grid_feats, grid_img_mask, self.topk_grid_dict, default_k=196, layer_idx=layer_idx
            )
            selected_region = self.token_aware_select(
                x_b, region_feats, region_img_mask, self.topk_region_dict, default_k=36, layer_idx=layer_idx
            )
            # 1. grid / region 自注意力建模（独立）
            selected_grid = self.cross_attn(
                selected_grid.transpose(0, 1), selected_grid.transpose(0, 1), selected_grid.transpose(0, 1)
            )[0].transpose(0, 1)

            selected_region = self.cross_attn(
                selected_region.transpose(0, 1), selected_region.transpose(0, 1), selected_region.transpose(0, 1)
            )[0].transpose(0, 1)

            # 2. 拼接融合后再次视觉自注意力（grid + region）
            visual_feats = torch.cat([selected_grid, selected_region], dim=1)  # (B, K1 + K2, D)
            visual_feats = self.cross_attn(
                visual_feats.transpose(0, 1),
                visual_feats.transpose(0, 1),
                visual_feats.transpose(0, 1),
            )[0].transpose(0, 1)

            # 3. 文本-视觉多模态注意力（这里记录高层跨模态注意力）
            x_b = x.transpose(0, 1)  # (B, T, D)

            if self.save_vis:
                # need_weights=True 得到 [B, T, K] 的注意力矩阵
                img_feats_t, attn = self.cross_attn(
                    x_b.transpose(0, 1),  # (T, B, D)
                    visual_feats.transpose(0, 1),  # (K, B, D)
                    visual_feats.transpose(0, 1),
                    need_weights=True,
                    need_head_weights=False,
                )
                # attn: [B, T, K]
                self.vis_cache["high_cross_attn"] = attn.detach().cpu()
                img_feats = img_feats_t.transpose(0, 1)  # (B, T, D)
            else:
                img_feats = self.cross_attn(
                    x_b.transpose(0, 1),
                    visual_feats.transpose(0, 1),
                    visual_feats.transpose(0, 1),
                    need_weights=False,
                )[0].transpose(0, 1)  # (B, T, D)

            # 4. 门控融合图文特征
            gate_input = torch.cat([x_b, self.img_proj(img_feats)], dim=-1)  # (B, T, 2D)
            gate = torch.sigmoid(self.mix_gate(gate_input))  # (B, T, D)
            x = gate * x_b + (1 - gate) * self.img_proj(img_feats)
            x = x.transpose(0, 1)  # 回到 (T, B, D)

        # === 3. 残差连接与FFN ===
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout_module(x)
        x = self.fc2(x)
        x = residual + x
        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
