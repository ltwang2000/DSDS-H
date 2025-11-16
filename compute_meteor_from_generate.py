#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re, sys, math
from collections import OrderedDict
from typing import List, Dict, Tuple
from nltk.translate.meteor_score import meteor_score as nltk_meteor_score

# —— 正则：严格匹配以 TAB 分隔的 fairseq 输出 —— #
PAT_T = re.compile(r"^T-(\d+)\t(.*)$")
PAT_D = re.compile(r"^D-(\d+)\t[^\t]*\t(.*)$")
PAT_H = re.compile(r"^H-(\d+)\t[^\t]*\t(.*)$")
PAT_PLAIN = re.compile(r"^(\d+)\t(.*)$")

# —— 与示例一致的规范化分词（去标点，不改大小写） —— #
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")

def to_tokens(text: str) -> List[str]:
    """去除标点（保留字母数字与空白），再基于空白切分；不做 lower。"""
    no_punct = _PUNCT_RE.sub("", text)
    no_punct = _WS_RE.sub(" ", no_punct).strip()
    return [] if not no_punct else no_punct.split(" ")

def tokens_to_str(tokens: List[str]) -> str:
    """把 tokens 拼回字符串，交给 NLTK（它内部会再 .split()）。"""
    return " ".join(tokens)

def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def parse_generate_out(path: str, prefer: str = "last") -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    返回: (refs, hyps)，key 为 sample id（int）
    兼容:
      - T-<id>\t<ref>
      - D-/H-<id>\t...\t<hyp>   (只取每个 id 的一条；prefer=first/last)
      - <id>\t<hyp>             (纯自定义输出，无 T- 行)
    """
    refs, hyps = OrderedDict(), OrderedDict()
    lines = read_lines(path)

    def set_hyp(i: int, txt: str):
        if prefer == "first":
            if i not in hyps:
                hyps[i] = txt
        else:
            hyps[i] = txt  # last 覆盖

    for line in lines:
        m = PAT_T.match(line)
        if m:
            refs[int(m.group(1))] = m.group(2)
            continue
        m = PAT_D.match(line) or PAT_H.match(line)
        if m:
            i = int(m.group(1)); set_hyp(i, m.group(2)); continue
        m = PAT_PLAIN.match(line)
        if m:
            i = int(m.group(1)); set_hyp(i, m.group(2)); continue
    return refs, hyps

def load_refs_files(ref_paths: List[str], ids: List[int], by_index: bool = True) -> Dict[int, List[str]]:
    """
    从外部参考文件构造 {id: [ref1, ref2, ...]}。
    - by_index=True: 把 id 当作参考文件的行号（0-based），常见于 fairseq 的 sample_id。
    - 如果有多参考，按同一行号收集。
    """
    ref_lists = [read_lines(p) for p in ref_paths]
    refs: Dict[int, List[str]] = {}
    for i in ids:
        cand = []
        for lst in ref_lists:
            if by_index and 0 <= i < len(lst):
                cand.append(lst[i])
            elif not by_index and i < len(lst):
                cand.append(lst[i])
        if cand:
            refs[i] = cand
    return refs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_out", help="fairseq 输出文件：含 T-/D-/H- 或 纯 <id>\\t<hyp>")
    ap.add_argument("--refs", nargs="*", default=None,
                    help="参考文件路径（可多份，多参考会一起评测）。如果 gen_out 无 T- 行，必须提供。")
    ap.add_argument("--prefer", choices=["first", "last"], default="last",
                    help="同一 id 多次出现时取哪一个假设（默认 last）")
    ap.add_argument("--id-as-index", action="store_true",
                    help="把样本 id 当作参考文件的行号（0-based）。常见于 fairseq。默认开启。")
    ap.add_argument("--no-id-as-index", dest="id_as_index", action="store_false")
    ap.set_defaults(id_as_index=True)
    ap.add_argument("--dump", default=None, help="把句级结果写到 TSV（id/ref(s)/hyp/meteor）")
    args = ap.parse_args()

    refs_in, hyps = parse_generate_out(args.gen_out, prefer=args.prefer)
    if not hyps:
        print("未解析到假设译文（hyp）。请检查输入文件。", file=sys.stderr); sys.exit(1)

    # 参考优先级：文件内 T- 行 > --refs 外部文件
    ref_multi: Dict[int, List[str]] = {}
    if refs_in:
        ref_multi = {i: [r] for i, r in refs_in.items()}  # 文件内只有一条参考
    elif args.refs:
        ref_multi = load_refs_files(args.refs, list(hyps.keys()), by_index=args.id_as_index)
    else:
        print("没有参考译文（既无 T- 行也未提供 --refs）。METEOR 无法计算。", file=sys.stderr)
        print("METEOR = 0.00")
        sys.exit(0)

    # 对齐 id：只评测同时存在 ref & hyp 的样本
    common_ids = [i for i in hyps.keys() if i in ref_multi]
    if not common_ids:
        print("参考与假设无法按 id 对齐；请检查 --id-as-index 设置或参考文件顺序。", file=sys.stderr)
        sys.exit(1)

    scores, rows = [], []
    for i in common_ids:
        hyp_str = hyps[i]
        ref_strs = ref_multi[i]  # list[str]

        # 1) 去标点 + 分词（与示例一致，不做 lower）
        hyp_tokens = to_tokens(hyp_str)
        ref_token_lists = [to_tokens(r) for r in ref_strs]

        # 2) tokens 拼回字符串（因为你环境里的 NLTK 版本要求 str）
        hyp_for_nltk = tokens_to_str(hyp_tokens)
        refs_for_nltk = [tokens_to_str(toks) for toks in ref_token_lists]

        # 3) 关闭 NLTK 的默认预处理（preprocess=lambda s: s），避免再 lower
        s = nltk_meteor_score(refs_for_nltk, hyp_for_nltk, preprocess=lambda s: s)

        scores.append(s)
        rows.append((i, " ||| ".join(ref_strs), hyp_str, s))

    corpus = sum(scores) / len(scores) if scores else 0.0
    print(f"METEOR = {corpus*100:.2f}")

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fo:
            fo.write("id\treferences\thypothesis\tmeteor\n")
            for (i, refs_str, hyp, s) in rows:
                score_str = "" if s is None or (isinstance(s, float) and math.isnan(s)) else f"{s:.6f}"
                fo.write(f"{i}\t{refs_str}\t{hyp}\t{score_str}\n")
        print(f"[OK] 句级结果写入 {args.dump}")

if __name__ == "__main__":
    main()
