#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python compute_magic_prime.py /root/fineweb-edu-binidx --ctx_len 4096
# # 只看 fineweb-edu.part*.bin 这类前缀
# python compute_magic_prime.py /root/fineweb-edu-binidx --ctx_len 4096 --pattern "fineweb-edu.part*.bin"
# python compute_magic_prime.py /root/fineweb-edu-binidx --ctx_len 4096 --summary_only


import os, sys, argparse, glob
from typing import Dict, List, Tuple
from src.binidx import MMapIndexedDataset

def is_prime(n: int) -> bool:
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

def find_prefixes(dir_path: str, pattern: str | None) -> List[str]:
    """
    在目录下找所有「前缀」，即同时拥有 .bin 与 .idx 的文件。
    可用 --pattern 进一步过滤（例如 'fineweb-edu.part*.bin'）
    """
    if pattern:
        # 支持类似 'fineweb-edu.part*.bin' 的过滤
        bins = glob.glob(os.path.join(dir_path, pattern))
        bins = [b for b in bins if b.endswith(".bin")]
    else:
        bins = glob.glob(os.path.join(dir_path, "*.bin"))
    prefixes = []
    for b in sorted(bins):
        pfx = b[:-4]  # 去掉 .bin
        idx = pfx + ".idx"
        if os.path.exists(idx):
            prefixes.append(pfx)
    return prefixes

def compute_magic_prime(prefix: str, ctx_len: int) -> Tuple[int,int,int,int | None]:
    """
    返回 (data_size, data_len, dtype_size, magic_prime)
    magic_prime 不存在则返回 None
    """
    data = MMapIndexedDataset(prefix)
    data_len = len(data)  # 文档数（items）
    dtype_size = data._index._dtype_size
    data_size = len(data._bin_buffer) // dtype_size  # token 数
    n_chunk = int(data_size // ctx_len) - 1
    magic = None
    for i in range(n_chunk, 0, -1):
        if i % 3 == 2 and is_prime(i):
            magic = i
            break
    return data_size, data_len, dtype_size, magic

def main():
    ap = argparse.ArgumentParser(description="Scan a folder of *.bin/*.idx and compute magic_prime for each prefix.")
    ap.add_argument("DIR", type=str, help="包含 .bin/.idx 的目录")
    ap.add_argument("--ctx_len", type=int, default=4096, help="上下文长度（默认 4096）")
    ap.add_argument("--pattern", type=str, default=None,
                    help="可选的文件过滤（glob），例如 'fineweb-edu.part*.bin' 只处理这些前缀")
    ap.add_argument("--summary_only", action="store_true",
                    help="只打印汇总行（不逐个前缀详细输出）")
    args = ap.parse_args()

    dir_path = os.path.abspath(args.DIR)
    assert os.path.isdir(dir_path), f"Not a directory: {dir_path}"

    prefixes = find_prefixes(dir_path, args.pattern)
    if not prefixes:
        print(f"No *.bin/*.idx pairs found in: {dir_path}")
        sys.exit(1)

    total_tokens = 0
    total_items = 0

    rows = []
    for pfx in prefixes:
        try:
            data_size, data_len, dtype_size, magic = compute_magic_prime(pfx, args.ctx_len)
        except Exception as e:
            print(f"[ERROR] Failed on {pfx}: {e}")
            continue

        total_tokens += data_size
        total_items += data_len

        if not args.summary_only:
            print("-" * 80)
            print(f"[{os.path.basename(pfx)}]  ctx_len={args.ctx_len}")
            print(f"tokens = {data_size:,}   items = {data_len:,}   dtype_size = {dtype_size}")
            if magic is not None:
                print(f"magic_prime = {magic}")
                print(f"--my_exit_tokens {data_size} --magic_prime {magic} --ctx_len {args.ctx_len}")
            else:
                print("magic_prime = (not found, dataset too small for this ctx_len)")

        rows.append((pfx, data_size, data_len, magic))

    # 汇总
    print("=" * 80)
    print(f"Scanned {len(rows)} shard(s) in {dir_path}")
    print(f"TOTAL tokens = {total_tokens:,}   TOTAL items = {total_items:,}")
    # 也给一个“整合训练时”的参考行（注意：magic_prime 必须针对**单一前缀的 binidx**计算；
    # 如果你把多个前缀在训练时交替采样，就各自用各自的 magic_prime，没有“全局 magic_prime”。）
    print("\n# Per-shard recommended args:")
    for pfx, tok, it, magic in rows:
        name = os.path.basename(pfx)
        if magic is None:
            print(f"# {name}: too small for ctx_len={args.ctx_len}")
        else:
            print(f"# {name}: --my_exit_tokens {tok} --magic_prime {magic} --ctx_len {args.ctx_len}")

if __name__ == "__main__":
    main()
