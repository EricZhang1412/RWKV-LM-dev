# make_data_streaming_noshuffle.py
# Usage:
#   python make_data_streaming_noshuffle.py "/data/fineweb-edu/**/*.parquet" 1 4096 --out_name fineweb-edu
# Notes:
#   - 保持原始顺序：不进行 shuffle / 近似随机
#   - N_EPOCH=1 即只过一遍数据；>1 时将按相同顺序重复 N_EPOCH 次
#   - 可加 --skip_decode_check 跳过 encode/decode 一致性校验（遇到极端字符时更稳）

import os, sys, json, glob, argparse
import numpy as np
from datasets import load_dataset

# --------------------------------------------------------------------------------
# Tokenizer & binidx（与你提供的逻辑保持一致）
# --------------------------------------------------------------------------------
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")

from src.binidx import MMapIndexedDataset
def index_file_path(prefix_path): return prefix_path + ".idx"
def data_file_path(prefix_path): return prefix_path + ".bin"

class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype=np.uint16):
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]
    def add_item(self, np_array):
        assert np_array.dtype == self._dtype
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)
    def end_document(self):
        self._doc_idx.append(len(self._sizes))
    def finalize(self, index_file):
        self._data_file.close()
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)

cnt = 0
SKIP_DECODE_CHECK = False

def add_raw(raw):
    global builder, cnt, SKIP_DECODE_CHECK
    # 原样写入，不做清洗
    out = tokenizer.encode(raw)
    if not SKIP_DECODE_CHECK:
        try:
            if tokenizer.decode(out) != raw:
                print("Tokenizer roundtrip mismatch. Use --skip_decode_check to ignore.")
                sys.exit(1)
        except Exception as e:
            print(f"Tokenizer decode error: {e}. Use --skip_decode_check to ignore.")
            sys.exit(1)
    # [0] = end_of_doc for rwkv tokenizer
    out.append(0)
    builder.add_item(np.array(out, dtype=np.uint16))
    builder.end_document()
    if cnt % 1000 == 0:
        print(cnt, end=" ", flush=True)
    cnt += 1

def is_prime(n):
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

# --------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------
def expand_parquet_files(path_like: str):
    # 如果是目录，则递归找 parquet；否则按通配符展开
    if os.path.isdir(path_like):
        pattern = os.path.join(path_like, "**", "*.parquet")
        files = glob.glob(pattern, recursive=True)
    else:
        files = glob.glob(path_like, recursive=True)
    files = [f for f in files if f.endswith(".parquet")]
    files.sort()  # 关键：稳定顺序
    return files

def decide_out_name(input_path: str, user_out_name: str | None, files: list[str]):
    if user_out_name: 
        return user_out_name
    # 没指定则尽量从目录名推断
    if os.path.isdir(input_path):
        base = os.path.basename(os.path.normpath(input_path))
        return base or "dataset"
    if any(ch in input_path for ch in ["*", "?", "["]):
        if files:
            parent = os.path.basename(os.path.dirname(files[0]))
            return parent or "dataset"
        return "dataset"
    # 普通文件名（很少见于 parquet 单文件）
    return os.path.splitext(os.path.basename(input_path))[0] or "dataset"

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("IN_PATH", type=str, help='Parquet 路径或通配符（例如 "/data/**/*.parquet" 或目录路径）')
    parser.add_argument("N_EPOCH", type=int, help="建议设为 1 以保持原样；>1 会按相同顺序重复数据")
    parser.add_argument("CTX_LEN", type=int, help="上下文长度，用于计算 magic_prime")
    parser.add_argument("--out_name", type=str, default=None, help="输出前缀名（默认从目录名推断）")
    parser.add_argument("--skip_decode_check", action="store_true", help="跳过 encode/decode 一致性校验")
    parser.add_argument("--max_rows", type=int, default=None, help="仅处理前 N 行（调试用）")
    args = parser.parse_args()

    global builder, SKIP_DECODE_CHECK, cnt
    SKIP_DECODE_CHECK = args.skip_decode_check
    IN_PATH = args.IN_PATH
    N_EPOCH = int(args.N_EPOCH)
    CTX_LEN = int(args.CTX_LEN)

    files = expand_parquet_files(IN_PATH)
    if not files:
        print(f"No parquet files found for: {IN_PATH}")
        sys.exit(1)

    OUT_NAME = decide_out_name(IN_PATH, args.out_name, files)
    print(f"### Convert (no-shuffle) {IN_PATH} -> {OUT_NAME}.bin/idx")
    print(f"### Found {len(files)} parquet files (ordered).")
    if N_EPOCH != 1:
        print(f"### WARNING: N_EPOCH={N_EPOCH}. 数据将按相同顺序重复 {N_EPOCH} 次。若要完全原样只过一遍，请设为 1。")

    builder = MMapIndexedDatasetBuilder(f"{OUT_NAME}.bin")

    total = 0
    for ep in range(N_EPOCH):
        print(f"Epoch {ep+1}/{N_EPOCH} (no shuffle)")
        # 每个 epoch 重新构造 streaming，可避免迭代器复用的潜在问题
        stream = load_dataset("parquet", data_files=files, split="train", streaming=True)
        for row in stream:
            x = row.get("text")
            if not isinstance(x, str):
                # 跳过非字符串或空文本
                continue
            add_raw(x)
            total += 1
            if args.max_rows and total >= args.max_rows:
                print("\n### Reached --max_rows limit; stopping early for debug.")
                break
        if args.max_rows and total >= args.max_rows:
            break

    builder.finalize(f"{OUT_NAME}.idx")
    print("\n### Build done.")

    # ---------------- Verify & magic_prime (与原脚本一致) ----------------
    print("### Verifying result...")
    data = MMapIndexedDataset(OUT_NAME)
    data_len = len(data)
    data_size = len(data._bin_buffer) // data._index._dtype_size

    TODO = [0, data_len - 1] if data_len > 1 else [0]
    PREVIEW_LIMIT = 100
    for idx in TODO:
        ptr, size = data._index[idx]
        dix = data.get(idx=idx, offset=0, length=size).astype(int)
        print("-" * 70 + f"[{OUT_NAME} idx {idx} sz {size}]")
        assert dix[-1] == 0
        dix = dix[:-1]
        if len(dix) > PREVIEW_LIMIT:
            try:
                print(tokenizer.decode(dix[:PREVIEW_LIMIT]))
            except:
                try:
                    print(tokenizer.decode(dix[: PREVIEW_LIMIT + 1]))
                except:
                    print(tokenizer.decode(dix[: PREVIEW_LIMIT + 2]))
            print("· " * 30)
            try:
                print(tokenizer.decode(dix[-PREVIEW_LIMIT:]))
            except:
                try:
                    print(tokenizer.decode(dix[-PREVIEW_LIMIT - 1 :]))
                except:
                    print(tokenizer.decode(dix[-PREVIEW_LIMIT - 2 :]))
        else:
            print(tokenizer.decode(dix))

    print(f"{'-'*80}\n### Final {OUT_NAME}.bin/idx has {data_size} tokens, {data_len} items. Dtype {data._index.dtype}")

    if data_size >= CTX_LEN * 3:
        n_chunk = int(data_size // CTX_LEN) - 1
        for i in range(n_chunk, 0, -1):
            if i % 3 == 2:
                if is_prime(i):
                    print(f"\n### magic_prime = {i} (for ctxlen {CTX_LEN})")
                    print(f'\n--my_exit_tokens {data_size} --magic_prime {i} --ctx_len {CTX_LEN}\n')
                    sys.exit(0)

if __name__ == "__main__":
    main()
