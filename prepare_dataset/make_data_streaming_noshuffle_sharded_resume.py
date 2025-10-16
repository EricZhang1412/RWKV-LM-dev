# make_data_streaming_noshuffle_sharded_resume.py

# # 首次运行（每 2,000,000 行切一卷）
# python make_data_streaming_noshuffle_sharded_resume.py "/data/fineweb-edu/**/*.parquet" 1 4096 \
#   --out_name fineweb-edu --output_dir /out --rows_per_shard 2000000

# # 中断后继续（从自动保存的 checkpoint 恢复）
# python make_data_streaming_noshuffle_sharded_resume.py "/data/fineweb-edu/**/*.parquet" 1 4096 \
#   --out_name fineweb-edu --output_dir /out --rows_per_shard 2000000 --resume

# # 自定义 checkpoint 文件路径
# python make_data_streaming_noshuffle_sharded_resume.py "/data/fineweb-edu/**/*.parquet" 1 4096 \
#   --out_name fineweb-edu --output_dir /out --rows_per_shard 2000000 \
#   --resume --checkpoint /out/fwe.resume.json

# python make_data_streaming_noshuffle_sharded_resume.py \
#   "/data/malulab/datasets/fineweb-edu/**/*.parquet" \
#   1 4096 \
#   --out_name fineweb-edu \
#   --output_dir /data/malulab/datasets/fineweb-edu/binidx-samples \
#   --rows_per_shard 2000000 \
#   --resume



import os, sys, glob, json, argparse, signal
import numpy as np
from typing import List, Tuple, Optional
from datasets import load_dataset
import re

import bisect
try:
    import pyarrow.parquet as pq
except Exception as _e:
    pq = None
    print("### WARN: pyarrow 未安装，将回退到慢速定位（会非常慢）。建议 pip install pyarrow")

# 新增：快速统计每个 parquet 的行数
def _parquet_num_rows(path: str) -> int:
    if pq is None:
        # 极端回退：用流式迭代（很慢）
        from datasets import load_dataset
        n = 0
        for _ in load_dataset("parquet", data_files=[path], split="train", streaming=True):
            n += 1
        return n
    return pq.ParquetFile(path).metadata.num_rows

def build_cumulative_rows(files: List[str]) -> Tuple[List[int], int]:
    """
    返回 (prefix_sums, total_rows)
    prefix_sums[i] = files[0..i-1] 的累计行数（前缀和，长度 = len(files)+1）
    """
    prefix = [0]
    for f in files:
        prefix.append(prefix[-1] + _parquet_num_rows(f))
    return prefix, prefix[-1]

def seek_stream_to_global_offset(files: List[str], global_skip: int) -> Tuple[int, int, Optional[object]]:
    """
    用 parquet footer 快速定位 (file_idx, row_offset_in_file, iterator)
    - 如果 global_skip == total_rows：说明一切都处理完了，返回 (len(files), 0, None)
    """
    prefix, total = build_cumulative_rows(files)
    if global_skip >= total:
        return len(files), 0, None
    # 在前缀和里找第一个 > global_skip 的位置
    # prefix: [0, n0, n0+n1, ...]
    fi = bisect.bisect_right(prefix, global_skip) - 1
    row_offset = global_skip - prefix[fi]

    # 构造迭代器并丢掉 row_offset 行
    stream = load_dataset("parquet", data_files=[files[fi]], split="train", streaming=True)
    it = iter(stream)
    for _ in range(row_offset):
        next(it, None)
    return fi, row_offset, it

# ---------- Tokenizer & binidx ----------
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")

from src.binidx import MMapIndexedDataset
def index_file_path(prefix_path): return prefix_path + ".idx"
def data_file_path(prefix_path):  return prefix_path + ".bin"

# === 新增工具函数：扫描既有分卷，统计总样本，给出下一分卷号 ===
def scan_existing_shards(base_prefix: str) -> Tuple[int, int]:
    """
    返回 (next_part_idx, total_docs_upto_prev_parts)
    如果没有任何完成分卷，返回 (0, 0)
    """
    out_dir = os.path.dirname(base_prefix)
    out_name = os.path.basename(base_prefix)
    # 形如 fineweb-edu.partNNN.idx
    idx_pat = re.compile(rf"^{re.escape(out_name)}\.part(\d+)\.idx$")
    part_numbers = []
    for fn in os.listdir(out_dir):
        m = idx_pat.match(fn)
        if m:
            part_numbers.append(int(m.group(1)))
    if not part_numbers:
        return 0, 0

    part_numbers.sort()
    total_docs = 0
    for p in part_numbers:
        prefix = f"{base_prefix}.part{p:03d}"
        try:
            data = MMapIndexedDataset(prefix)
            total_docs += len(data)   # 每个分卷的 item 数
        except Exception as e:
            print(f"### WARN: failed to open {prefix}.idx/bin: {e}. Skipping this shard.")
            continue

    next_part_idx = part_numbers[-1] + 1
    return next_part_idx, total_docs

# === 新增工具函数：在输入数据流上跳过 N 条（全局）===
def seek_stream_to_global_offset(files: List[str], global_skip: int) -> Tuple[int, int, Optional[object]]:
    """
    在 files 顺序上跳过 global_skip 条记录，返回 (file_idx, row_offset_in_file, iterator)
    - 如果 global_skip 为 0：返回 (0, 0, None) 表示从头开始
    - iterator 若不为 None，表示已就位在目标文件上、从 offset 开始的可迭代器
    """
    if global_skip <= 0:
        return 0, 0, None

    remaining = global_skip
    for fi, filepath in enumerate(files):
        # 先做一次快速流式计数
        stream = load_dataset("parquet", data_files=[filepath], split="train", streaming=True)
        consumed = 0
        for _ in stream:
            consumed += 1
            if consumed >= remaining:
                # 命中在这个文件里
                # 重新创建 iterator，并丢弃 offset 行
                stream2 = load_dataset("parquet", data_files=[filepath], split="train", streaming=True)
                it = iter(stream2)
                for _ in range(remaining):
                    next(it, None)
                return fi, remaining, it
        # 还不够，跨到下一文件
        remaining -= consumed

    # 如果执行到这里，说明 global_skip 超过了总行数（输入已被完全处理过）
    return len(files), 0, None

# ---------- Resumable builder (append) ----------
class ResumableBuilder:
    """
    写入到 <prefix>.bin.part（可追加），并把每个样本的长度 sizes 以及 doc_idx 增量
    记录到 <prefix>.sizes.tmp / <prefix>.docidx.tmp。finalize 时写 .idx 并把 .bin.part
    原子重命名为 .bin。
    """
    def __init__(self, prefix: str, resume: bool = False, dtype=np.uint16):
        self.prefix = prefix
        self.dtype = dtype
        self.bin_part = self.prefix + ".bin.part"
        self.sizes_tmp = self.prefix + ".sizes.tmp"
        self.docidx_tmp = self.prefix + ".docidx.tmp"

        # 以追加模式打开 .bin.part
        self._data_file = open(self.bin_part, "ab" if resume else "wb")

        # 读取或初始化 sizes / doc_idx（doc_idx 的定义与原版一致：每个 doc 结束时记录“累计 doc 数”的边界）
        if resume and os.path.exists(self.sizes_tmp):
            with open(self.sizes_tmp, "r") as f:
                self._sizes = [int(x) for x in f if x.strip()]
        else:
            self._sizes = []

        if resume and os.path.exists(self.docidx_tmp):
            with open(self.docidx_tmp, "r") as f:
                self._doc_idx = [int(x) for x in f if x.strip()]
            if not self._doc_idx:
                self._doc_idx = [0]
        else:
            self._doc_idx = [0]

    def add_item(self, np_array):
        assert np_array.dtype == self.dtype
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)
        # 追加写 sizes 到磁盘，避免掉电丢进度
        with open(self.sizes_tmp, "a") as f:
            f.write(str(np_array.size) + "\n")

    def end_document(self):
        self._doc_idx.append(len(self._sizes))
        with open(self.docidx_tmp, "a") as f:
            f.write(str(self._doc_idx[-1]) + "\n")

    def finalize(self):
        self._data_file.flush()
        self._data_file.close()
        # 写 .idx
        with MMapIndexedDataset.Index.writer(self.prefix + ".idx", self.dtype) as index:
            index.write(self._sizes, self._doc_idx)
        # 原子重命名 .bin.part -> .bin
        os.replace(self.bin_part, self.prefix + ".bin")
        # 清理 tmp
        for p in [self.sizes_tmp, self.docidx_tmp]:
            if os.path.exists(p):
                os.remove(p)

# ---------- Misc ----------
cnt = 0
SKIP_DECODE_CHECK = False
INTERRUPTED = False

def add_raw(raw: str, builder: ResumableBuilder):
    global cnt, SKIP_DECODE_CHECK
    out = tokenizer.encode(raw)
    if not SKIP_DECODE_CHECK:
        try:
            if tokenizer.decode(out) != raw:
                print("Tokenizer roundtrip mismatch; use --skip_decode_check to ignore."); sys.exit(1)
        except Exception as e:
            print(f"Tokenizer decode error: {e}; use --skip_decode_check to ignore."); sys.exit(1)
    out.append(0)  # [0] = end_of_doc
    builder.add_item(np.array(out, dtype=np.uint16))
    builder.end_document()
    if cnt % 1000 == 0: print(cnt, end=" ", flush=True)
    cnt += 1

def is_prime(n: int) -> bool:
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i+2) == 0: return False
        i += 6
    return True

def expand_parquet_files(path_like: str) -> List[str]:
    if os.path.isdir(path_like):
        files = glob.glob(os.path.join(path_like, "**", "*.parquet"), recursive=True)
    else:
        files = glob.glob(path_like, recursive=True)
    files = [f for f in files if f.endswith(".parquet")]
    files.sort()
    return files

def decide_out_name(input_path: str, user_out_name: Optional[str], files: List[str]) -> str:
    if user_out_name: return user_out_name
    if os.path.isdir(input_path):
        base = os.path.basename(os.path.normpath(input_path)); return base or "dataset"
    if any(ch in input_path for ch in ["*", "?", "["]):
        if files:
            parent = os.path.basename(os.path.dirname(files[0])); return parent or "dataset"
        return "dataset"
    return os.path.splitext(os.path.basename(input_path))[0] or "dataset"

# ---------- Checkpoint ----------
def default_ckpt_path(output_dir: str, out_name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{out_name}.resume.json")

def save_ckpt(path: str, state: dict):
    with open(path, "w") as f: json.dump(state, f, ensure_ascii=False, indent=2)

def load_ckpt(path: str) -> dict:
    with open(path, "r") as f: return json.load(f)

def setup_signal_handlers(on_interrupt):
    def handler(signum, frame):
        print(f"\n### Caught signal {signum}, saving checkpoint...")
        on_interrupt()
        sys.exit(1)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

# ---------- Verify (optional) ----------
def verify_and_report(prefix: str, ctx_len: int, preview: int = 100):
    print(f"### Verifying {prefix}.bin/idx ...")
    data = MMapIndexedDataset(prefix)
    data_len = len(data)
    data_size = len(data._bin_buffer) // data._index._dtype_size
    todo = [0, data_len - 1] if data_len > 1 else [0]
    for idx in todo:
        ptr, size = data._index[idx]
        dix = data.get(idx=idx, offset=0, length=size).astype(int)
        print("-" * 70 + f"[{os.path.basename(prefix)} idx {idx} sz {size}]")
        assert dix[-1] == 0
        dix = dix[:-1]
        head = dix[:preview]
        tail = dix[-preview:] if len(dix) > preview else dix
        try: print(tokenizer.decode(head))
        except: print(tokenizer.decode(dix[:preview+2]))
        if len(dix) > preview:
            print("· " * 30)
            try: print(tokenizer.decode(tail))
            except: print(tokenizer.decode(dix[-(preview+2):]))
    print(f"{'-'*80}\n### Final {os.path.basename(prefix)} has {data_size} tokens, {data_len} items. Dtype {data._index.dtype}")
    if data_size >= ctx_len * 3:
        n_chunk = int(data_size // ctx_len) - 1
        for i in range(n_chunk, 0, -1):
            if i % 3 == 2 and is_prime(i):
                print(f"\n### magic_prime = {i} (for ctxlen {ctx_len})")
                print(f'--my_exit_tokens {data_size} --magic_prime {i} --ctx_len {ctx_len}\n')
                break

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("IN_PATH", type=str)
    parser.add_argument("N_EPOCH", type=int)
    parser.add_argument("CTX_LEN", type=int)
    parser.add_argument("--out_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--rows_per_shard", type=int, required=True)
    parser.add_argument("--skip_decode_check", action="store_true")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--verify_each_shard", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    global SKIP_DECODE_CHECK
    SKIP_DECODE_CHECK = args.skip_decode_check

    files = expand_parquet_files(args.IN_PATH)
    if not files:
        print(f"No parquet files found for: {args.IN_PATH}"); sys.exit(1)

    base_out = decide_out_name(args.IN_PATH, args.out_name, files)
    out_dir = args.output_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = args.checkpoint or default_ckpt_path(out_dir, base_out)

    # 生成基础前缀
    base_prefix = os.path.join(out_dir, base_out)

    # 初始状态
    state = dict(
        in_path=args.IN_PATH,
        files=files,
        n_epoch=args.N_EPOCH,
        ctx_len=args.CTX_LEN,
        rows_per_shard=args.rows_per_shard,
        out_dir=out_dir,
        out_name=base_out,
        skip_decode_check=bool(args.skip_decode_check),

        epoch_idx=0,              # 当前 epoch 下标
        file_idx=0,               # 当前处理到的文件索引
        row_offset_in_file=0,     # 当前文件内已跳过的行数（恢复时会先 skip 这么多行）
        part_idx=0,               # 当前分卷号
        docs_in_part=0,           # 当前分卷内已写入的文档数
        total_docs=0,             # 总计已写文档
    )

    # 恢复
    if args.resume and os.path.exists(ckpt_path):
        loaded = load_ckpt(ckpt_path)
        keys_to_check = ["in_path","n_epoch","ctx_len","rows_per_shard","out_dir","out_name","skip_decode_check"]
        for k in keys_to_check:
            if loaded.get(k) != state.get(k):
                print(f"Checkpoint mismatch on '{k}': {loaded.get(k)} != {state.get(k)}"); sys.exit(1)
        lf, cf = loaded["files"], state["files"]
        if len(lf)!=len(cf) or (lf and cf and (lf[0]!=cf[0] or lf[-1]!=cf[-1])):
            print("Checkpoint files differ from current files list."); sys.exit(1)
        state = loaded
        print(f"### Resuming from checkpoint: {ckpt_path}")
    elif args.resume:
        print(f"### --resume provided but checkpoint not found: {ckpt_path}. Trying to infer from existing shards...")
        base_prefix = os.path.join(out_dir, base_out)
        next_part, done_docs = scan_existing_shards(base_prefix)
        print(f"### Found completed shards up to part{next_part-1:03d} (total_docs={done_docs}).")

        fi, offset, it = seek_stream_to_global_offset(files, done_docs)

        # 所有输入都被处理完了：直接退出（不新开空分卷）
        if fi >= len(files):
            print("### All input rows have already been converted. Nothing left to do. Exiting cleanly.")
            return

        state.update(dict(
            epoch_idx=0,
            file_idx=fi,
            row_offset_in_file=offset,
            part_idx=next_part,
            docs_in_part=0,
            total_docs=done_docs,
        ))
        inferred_iterator = it
    else:
        inferred_iterator = None


    # 信号 / 异常时保存 checkpoint
    def on_interrupt():
        print(f"### Saving checkpoint to {ckpt_path}")
        save_ckpt(ckpt_path, state)
    setup_signal_handlers(on_interrupt)

    # 打开/恢复当前分卷
    def shard_prefix(part_idx: int) -> str:
        return f"{base_prefix}.part{part_idx:03d}"

    builder = ResumableBuilder(shard_prefix(state["part_idx"]),
                               resume=args.resume and state["docs_in_part"]>0)

    print(f"### Start (no-shuffle) -> {base_prefix}.partXXX.bin/idx")
    print(f"### Found {len(files)} parquet files (ordered).")
    if args.N_EPOCH != 1:
        print(f"### WARNING: N_EPOCH={args.N_EPOCH}. 按相同顺序重复；完全原样建议设为 1。")

    try:
        # 主循环
        while state["epoch_idx"] < args.N_EPOCH:
            print(f"\n=== Epoch {state['epoch_idx']+1}/{args.N_EPOCH} ===")
            # 逐文件读取；当前文件可能需要先 skip 已处理行
            for fi in range(state["file_idx"], len(files)):
                filepath = files[fi]
                if 'inferred_iterator' in locals() and inferred_iterator is not None and fi == state["file_idx"]:
                    stream = inferred_iterator
                    inferred_iterator = None
                else:
                    stream = load_dataset("parquet", data_files=[filepath], split="train", streaming=True)

                # 如果需要跳过文件内的若干行（老的 JSON 恢复路径）
                skipped = 0
                if state["row_offset_in_file"] > 0 and stream is not None and not hasattr(stream, '__next__'):
                    for _ in stream:
                        skipped += 1
                        if skipped >= state["row_offset_in_file"]:
                            break
                    stream = load_dataset("parquet", data_files=[filepath], split="train", streaming=True)
                    it = iter(stream)
                    for _ in range(state["row_offset_in_file"]):
                        next(it, None)
                    stream = it


                row_idx_in_file = state["row_offset_in_file"]
                for row in stream:
                    x = row.get("text")
                    if isinstance(x, str) and x:
                        add_raw(x, builder)
                        state["docs_in_part"] += 1
                        state["total_docs"] += 1

                        # 分卷滚动
                        if state["docs_in_part"] >= args.rows_per_shard:
                            # 完成当前分卷
                            prefix = shard_prefix(state["part_idx"])
                            builder.finalize()
                            print(f"\n### Closed shard {state['part_idx']}: {prefix}.bin/idx (docs={state['docs_in_part']})")
                            if args.verify_each_shard:
                                verify_and_report(prefix, args.CTX_LEN)
                            # 开新卷
                            state["part_idx"] += 1
                            state["docs_in_part"] = 0
                            builder = ResumableBuilder(shard_prefix(state["part_idx"]), resume=False)

                    row_idx_in_file += 1
                    # 调试上限
                    if args.max_rows and state["total_docs"] >= args.max_rows:
                        state["file_idx"] = fi
                        state["row_offset_in_file"] = row_idx_in_file
                        on_interrupt()
                        print("### Reached --max_rows. Stopping early.")
                        return

                # 当前文件结束，推进到下一个文件
                state["file_idx"] = fi + 1
                state["row_offset_in_file"] = 0

            # 一个 epoch 完成
            state["epoch_idx"] += 1
            state["file_idx"] = 0
            state["row_offset_in_file"] = 0

        # 收尾：最后一卷 finalize
        last_prefix = shard_prefix(state["part_idx"])
        builder.finalize()
        print(f"\n### Build done for last shard {state['part_idx']}: {last_prefix}.bin/idx (docs_in_shard={state['docs_in_part']}, total_docs={state['total_docs']})")
        if args.verify_each_shard:
            verify_and_report(last_prefix, args.CTX_LEN)

        # 成功完成后，清理 checkpoint
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
            print(f"### Removed checkpoint: {ckpt_path}")

    except Exception as e:
        print(f"\n### Exception: {e}")
        on_interrupt()
        raise

if __name__ == "__main__":
    main()
