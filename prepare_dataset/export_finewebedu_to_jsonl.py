# save as: export_finewebedu_to_jsonl.py
import os, json, sys
from datasets import load_dataset

parquet_glob = sys.argv[1]              # 例：/data/fineweb-edu/**/*.parquet
out_jsonl = sys.argv[2]                 # 例：fineweb-edu.jsonl
max_rows = int(sys.argv[3]) if len(sys.argv) > 3 else None  # 可选：仅导出前N行做测试

ds = load_dataset(
    "parquet",
    data_files=parquet_glob,
    split="train",
    streaming=True
)

cnt = 0
with open(out_jsonl, "w", encoding="utf-8") as f:
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        cnt += 1
        if max_rows and cnt >= max_rows:
            break

print(f"wrote {cnt} lines to {out_jsonl}")
