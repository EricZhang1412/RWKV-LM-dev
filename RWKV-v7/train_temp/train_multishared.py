#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
# Patched: continuous LR across FineWeb-Edu shards (no trainer/model re-init)
########################################################################################################

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities import rank_zero_info
    import pytorch_lightning as pl

    rank_zero_info("########## work in progress (continuous-shards) ##########")


    parser = ArgumentParser()

    # === New: multi-shard options ===
    parser.add_argument("--data_dir", default="", type=str, help="Directory containing fineweb-edu.part*** shards")
    parser.add_argument("--shard_range", default="", type=str, help="Optional range filter like 0-23 or 5-12")
    parser.add_argument("--skip_missing", default=1, type=int, help="1 to skip missing shards, 0 to error")

    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--wandb", default="", type=str)
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--random_seed", default="-1", type=int)

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str)
    parser.add_argument("--vocab_size", default=0, type=int)

    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--epoch_steps", default=1000, type=int)
    parser.add_argument("--epoch_count", default=500, type=int)
    parser.add_argument("--epoch_begin", default=0, type=int)
    parser.add_argument("--epoch_save", default=5, type=int)

    parser.add_argument("--micro_bsz", default=12, type=int)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)

    parser.add_argument("--lr_init", default=6e-4, type=float)
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=-1, type=int)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)
    parser.add_argument("--adam_eps", default=1e-18, type=float)
    parser.add_argument("--grad_cp", default=0, type=int)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)

    parser.add_argument("--train_stage", default=0, type=int)
    parser.add_argument("--ds_bucket_mb", default=200, type=int)

    parser.add_argument("--head_size", default=64, type=int)
    parser.add_argument("--load_partial", default=0, type=int)
    parser.add_argument("--magic_prime", default=0, type=int)
    parser.add_argument("--my_testing", default='x070', type=str)
    parser.add_argument("--my_exit_tokens", default=0, type=int)

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    ########################################################################################################

    import os, warnings, datetime, sys
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    if "deepspeed" in args.strategy:
        import deepspeed
    from pytorch_lightning import seed_everything
    from pathlib import Path

    # -------------------- per-shard table --------------------
    SHARDS = {
        "fineweb-edu.part000": {"my_exit_tokens": 2120280475, "magic_prime": 517637, "ctx_len": 4096},
        "fineweb-edu.part001": {"my_exit_tokens": 2101401482, "magic_prime": 513017, "ctx_len": 4096},
        "fineweb-edu.part002": {"my_exit_tokens": 2086713516, "magic_prime": 509441, "ctx_len": 4096},
        "fineweb-edu.part003": {"my_exit_tokens": 2060152141, "magic_prime": 502961, "ctx_len": 4096},
        "fineweb-edu.part004": {"my_exit_tokens": 2046983452, "magic_prime": 499739, "ctx_len": 4096},
        "fineweb-edu.part005": {"my_exit_tokens": 2081830848, "magic_prime": 508229, "ctx_len": 4096},
        "fineweb-edu.part006": {"my_exit_tokens": 2120329917, "magic_prime": 517637, "ctx_len": 4096},
        "fineweb-edu.part007": {"my_exit_tokens": 2089167406, "magic_prime": 510047, "ctx_len": 4096},
        "fineweb-edu.part008": {"my_exit_tokens": 2069240063, "magic_prime": 505181, "ctx_len": 4096},
        "fineweb-edu.part009": {"my_exit_tokens": 2057392009, "magic_prime": 502277, "ctx_len": 4096},
        "fineweb-edu.part010": {"my_exit_tokens": 2056599108, "magic_prime": 502079, "ctx_len": 4096},
        "fineweb-edu.part011": {"my_exit_tokens": 2134159284, "magic_prime": 521021, "ctx_len": 4096},
        "fineweb-edu.part012": {"my_exit_tokens": 2119528867, "magic_prime": 517457, "ctx_len": 4096},
        "fineweb-edu.part013": {"my_exit_tokens": 2105081567, "magic_prime": 513923, "ctx_len": 4096},
        "fineweb-edu.part014": {"my_exit_tokens": 2095016072, "magic_prime": 511463, "ctx_len": 4096},
        "fineweb-edu.part015": {"my_exit_tokens": 2087021892, "magic_prime": 509513, "ctx_len": 4096},
        "fineweb-edu.part016": {"my_exit_tokens": 2129394019, "magic_prime": 519863, "ctx_len": 4096},
        "fineweb-edu.part017": {"my_exit_tokens": 2163279950, "magic_prime": 528137, "ctx_len": 4096},
        "fineweb-edu.part018": {"my_exit_tokens": 2144135183, "magic_prime": 523463, "ctx_len": 4096},
        "fineweb-edu.part019": {"my_exit_tokens": 2135235623, "magic_prime": 521267, "ctx_len": 4096},
        "fineweb-edu.part020": {"my_exit_tokens": 2096266391, "magic_prime": 511757, "ctx_len": 4096},
        "fineweb-edu.part021": {"my_exit_tokens": 2106655158, "magic_prime": 514313, "ctx_len": 4096},
        "fineweb-edu.part022": {"my_exit_tokens": 2162995148, "magic_prime": 528053, "ctx_len": 4096},
        "fineweb-edu.part023": {"my_exit_tokens": 2141944157, "magic_prime": 522887, "ctx_len": 4096},
    }

    def shard_index(name: str) -> int:
        try:
            return int(name.split("part")[-1])
        except Exception:
            return 0

    # === NEW: robust resolver for shard files (supports .bin+.idx, single .binidx, or basename) ===
    from glob import glob
    def resolve_shard_path(data_dir: str, shard_name: str):
        base = Path(data_dir) / shard_name
        # 1) exact basename present (some setups create a directory or a symlink with this name)
        if base.exists():
            return str(base)
        # 2) bin+idx pair
        if base.with_suffix('.bin').exists() and base.with_suffix('.idx').exists():
            return str(base)  # MyDataset usually wants the basename w/o extension
        # 3) single .binidx file
        if base.with_suffix('.binidx').exists():
            return str(base.with_suffix('.binidx'))  # many binidx readers accept the single-file path
        # 4) glob fallback (handles extra suffixes like .bin.XXXX, .idx.XXXX)
        candidates = sorted(glob(str(base) + '*'))
        # try to pick a stem that has both .bin and .idx variants
        stems = {}
        for c in candidates:
            p = Path(c)
            s = str(p).split('.bin')[0].split('.idx')[0].split('.binidx')[0]
            stems.setdefault(s, set()).add(p.suffix)
        for s, suf in stems.items():
            if '.bin' in suf and '.idx' in suf:
                return s  # return the common stem
        # otherwise, return first candidate if exists
        if candidates:
            return candidates[0]
        return None

    # -------------------- micro_bsz auto-fix --------------------
    def fix_micro_bsz(mbsz: int, devices: int, nodes: int) -> int:
        real = int(devices) * int(nodes)
        b = max(1, int(mbsz))
        while b >= 1:
            if 40320 % (real * b) == 0:
                return b
            b -= 1
        return 1

    import warnings
    if args.random_seed >= 0:
        print((f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n") * 3)
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = args.grad_clip
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = -1
    args.betas = (args.beta1, args.beta2)

    # auto-fix micro_bsz
    fixed = fix_micro_bsz(args.micro_bsz, args.devices, args.num_nodes)
    if fixed != args.micro_bsz:
        rank_zero_info(f"[auto-fix] micro_bsz {args.micro_bsz} -> {fixed} to satisfy divisibility of 40320")
        args.micro_bsz = fixed
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz

    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE"] = str(args.head_size)

    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)

    # Initialize a default run_name (needed by callbacks like train_callback.on_train_batch_start)
    # Use current ctx_len / model size; vocab_size may update after building the first dataset.
    try:
        args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
    except Exception:
        args.run_name = f"ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"

    # proj dir
    Path(args.proj_dir).mkdir(parents=True, exist_ok=True)

    # precision & cudnn
    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for _ in range(3):
            rank_zero_info("Note: fp32 is slow. Consider bf16 / tf32.")
    if args.precision == "fp16":
        rank_zero_info("Note: fp16 might overflow. Consider bf16 / tf32.")

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        pl_precision = 32
    elif args.precision == "fp16":
        pl_precision = 16
    else:
        pl_precision = "bf16"

    os.environ["RWKV_JIT_ON"] = "0" if "deepspeed_stage_3" in args.strategy else "1"

    # === Build model first; dataset will be created per shard ===
    from src.trainer import train_callback, generate_init_weight
    from src.model import RWKV

    model = RWKV(args)

    # init / load weights
    if len(args.load_model) == 0 or args.train_stage == 1:
        init_weight_name = f"{args.proj_dir}/rwkv-init.pth"
        generate_init_weight(model, init_weight_name)
        args.load_model = init_weight_name

    rank_zero_info(f"########## Loading {args.load_model}... ##########")
    try:
        load_dict = torch.load(args.load_model, map_location="cpu")
        for k in list(load_dict.keys()):
            if k.startswith('_forward_module.'):
                load_dict[k.replace('_forward_module.','')] = load_dict[k]
                del load_dict[k]
    except Exception:
        rank_zero_info(f"Bad checkpoint {args.load_model}")
        if args.train_stage >= 2:
            # fallback to previous as in original script
            list_p = []
            for p in os.listdir(args.proj_dir):
                if p.startswith("rwkv") and p.endswith(".pth"):
                    pp = ((p.split("-"))[1].split("."))[0]
                    if pp != "final":
                        pp = -1 if pp == "init" else int(pp)
                        list_p += [pp]
            list_p.sort()
            max_p = list_p[-1]
            args.load_model = f"{args.proj_dir}/rwkv-init.pth" if max_p == -1 else f"{args.proj_dir}/rwkv-{max_p}.pth"
            rank_zero_info(f"Trying {args.load_model}")
            load_dict = torch.load(args.load_model, map_location="cpu")
    model.load_state_dict(load_dict)

    # Single Trainer (keep optimizer/scheduler state across shards)
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[train_callback(args)],
        precision=pl_precision,
        enable_checkpointing=args.enable_checkpointing,
        gradient_clip_val=args.gradient_clip_val,
        logger=args.logger,
        num_sanity_val_steps=args.num_sanity_val_steps,
        log_every_n_steps=args.log_every_n_steps,
        max_epochs=args.max_epochs,
        enable_progress_bar=True,
    )

    if "deepspeed" in args.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = args.ds_bucket_mb * 1000 * 1000

    # Helper: build dataloader for given args.data_file / magic_prime / ctx_len
    from src.dataset import MyDataset
    def build_loader() -> DataLoader:
        train_data = MyDataset(args)
        # update vocab_size if any
        try:
            args.vocab_size = getattr(train_data, "vocab_size", args.vocab_size)
        except Exception:
            pass
        # refresh run_name after vocab known
        try:
            args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
        except Exception:
            pass
        return DataLoader(
            train_data,
            shuffle=False,
            pin_memory=True,
            batch_size=args.micro_bsz,
            num_workers=1,
            persistent_workers=False,
            drop_last=True,
        )

    # If data_dir is given -> run multi-shard continuous
    ran_any = False
    if args.data_dir:
        # shard filter
        all_names = sorted(SHARDS.keys(), key=lambda n: int(n.split("part")[-1]))
        if args.shard_range:
            try:
                a,b = args.shard_range.split("-")
                lo, hi = int(a), int(b)
                all_names = [n for n in all_names if lo <= int(n.split("part")[-1]) <= hi]
            except Exception:
                rank_zero_info(f"[warn] bad --shard_range {args.shard_range}, ignoring")

        # Log what we actually found once (rank0)
        if trainer.global_rank == 0:
            rank_zero_info("Scanning shards in data_dir ...")
        
        for name in all_names:
            cfg = SHARDS[name]
            resolved = resolve_shard_path(args.data_dir, name)
            if resolved is None:
                if int(args.skip_missing) == 1:
                    rank_zero_info(f"[skip] missing shard: {Path(args.data_dir)/name}")
                    continue
                else:
                    raise FileNotFoundError(f"Missing shard: {Path(args.data_dir)/name}")

            if trainer.global_rank == 0:
                rank_zero_info(f"[use] {name} -> {resolved}")

            # per-shard overrides
            os.environ["RWKV_CTXLEN"] = str(cfg["ctx_len"])  # some codepaths read from env
            args.ctx_len = cfg["ctx_len"]
            args.magic_prime = cfg["magic_prime"]
            args.my_exit_tokens = cfg["my_exit_tokens"]
            args.data_type = "binidx"
            args.data_file = str(resolved)

            # derive per-shard schedule (same formula as original)
            args.epoch_count = args.magic_prime // 40320
            args.epoch_steps = 40320 // args.real_bsz
            assert args.epoch_steps * args.real_bsz == 40320

            samples_per_epoch = args.epoch_steps * args.real_bsz
            tokens_per_epoch = samples_per_epoch * args.ctx_len

            rank_zero_info(
                f"""
############################################################################
# CONTINUOUS SHARD TRAINING
# Data = {args.data_file} (binidx), ProjDir = {args.proj_dir}
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
# Model = L{args.n_layer} D{args.n_embd} ctx{args.ctx_len}
# Adam = lr {args.lr_init} -> {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
###########################################################################
""")

            loader = build_loader()
            if trainer.global_rank == 0 and not ran_any:
                # print param shapes only once
                for n in model.state_dict():
                    shape = model.state_dict()[n].shape
                    s = list(shape) + ["", "", "", ""]
                    print(f"{str(s[0]).ljust(5)} {str(s[1]).ljust(5)} {str(s[2]).ljust(5)} {str(s[3]).ljust(5)} {n}")

            if trainer.global_rank == 0 and not ran_any:
                print(f'### Preparing for training (loaded {args.load_model}). Please wait...')
            # Fit on this shard WITHOUT recreating trainer/model -> LR schedule is continuous
            trainer.fit(model, loader)
            ran_any = True

        if not ran_any:
            raise RuntimeError("No shards were trained (all missing / filtered).")

    else:
        # Fallback: original single-file behavior
        assert args.data_type in ["binidx"]
        # derive schedule
        args.epoch_count = args.magic_prime // 40320
        args.epoch_steps = 40320 // args.real_bsz
        assert args.epoch_steps * args.real_bsz == 40320

        from src.dataset import MyDataset
        train_data = MyDataset(args)
        args.vocab_size = train_data.vocab_size
        data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)
        if trainer.global_rank == 0:
            print(f'### Preparing for training (loaded {args.load_model}). Please wait...')

        trainer.fit(model, data_loader)