#!/bin/bash
#######################################################################################################################
#
# Run demo-training-prepare.sh with the same MODEL_TYPE & N_LAYER & N_EMBD first
# Or, rename your base model to rwkv-init.pth and put it in the output folder
#
# The trainer will load the last rwkv-*.pth in the folder, such that it can continue from a stopped run
# Therefore check the log (### Loading rwkv-xxx.pth... ###), and make sure you don't have extra rwkv-*.pth there
#
#######################################################################################################################
#
MODEL_TYPE="x070" # x070 => rwkv-7.0
#
N_LAYER="24"
N_EMBD="2048"
#
CTX_LEN="4096" # !!! change magic_prime if you change ctx_len !!!
PROJ_DIR="out/L"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE # set output folder
#
# !!! by default train.py will load the last .pth in PROJ_DIR, and continue training from it !!!
# !!! so here we will REMOVE all previous checkpts in PROJ_DIR, so they won't be loaded !!!
# !!! comment these if you don't want this behavior !!!
#
rm "$PROJ_DIR"/rwkv-*0.pth
rm "$PROJ_DIR"/rwkv-final.pth
#
#######################################################################################################################
#
# Note bsz & lr affects model & training performance
# Small data => use smaller bsz & slightly smaller LR
# Large data => use larger bsz & slightly larger LR
# Larger model => use smaller LR
# Finetuning => use very small LR, such as 1e-5
#
M_BSZ="8" # takes ~9G VRAM here => reduce this to save VRAM, increase this for faster speed
LR_INIT="5e-5"
# Initial learning rate. The formula is 0.45 / N_EMBD, rounded appropriately. For example, for an L12-D768 0.1B model, the initial learning rate is 0.45/768=0.0005859375, which is rounded to 6e-4.Initial learning rate. The formula is 0.45 / N_EMBD, rounded appropriately. For example, for an L12-D768 0.1B model, the initial learning rate is 0.45/768=0.0005859375, which is rounded to 6e-4.
LR_FINAL="3e-5"
# Final learning rate. The formula is 0.04 / N_EMBD, rounded appropriately.
GRAD_CP=1 # 1 => slower, save VRAM; 0 => faster, more VRAM
EPOCH_SAVE=1 # save every 10 "miniepochs" (1 miniepoch = 40320 * ctx_len tokens) => decrease if your GPU is weak
#
#######################################################################################################################
#
# magic_prime = the largest 3n+2 prime smaller than datalen/ctxlen-1 (= 1498226207/512-1 = 2926222.06 in this case) = 2926181 in this case
# use https://www.dcode.fr/prime-numbers-search
#
N_NODE=1 # number of nodes
GPU_PER_NODE=8 # number of GPUs per node
#
DS_BUCKET_MB=200 # set to 2 for consumer GPUs, set to 200 for A100 / H100 (affects speed & vram usage)
#
MY_EXIT_TOKENS=12497361914
MAGIC_PRIME=508229

python train.py --load_model "0" --wandb "Test" --proj_dir $PROJ_DIR --my_testing $MODEL_TYPE \
 --ctx_len $CTX_LEN --train_stage 3 --epoch_count 999999 --epoch_begin 0 \
 --data_file "/data/malulab/datasets/fineweb-edu/binidx-samples/fineweb-edu.part005" \
 --my_exit_tokens $MY_EXIT_TOKENS --magic_prime $MAGIC_PRIME \
 --num_nodes $N_NODE --micro_bsz $M_BSZ --n_layer $N_LAYER --n_embd $N_EMBD \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 \
 --data_type "binidx" --vocab_size 65536 \
 --weight_decay 0.001 --epoch_save $EPOCH_SAVE --head_size 64 \
 --accelerator gpu --devices $GPU_PER_NODE --precision bf16 --strategy deepspeed_stage_2 --grad_cp $GRAD_CP --enable_progress_bar True --ds_bucket_mb $DS_BUCKET_MB
