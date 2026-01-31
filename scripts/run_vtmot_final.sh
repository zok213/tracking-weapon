#!/bin/bash
# Real AI Engineering: Robust Detachment Script
# Prevention of SIGHUP (Signal 1) when IDE/SSH closes.

# 1. Kill old processes
pkill -f train_vtmot_rgbt_ddp.py
pkill -f torchrun

# 2. Launch with setsid (New Session) + nohup
# setsid forces the process into a new session, detaching it from current TTY
echo "Launching DDP Training (Batch 24) in NEW SESSION..."

setsid nohup torchrun \
    --nproc_per_node=2 \
    --master_port=29508 \
    scripts/train_vtmot_rgbt_ddp.py \
    > logs/rgbt_vtmot_ddp_final.log 2>&1 < /dev/null &

# 3. Capture PID
PID=$!
echo "Process Launched with PID: $PID"

# 4. Disown explicitly (Bash specific)
disown -h $PID

echo "✅ Success! Process $PID is fully detached."
echo "You can verify with: ps -o pid,pgid,sid,cmd -p $PID"
echo "Monitor with: tail -f logs/rgbt_vtmot_ddp_final.log"
