srun -p rtx3090_slab -w slabgpu08 --gres=gpu:1 \
    --job-name=test --kill-on-bad-exit=1 python3 FILM.py