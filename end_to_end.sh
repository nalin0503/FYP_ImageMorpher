srun -p rtx3090_slab -w slabgpu05 --gres=gpu:1 \
    --job-name=test --kill-on-bad-exit=1 python3 run_morphing.py \
    --image_path_0 "DiffMorpher/assets/Trump.jpg" \
    --image_path_1 "DiffMorpher/assets/Biden.jpg" \
    --prompt_0 "A photo of an American man" \
    --prompt_1 "A photo of an American man" \
    --output_path "results/Trump_Biden_New" \
    --use_adain \
    --use_reschedule \
    --save_inter \
    --num_frames 16 \
    --duration 100 \
    --use_film \
    --film_output_folder "FILM_Results" \
    --film_fps 40 \
    --film_num_recursions 3
    # --use_lcm

# TODO: handle the other nodes being CuDNN-available