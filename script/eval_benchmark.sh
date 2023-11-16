export CUDA_VISIBLE_DEVICES=$1

for grid in 0.35 0.55 0.75 0.95
do
    python run.py --mode eval \
                --dataset_path "../ray/MLSES/datasets/grid_robust_test/bench_$grid/*/*.dat" \
                --input_dim 96 \
                --hidden_dim_1 64 \
                --hidden_dim_2 32 \
                --batch_size 2048 \
                --load_ckpt_dir output/exp_with_label_transform/translation/train-2023-02-19-23-00-38 \
                --load_ckpt_path output/exp_with_label_transform/translation/train-2023-02-19-23-00-38/model_0.pth \
                --probe_radius_upperbound 1.5 \
                --probe_radius_lowerbound -1.5
done
