export CUDA_VISIBLE_DEVICES=$1

python run.py --mode train \
              --git_info `git describe --always`\
              --dataset_path 'dataset/benchmark_data_0.5_neg_part/*/*/*.dat' \
              --dataset_split_path dataset/benchmark_data_0.5_neg_part_split_indices \
              --input_dim 96 \
              --batch_size 64 \
              --lr 1e-4 \
              --epoch 20 \
              --number_worker 8 \
              --log_freq 1000 \
              --eval_freq 10 \
              --save_ratio 0 \
              --ckpt_dir output/exp_with_label_transform/translation \
              --probe_radius_upperbound -1.5 \
              --probe_radius_lowerbound -5
