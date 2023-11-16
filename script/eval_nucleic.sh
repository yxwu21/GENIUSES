export CUDA_VISIBLE_DEVICES=$1

python run.py --mode eval \
              --dataset_path "dataset/others/nucleic_acid/nucleic_acid_data_0.95/*.dat" \
              --input_dim 96 \
              --hidden_dim_1 64 \
              --hidden_dim_2 32 \
              --batch_size 64 \
              --load_ckpt_dir output/exp_with_label_transform/translation/train-2023-02-19-23-00-38 \
              --load_ckpt_path output/exp_with_label_transform/translation/train-2023-02-19-23-00-38/model_0.pth \
              --probe_radius_upperbound 1.5 \
              --probe_radius_lowerbound -1.5
