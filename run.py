from inspect import trace
import os
import time
import glob

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import random_split
from torch.utils.data import DataLoader
from src.trainer import Trainer
from src.args import get_common_args
from src.data import RefinedMlsesMapDataset, LabelTransformer, RefinedMlsesMemoryMapDataset, Subset, ExpLabelTransformer, TranslationLabelTransformer
from src.model import MLSESModel, RefinedMLSESModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    parser = get_common_args()
    args = parser.parse_args()

    # reproducible
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # post-processing arguments
    current_time = time.strftime(
        "%Y-%m-%d-%H-%M-%S",
        time.localtime(time.time())
    )
    experiment_name = f'{args.mode}-{current_time}'
    experiment_path = os.path.join(args.ckpt_dir, experiment_name)
    args.ckpt_dir = experiment_path
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    label_transformer = TranslationLabelTransformer(args.probe_radius_upperbound, args.probe_radius_lowerbound)
    args.sign_threshold = label_transformer.sign_threshold
    model = MLSESModel(args.input_dim, args.hidden_dim_1, args.hidden_dim_2)
    trainer = Trainer(args, model=model)
    if args.mode == 'process_data':
        dat_files = glob.glob(args.dataset_path)

        # read files and build offset index
        dataset = RefinedMlsesMapDataset(
            dat_files, input_dim=args.input_dim
        )

        # split dataset into training, eval, and test
        train_indices, test_indices = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=args.seed)
        train_indices, eval_indices = train_test_split(train_indices, test_size=0.1, random_state=args.seed)

        # write split indices files to the disk
        if not os.path.exists(args.dataset_split_path):
            os.makedirs(args.dataset_split_path)
        split_names = ['train', 'eval', 'test']
        for split_name, split in zip(split_names, [train_indices, eval_indices, test_indices]):
            with open(f'{args.dataset_split_path}/{split_name}.indices', 'wb') as f:
                total_line = len(split)
                f.write(total_line.to_bytes(dataset.index_byte, 'big'))
                for indice in split:
                    f.write(indice.item().to_bytes(dataset.index_byte, 'big'))
                print(f'{split_name} size:', total_line)
    elif args.mode == 'train':
        dat_files = glob.glob(args.dataset_path)
        dataset = RefinedMlsesMapDataset(
            dat_files, input_dim=args.input_dim, label_transformer=label_transformer
        )

        # split training, developing, and testing datasets
        train_dataset = Subset(dataset, args.dataset_split_path, split='train')
        eval_dataset = Subset(dataset, args.dataset_split_path, split='eval')
        test_dataset = Subset(dataset, args.dataset_split_path, split='test')
        trainer.logger.info('Dataset details:')
        trainer.logger.info(f'total: {len(dataset)}')
        trainer.logger.info(f'train: {len(train_dataset)} eval: {len(eval_dataset)} test: {len(test_dataset)}')

        # start training
        trainer.train(train_dataset, eval_dataset)

        # report performance on testing dataset
        trainer.logger.info("Evaluate test dataset...")
        test_output, _ = trainer.eval(test_dataset, num_workers=args.eval_number_worker)
        for k, v in test_output.items():
            trainer.logger.info(f"{k}: {v}")
    elif args.mode == 'finetune':
        dat_files = glob.glob(args.dataset_path)
        dataset = RefinedMlsesMapDataset(
            dat_files, input_dim=args.input_dim, label_transformer=label_transformer
        )

        # split training, developing, and testing datasets
        train_dataset = Subset(dataset, args.dataset_split_path, split='train')
        eval_dataset = Subset(dataset, args.dataset_split_path, split='eval')
        test_dataset = Subset(dataset, args.dataset_split_path, split='test')
        trainer.logger.info('Dataset details:')
        trainer.logger.info(f'total: {len(dataset)}')
        trainer.logger.info(f'train: {len(train_dataset)} eval: {len(eval_dataset)} test: {len(test_dataset)}')

        # reload model checkpoint
        trainer.resume(args.resume_ckpt)

        trainer.logger.info("Evaluate eval dataset before finetuning...")
        eval_output, _ = trainer.eval(eval_dataset, num_workers=args.eval_number_worker)
        for k, v in eval_output.items():
            trainer.logger.info(f"{k}: {v}")
            trainer.writer.add_scalar(f'train/before_finetune_{k}', v, 0)

        # label transform don't do truncate during finetuing
        label_transformer.do_truncate = False

        # start finetuning
        trainer.logger.info("Start finetuning...")
        trainer.finetune(train_dataset, eval_dataset)

        # report performance on testing dataset
        trainer.logger.info("Evaluate test dataset...")
        test_output, _ = trainer.eval(test_dataset, num_workers=args.eval_number_worker)
        for k, v in test_output.items():
            trainer.logger.info(f"{k}: {v}")
    elif args.mode == 'eval':
        trainer.logger.info(f'Loading model checkpoint from {args.load_ckpt_path}...')
        trainer.model.load_state_dict(torch.load(args.load_ckpt_path, map_location=trainer.device))

        dat_files = glob.glob(args.dataset_path)
        save_dir = f'{args.ckpt_dir}/prediction-{current_time}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for dat_file in tqdm(dat_files):
            trainer.logger.info(f'Evaluating on {dat_file}...')
            dataset = RefinedMlsesMapDataset(
                [dat_file, ], input_dim=args.input_dim,
                label_transformer=label_transformer
            )

            log_dict, output = trainer.eval(dataset, num_workers=args.eval_number_worker)
            y_pred = output['y_pred']
            y_true = output['y_true']

            for k, v in log_dict.items():
                trainer.logger.info(f"{k}: {v}")

            dat_name = os.path.split(dat_file)[-1].replace('bench', 'mlses_pred')

            # post processing pred value
            if label_transformer is not None:
                y_pred = label_transformer.inv_transform(y_pred)
                y_true = label_transformer.inv_transform(y_true)
                log_dict = trainer.eval_on_prediction(y_pred, y_true)
                for k, v in log_dict.items():
                    trainer.logger.info(f"raw_{k}: {v}")

                y_pred = y_pred.cpu().numpy()
                y_true = y_true.cpu().numpy()
            else:
                y_pred = y_pred.cpu().numpy()
            np.savetxt(f'{save_dir}/{dat_name}', y_pred)
    elif args.mode == 'trace':
        trainer.logger.info(f'Loading model checkpoint from {args.load_ckpt_path}...')
        trainer.model.load_state_dict(torch.load(args.load_ckpt_path, map_location=trainer.device))

        cpu_model = trainer.model.cpu().eval()
        traced_model = torch.jit.trace(cpu_model, torch.randn(1, args.input_dim))
        torch.jit.save(traced_model, f'{args.ckpt_dir}/remlses_traced.pt')
        trainer.logger.info(traced_model.code)

        forzen_model = torch.jit.freeze(traced_model)
        torch.jit.save(forzen_model, f'{args.ckpt_dir}/remlses_frozen.pt')
        trainer.logger.info(forzen_model.code)

        # save all parameters to txt files
        for name, param in cpu_model.named_parameters():
            param_array: np.ndarray = param.detach().numpy()
            flatten_param = param_array.flatten()

            # format string
            fmt_param = []
            for v in flatten_param:
                fmt_param.append('%.10E' % v)

            # write to file with different required formats
            group_param = []
            for i in range(0, len(fmt_param), 4):
                param_str = fmt_param[i: i + 4]
                template = "{}, " * len(param_str)
                repr_str = template.format(*param_str)
                group_param.append(repr_str)

            with open(f'{args.ckpt_dir}/{name}_cuda.txt', 'w') as f:
                f.write('\n'.join(group_param))

            with open(f'{args.ckpt_dir}/{name}_fortran.txt', 'w') as f:
                f.write(' &\n'.join(group_param))

    elif args.mode == 'count':
        dat_files = glob.glob(args.dataset_path)
        dataset = RefinedMlsesMapDataset(dat_files)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.number_worker
        )

        pos_num = 0
        neg_num = 0
        for feat, label in tqdm(dataloader):
            pos_num += (label > 0).float().sum().item()
            neg_num += (label < 0).float().sum().item()

        ratio = pos_num / neg_num

        print("number of +1:", pos_num)
        print("number of -1:", neg_num)

        print("+1/-1:", ratio)
    elif args.mode == 'plot':
        dat_files = glob.glob(args.dataset_path)
        # dat_files = ["dataset/benchmark_data_0.2/bench_negative_0.79", ]
        dataset = RefinedMlsesMapDataset(dat_files, label_only=True)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.number_worker
        )

        datas = []
        for label in tqdm(dataloader):
            data = label.squeeze().flatten().tolist()
            datas.extend(data)

        counts, bins = np.histogram(datas, bins=1000)
        torch.save(datas, f'{args.ckpt_dir}/label.pt')
        trainer.logger.info(f"Number of data files loading: {len(dataset)}")
        trainer.logger.info(f"Total lines: {len(datas)}")
        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
        # axs.hist(datas, bins=50)
        axs.stairs(counts, bins)
        fig.savefig(f"{args.ckpt_dir}/label_dist_hist.png")
    elif args.mode == 'feature':
        dat_files = glob.glob(args.dataset_path)
        dataset = RefinedMlsesMapDataset(dat_files, label_only=True)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.number_worker
        )

        trivial_feature_num = []
        nontrivial_feature_num = []
        for feature_num, label in tqdm(dataloader):
            trivial_mask = label.squeeze().flatten() < -2
            feature_num = feature_num.squeeze().flatten()
            nontrivial_mask = torch.logical_not(trivial_mask)

            trivial_feature_num.extend(feature_num[trivial_mask].tolist())
            nontrivial_feature_num.extend(feature_num[nontrivial_mask].tolist())

        torch.save(
            {'trivial_feature_num': trivial_feature_num, 'nontrivial_feature_num': nontrivial_feature_num},
            f'{args.ckpt_dir}/feature_dist.pt'
        )
        trivial_counts, trivial_bins = np.histogram(trivial_feature_num, bins=1000, density=True)
        nontrivial_counts, nontrivial_bins = np.histogram(nontrivial_feature_num, bins=1000, density=True)
        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
        axs.stairs(trivial_counts, trivial_bins, label='trivial')
        axs.stairs(nontrivial_counts, nontrivial_bins, label='nontrivial')
        axs.legend()
        fig.savefig(f"{args.ckpt_dir}/feature_num_dist_hist.png")
