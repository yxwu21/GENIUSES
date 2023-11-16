import logging
import os
import torch

from torch.nn import L1Loss, MSELoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from src.model import SquaredHingeLoss, PerceptronLoss
from src.utils import tanh_acc, tanh_f1, mae_score, relative_mae_score, eval_r2_score


class Trainer:
    def __init__(self, args, model):
        # set parsed arguments
        self.args = args

        # init logger and tensorboard
        self._init_logger()
        self._set_writer()

        # init ingredients
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # init model
        self.model = model
        self.model = self.model.to(self.device)

        # init optimizer and learning rate scheduler
        self.optimizer = SGD(
            self.model.parameters(),
            lr=self.args.lr
        )

        # log status
        self.logger.info('Experiment setting:')
        for k, v in sorted(vars(self.args).items()):
            self.logger.info(f'{k}: {v}')

    def _get_lr(self, epoch_index, min_lr=1e-6) -> float:
        start_reduce_epoch = self.args.epoch // 2
        if epoch_index < start_reduce_epoch:
            return self.args.lr

        delta_lr = (self.args.lr - min_lr) / (self.args.epoch - start_reduce_epoch)
        next_lr = self.args.lr - delta_lr * (epoch_index - start_reduce_epoch)
        return next_lr

    def resume(self, resume_ckpt_path: str):
        # resume checkpoint
        self.logger.info(f'Resume model checkpoint from {resume_ckpt_path}...')
        self.model.load_state_dict(torch.load(resume_ckpt_path))

    def train_loop(self, train_dataset, eval_dataset, step_func):
        """Training loop function for model training and finetuning.

        :param train_dataset: training dataset
        :param eval_dataset: evaluation dataset
        :param step_func: a callable function doing forward and optimize step and return loss log
        """
        self.model.train()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.number_worker
        )

        global_step = 0
        for epoch in range(0, self.args.epoch):
            # update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self._get_lr(epoch)

            # train steps
            for step, (feat, label) in enumerate(train_dataloader):
                feat: torch.Tensor = feat.to(self.device)
                label: torch.Tensor = label.to(self.device)

                # run step
                input_feats = {'feat': feat, 'label': label}
                loss_log = step_func(input_feats)

                # print loss
                if step % self.args.log_freq == 0:
                    loss_str = ' '.join([f'{k}: {v}' for k, v in loss_log.items()])
                    self.logger.info(f"Epoch: {epoch} Step: {step} | Loss: {loss_str}")
                    for k, v in loss_log.items():
                        self.writer.add_scalar(f'train/{k}', v, global_step)

                    # log current learning rate
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar(
                        'train/lr',
                        current_lr,
                        global_step
                    )

                # increase step
                global_step += 1

            if epoch % self.args.eval_freq == 0:
                self.logger.info(f"Evaluate eval dataset at epoch {epoch}...")
                eval_output, _ = self.eval(eval_dataset)
                for k, v in eval_output.items():
                    self.logger.info(f"{k}: {v}")
                    self.writer.add_scalar(f'train/eval_{k}', v, epoch)

                torch.save(
                    self.model.state_dict(),
                    f'{self.args.ckpt_dir}/model_{epoch}.pth'
                )

        # save the final model after training
        torch.save(
            self.model.state_dict(),
            f'{self.args.ckpt_dir}/model_final.pth'
        )

    def train_step(self, input_feats):
        feat, label = input_feats['feat'], input_feats['label']

        # clean gradient and forward
        self.optimizer.zero_grad()
        pred = self.model(feat)

        # compute loss
        regr_loss_fn = L1Loss()
        regr_loss = regr_loss_fn(pred, label)
        # sign_loss_fn = PerceptronLoss(self.args.sign_threshold)
        # sign_loss = sign_loss_fn(pred, label)
        sign_loss = torch.zeros_like(regr_loss)
        loss = regr_loss + sign_loss * self.args.lambda1

        # backward and update parameters
        loss.backward()
        self.optimizer.step()

        # prepare log dict
        log = {'loss': loss.item(), 'regr_loss': regr_loss.item(), 'sign_loss': sign_loss.item()}
        return log

    def finetune_step(self, input_feats):
        feat, label = input_feats['feat'], input_feats['label']

        # clean gradient and forward
        self.optimizer.zero_grad()
        pred = self.model(feat)

        # compute loss
        # regr_loss_fn = L1Loss()
        # regr_loss = regr_loss_fn(pred, label)
        sign_loss_fn = PerceptronLoss(self.args.sign_threshold)
        sign_loss = sign_loss_fn(pred, label)
        regr_loss = torch.zeros_like(sign_loss)
        loss = sign_loss

        # backward and update parameters
        loss.backward()
        self.optimizer.step()

        # prepare log dict
        log = {'loss': loss.item(), 'regr_loss': regr_loss.item(), 'sign_loss': sign_loss.item()}
        return log

    def train(self, train_dataset, eval_dataset):
        self.train_loop(train_dataset, eval_dataset, self.train_step)

    def finetune(self, train_dataset, eval_dataset):
        self.train_loop(train_dataset, eval_dataset, self.finetune_step)

    @torch.inference_mode()
    def eval(self, dataset, num_workers: int = 0):
        self.model.eval()
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        y_pred_list = []
        y_true_list = []
        for feat, label in tqdm(dataloader):
            feat: torch.Tensor = feat.to(self.device)
            pred = self.model(feat)

            y_pred_list.append(pred.cpu())
            y_true_list.append(label)

        y_pred = torch.cat(y_pred_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)
        score = self.eval_on_prediction(y_pred, y_true)

        output = {}
        output['y_pred'] = y_pred
        output['y_true'] = y_true
        return score, output

    @torch.inference_mode()
    def eval_on_prediction(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        mae = mae_score(y_pred, y_true)
        r2 = eval_r2_score(y_pred, y_true)

        threshold = self.args.sign_threshold
        diff_sign_mask = torch.logical_or(
            torch.logical_and(y_true < threshold, y_pred > threshold),
            torch.logical_and(y_true > threshold, y_pred < threshold)
        )
        sign_error_num = diff_sign_mask.float().sum().item()

        score = {}
        score['absolute_mae'] = mae
        score['r2'] = r2
        score['sign_error_num'] = sign_error_num
        return score

    def _set_writer(self):
        self.logger.info('Create writer at \'{}\''.format(self.args.ckpt_dir))
        self.writer = SummaryWriter(self.args.ckpt_dir)

    def _init_logger(self):
        logging.basicConfig(
            filename=os.path.join(self.args.ckpt_dir, f'mlses_{self.args.mode}.log'),
            level=logging.INFO,
            datefmt='%Y/%m/%d %H:%M:%S',
            format='%(asctime)s: %(name)s [%(levelname)s] %(message)s'
        )
        formatter = logging.Formatter(
            '%(asctime)s: %(name)s [%(levelname)s] %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S'
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(stream_handler)
