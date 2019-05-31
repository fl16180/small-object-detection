import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from model.metric import calculate_mAP

from constants import DEVICE

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.multiboxloss = loss(threshold=0.5, neg_pos_ratio=3,
                                 alpha=1., device=DEVICE)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        for batch_idx, (data, boxes, labels, _) in enumerate(self.data_loader):

            if batch_idx > 10:
                continue
            data = data.to(DEVICE)
            boxes = [b.to(DEVICE) for b in boxes]
            labels = [l.to(DEVICE) for l in labels]

            self.optimizer.zero_grad()
            output_boxes, output_scores = self.model(data)

            loss = self.multiboxloss(output_boxes, output_scores, boxes, labels)
            loss.backward()

            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            metric_log = self._valid_metric(epoch)

            log = {**log, **val_log, **metric_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        """
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, boxes, labels, _) in enumerate(self.valid_data_loader):
                if batch_idx > 10:
                    continue
                data = data.to(DEVICE)
                boxes = [b.to(DEVICE) for b in boxes]
                labels = [l.to(DEVICE) for l in labels]

                output_boxes, output_scores = self.model(data)

                loss = self.multiboxloss(output_boxes, output_scores, boxes, labels)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                # if batch_idx % self.log_step == 0:
                #     self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                #         epoch,
                #         batch_idx * self.data_loader.batch_size,
                #         self.data_loader.n_samples,
                #         100.0 * batch_idx / len(self.data_loader),
                #         loss.item()))
                #     self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))


        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
        }

    def _valid_metric(self, epoch):
        """ Compute mAP metric over validation set after certain number of
            epochs. Because the metric is computed over the whole dataset,
            I separated it from the validation loss method to lower
            training time.

            Return: A log with information about metrics
        """
        self.model.eval()

        # must compute mAP over entire dataset
        all_boxes = list()
        all_labels = list()
        all_scores = list()
        all_true_boxes = list()
        all_true_labels = list()
        all_difficulties = list()

        with torch.no_grad():
            for batch_idx, (data, boxes, labels, difficulties) in enumerate(self.valid_data_loader):
                if batch_idx > 10:
                    continue
                print(batch_idx)
                data = data.to(DEVICE)
                boxes = [b.to(DEVICE) for b in boxes]
                labels = [l.to(DEVICE) for l in labels]
                difficulties = [d.to(DEVICE) for d in difficulties]

                output_boxes, output_scores = self.model(data)

                batch_boxes, batch_labels, batch_scores = self.model.detect_objects(output_boxes, output_scores,
                                                                               min_score=0.01, max_overlap=0.45,
                                                                               top_k=200)

                all_boxes.extend(batch_boxes)
                all_labels.extend(batch_labels)
                all_scores.extend(batch_scores)
                all_true_boxes.extend(boxes)
                all_true_labels.extend(labels)
                all_difficulties.extend(difficulties)

            # Calculate mAP
            class_APs, mAP = calculate_mAP(all_boxes, all_labels, all_scores, all_true_boxes, all_true_labels, all_difficulties)
            # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            # self.writer.add_scalar('loss', loss.item())
            # total_val_loss += loss.item()
            # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        return {
            'val_mAP': mAP,
            'val_class_AP': class_APs
        }
