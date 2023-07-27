import time
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import _classification
from tqdm import tqdm
import os
import wandb


class Video2Roll_Trainer(object):
    def __init__(
        self,
        data_loader,
        test_data_loader,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        epochs,
        save_model_path,
        device,
    ):
        # self.save_model_path = './models/Video2Roll.pth' # change to your path
        self.save_model_path_loss = os.path.join(save_model_path, "Video2Roll_bestloss.pth")
        self.save_model_path_f1 = os.path.join(save_model_path, "Video2Roll_bestf1.pth")
        self.test_loader = test_data_loader
        self.data_loader = data_loader
        self.net = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        # Training config
        self.epochs = epochs
        self.device = device
        # logging
        self.step = 0
        self.global_step = 0
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.val_loss = torch.zeros(self.epochs)

    def train(self):
        # Train model multi-epoches
        pre_val_loss = 1e4
        pre_f1_score = 0.0
        for epoch in range(self.epochs):
            print("Training...")
            self.net.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            # training loop
            tr_avg_loss, tr_avg_precision, tr_avg_recall = self.train_loop()

            # evaluate
            print("Validating...")
            self.net.eval()
            metric_dict = self.validate()
            val_metric_best = 0
            val_f1_best = 0
            print("-" * 85)
            print(
                "Train Summary | Epoch {0} | Time {1:.2f}s | "
                "Train Loss {2:.3f}".format(
                    epoch + 1, time.time() - start, tr_avg_loss, tr_avg_precision, tr_avg_recall
                )
            )
            for ds_idx, metric_per_ds in metric_dict.items():
                print(ds_idx)
                print(metric_per_ds)
                if ds_idx == "ds0":
                    val_metric_best = metric_per_ds["val_loss"]
                    val_f1_best = metric_per_ds["val_f1_score_2"]
                wandb.log(
                    {
                        f"{ds_idx}_train_loss": tr_avg_loss,
                        f"{ds_idx}_train_precision": tr_avg_precision,
                        f"{ds_idx}_train_recall": tr_avg_recall,
                        f"{ds_idx}_val_loss": metric_per_ds["val_loss"],
                        f"{ds_idx}_val_precision": metric_per_ds["val_precision"],
                        f"{ds_idx}_val_recall": metric_per_ds["val_recall"],
                        f"{ds_idx}_val_accuracy": metric_per_ds["accuracy"],
                        f"{ds_idx}_val_f1": metric_per_ds["f1_score"],
                        f"{ds_idx}_val_precision(re)": metric_per_ds["val_precision_2"],
                        f"{ds_idx}_val_recall(re)": metric_per_ds["val_recall_2"],
                        f"{ds_idx}_val_f1(re)": metric_per_ds["val_f1_score_2"],
                    }
                )
            print("-" * 85)

            if val_metric_best < pre_val_loss:
                print(f"validation loss improved {pre_val_loss} -> {val_metric_best}")
                pre_val_loss = val_metric_best
                torch.save(self.net.state_dict(), self.save_model_path_loss)
            if val_f1_best > pre_f1_score:
                print(f"validation f1 improved {pre_f1_score} -> {val_f1_best}")
                pre_f1_score = val_f1_best
                torch.save(self.net.state_dict(), self.save_model_path_f1)
            # Save model each epoch
            self.val_loss[epoch] = val_metric_best
            self.tr_loss[epoch] = tr_avg_loss

    def train_loop(self):
        data_loader = self.data_loader
        epoch_loss = 0
        epoch_precision = 0
        epoch_recall = 0
        count = 0
        start = time.time()

        for i, data in enumerate(data_loader):
            imgs, label = data
            imgs, label = imgs.to(self.device), label.to(self.device)
            logits = self.net(imgs)
            loss = self.criterion(logits, label)
            # set the threshold of the logits
            pred_label = torch.sigmoid(logits) >= 0.4
            numpy_label = label.cpu().detach().numpy().astype(int)
            numpy_pre_label = pred_label.cpu().detach().numpy().astype(int)

            precision = metrics.precision_score(
                numpy_label, numpy_pre_label, average="samples", zero_division=1
            )
            recall = metrics.recall_score(
                numpy_label, numpy_pre_label, average="samples", zero_division=1
            )
            if self.global_step % 30 == 0:
                end = time.time()
                print(
                    "step {0}/{5} loss:{1:.4f} | precision:{2:.3f} | recall:{3:.3f} | time:{4:.2f}".format(
                        i, loss.item(), precision, recall, end - start, len(data_loader)
                    )
                )
                start = end

            epoch_precision += precision
            epoch_recall += recall
            epoch_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            count += 1
            self.global_step += 1
        self.lr_scheduler.step(epoch_loss / count)
        return epoch_loss / count, epoch_precision / count, epoch_recall / count

    def validate(self):
        val_metrics = {}
        for tl_idx, tl in enumerate(self.test_loader):
            epoch_loss = 0
            count = 0
            all_pred_label = []
            all_label = []
            with torch.no_grad():
                for i, data in enumerate(tqdm(tl)):
                    imgs, label = data
                    imgs, label = imgs.to(self.device), label.to(self.device)
                    logits = self.net(imgs)
                    loss = self.criterion(logits, label)
                    pred_label = torch.sigmoid(logits) >= 0.4
                    numpy_label = label.cpu().detach().numpy().astype(int)
                    numpy_pre_label = pred_label.cpu().detach().numpy().astype(int)
                    all_label.append(numpy_label)
                    all_pred_label.append(numpy_pre_label)
                    epoch_loss += loss.item()
                    count += 1
            all_label = np.vstack(all_label)
            all_pred_label = np.vstack(all_pred_label)
            labels = _classification._check_set_wise_labels(
                all_label, all_pred_label, labels=None, pos_label=1, average="samples"
            )
            MCM = metrics.multilabel_confusion_matrix(
                all_label, all_pred_label, sample_weight=None, labels=labels, samplewise=True
            )
            tp_sum = MCM[:, 1, 1]
            fp_sum = MCM[:, 0, 1]
            fn_sum = MCM[:, 1, 0]
            # tn_sum = MCM[:, 0, 0]
            accuracy = _prf_divide(tp_sum, tp_sum + fp_sum + fn_sum, zero_division=1)
            accuracy = np.average(accuracy)
            all_precision = metrics.precision_score(
                all_label, all_pred_label, average="samples", zero_division=1
            )
            all_recall = metrics.recall_score(
                all_label, all_pred_label, average="samples", zero_division=1
            )
            all_f1_score = metrics.f1_score(
                all_label, all_pred_label, average="samples", zero_division=1
            )

            precision_re = metrics.precision_score(
                all_label, all_pred_label, average=None, zero_division=np.nan
            )
            recall_re = metrics.recall_score(
                all_label, all_pred_label, average=None, zero_division=np.nan
            )
            f1_score_re = metrics.f1_score(
                all_label, all_pred_label, average=None, zero_division=np.nan
            )

            precision_re = (np.nansum(precision_re) / np.count_nonzero(~np.isnan(precision_re)),)
            recall_re = (np.nansum(recall_re) / np.count_nonzero(~np.isnan(recall_re)),)
            f1_score_re = np.nansum(f1_score_re) / np.count_nonzero(~np.isnan(f1_score_re))

            val_metrics[f"ds{tl_idx}"] = {
                "val_loss": epoch_loss / count,
                "val_precision": all_precision,
                "val_recall": all_recall,
                "accuracy": accuracy,
                "f1_score": all_f1_score,
                "val_precision_2": precision_re[0],
                "val_recall_2": recall_re[0],
                "val_f1_score_2": f1_score_re,
            }

        return val_metrics


def _prf_divide(numerator, denominator, zero_division="warn"):
    """Performs division and handles divide-by-zero.
    On zero-division, sets the corresponding result elements equal to
    0 or 1 (according to ``zero_division``). Plus, if
    ``zero_division != "warn"`` raises a warning.
    The metric, modifier and average arguments are used only for determining
    an appropriate warning.
    """
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator

    if not np.any(mask):
        return result

    # if ``zero_division=1``, set those with denominator == 0 equal to 1
    result[mask] = 0.0 if zero_division in ["warn", 0] else 1.0

    # the user will be removing warnings if zero_division is set to something
    # different than its default value. If we are computing only f-score
    # the warning will be raised only if precision and recall are ill-defined
    if zero_division != "warn":
        return result
