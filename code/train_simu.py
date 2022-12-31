import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from layers import ZINBLoss, MeanAct, DispAct, GaussianNoise
import numpy as np
from sklearn.cluster import KMeans
import math, os
from sklearn import metrics
from preprocess import *
from augmentation import *
import argparse
import random
from itertools import cycle
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import glob
import multiprocessing as mp
import gc
import pandas as pd


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target):
    num_correct = np.sum(output == target)
    res = num_correct / len(target)
    return res


def cluster_acc(y_pred, y_true):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def auxilarly_dis(pred):
    weight = (pred ** 2) / torch.sum(pred, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def entropy(x):
    """
    Helper function to compute the entropy over the batch
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    EPS = 1e-8
    x_ =  torch.clamp(x, min = EPS)
    b =  x_ * torch.log(x_)

    if len(b.size()) == 2: # Sample-wise entropy
        return - b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


def buildNetwork(layers, activation="relu", noise=False, batchnorm=False):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if noise:
            net.append(GaussianNoise())
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        if batchnorm:
            net.append(nn.BatchNorm1d(layers[i]))
    return nn.Sequential(*net)


class Prototype(nn.Module):
    def __init__(self, num_classes, input_size, tau=1.0):
        super(Prototype, self).__init__()
        self.cluster = Parameter(torch.Tensor(num_classes, input_size))
        nn.init.xavier_uniform_(self.cluster)
        self.tau = tau

    def forward(self, x):
        dist = torch.sum(torch.square(x.unsqueeze(1) - self.cluster), dim=2)
        sim = 1.0 / ((1.0 + dist / self.tau) ** ((self.tau + 1.0) / 2.0))
        return sim


class scOpen1(nn.Module):
    def __init__(self, input_dim, z_dim, shared_classes, total_classes, num_batches, encodeLayer=[], decodeLayer=[], activation="relu", tau=1.0):
        super(scOpen1, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.shared_classes = shared_classes
        self.total_classes = total_classes
        self.num_batches = num_batches
        self.activation = activation
        self.tau = tau
        self.encoder = buildNetwork([self.input_dim] + encodeLayer, activation=activation, noise=True, batchnorm=False)
        self.decoder = buildNetwork([self.z_dim + self.num_batches] + decodeLayer, activation=activation, batchnorm=False)
        self._enc_mu = nn.Linear(encodeLayer[-1], self.z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], self.input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], self.input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], self.input_dim), nn.Sigmoid())
        self._dec_mask = nn.Sequential(nn.Linear(decodeLayer[-1], self.input_dim), nn.Sigmoid())
        self.source_classifier = Prototype(self.shared_classes, self.z_dim, self.tau)
        self.target_classifier = Prototype(self.total_classes, self.z_dim, self.tau)

    def forward(self, x, batch):
        h = self.encoder(x)
        z = self._enc_mu(h)
        h = self.decoder(torch.cat([z, batch], dim=1))
        mean = self._dec_mean(h)
        disp = self._dec_disp(h)
        pi = self._dec_pi(h)
        mask = self._dec_mask(h)
        out_s = self.source_classifier(z)
        out_t = self.target_classifier(z)
        return z, mean, disp, pi, mask, out_s, out_t


def extractor(model, test_loader, device):
    model.eval()
    test_embedding = []
    test_output = []
    test_label = []
    test_index = []
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            x_t, label_t, index_t, batch_t = data[0].to(device), data[3].to(device), data[4].to(device), data[5].to(device)
            z_t, _, _, _, _, out_s, out_t = model(x_t, batch_t)
            test_embedding.append(z_t.detach())
            test_output.append(out_t.detach())
            test_label.append(label_t)
            test_index.append(index_t)
    test_embedding = torch.cat(test_embedding, dim=0)
    test_output = torch.cat(test_output, dim=0)
    test_label = torch.cat(test_label)
    test_index = torch.cat(test_index)
    _, test_indexes = torch.sort(test_index, descending=False)
    test_embedding = test_embedding[test_indexes]
    test_output = test_output[test_indexes]
    test_label = test_label[test_indexes]
    return test_embedding, test_output, test_label


def cluster_match(train_clusters, test_clusters):
    cluster_similarity = torch.sum(torch.square(train_clusters.unsqueeze(1) - test_clusters), dim=2)
    cluster_mapping = torch.argmin(cluster_similarity, dim=1)
    return cluster_mapping.cpu().numpy()


def test_new2(model, labeled_num, device, test_loader, cluster_mapping, epoch):
    model.eval()
    preds = np.array([])
    preds_open = np.array([])
    targets = np.array([])
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            x_t, label_t, index_t, batch_t = data[0].to(device), data[3].to(device), data[4].to(device), data[5].to(device)
            z, _, _, _, _, output_s, output_t = model(x_t, batch_t)
            output_c = torch.cat([output_s, output_t], dim=1)
            conf_c, pred_c = output_c.max(1)
            conf_t, pred_t = output_t.max(1)
            targets = np.append(targets, label_t.cpu().numpy())
            preds = np.append(preds, pred_c.cpu().numpy())
            preds_open = np.append(preds_open, pred_t.cpu().numpy())
    cluster_mapping = cluster_mapping + labeled_num
    for i in range(len(cluster_mapping)):
        preds[preds == cluster_mapping[i]] = i
    targets = targets.astype(int)
    preds = preds.astype(int)
    preds_open = preds_open.astype(int)
    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    overall_acc = cluster_acc(preds, targets)
    overall_acc2 = cluster_acc(preds_open, targets)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    seen_acc2 = cluster_acc(preds_open[seen_mask], targets[seen_mask])
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    unseen_acc2 = cluster_acc(preds_open[unseen_mask], targets[unseen_mask])
    print('In the old {}-th epoch, Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(epoch, overall_acc,
                                                                                                   seen_acc,
                                                                                                   unseen_acc))
    print('In the old {}-th epoch, Test overall acc2 {:.4f}, seen acc2 {:.4f}, unseen acc2 {:.4f}'.format(epoch, overall_acc2,
                                                                                                    seen_acc2,
                                                                                                    unseen_acc2))
    return overall_acc, seen_acc, unseen_acc, overall_acc2, seen_acc2, unseen_acc2


def main(args, X, Y, batch_label, device, cluster_num, ratio, mode, dropout_rate, dataseed):
    Y = Y.astype(np.int)
    ori_num_classes = len(np.unique(Y))
    source_X = X[batch_label == 0]
    source_Y = Y[batch_label == 0]
    source_batch = batch_label[batch_label == 0]
    target_X = X[batch_label == 1]
    target_Y = Y[batch_label == 1]
    target_batch = batch_label[batch_label == 1]

    for i in range(args.removal):
        source_X = source_X[source_Y != (ori_num_classes - i - 1)]
        source_batch = source_batch[source_Y != (ori_num_classes - i - 1)]
        source_Y = source_Y[source_Y != (ori_num_classes - i - 1)]
    total_classes = ori_num_classes
    shared_classes = ori_num_classes - args.removal

    X = np.concatenate((source_X, target_X), axis=0)
    Y = np.concatenate((source_Y, target_Y))
    batch_label = np.concatenate((source_batch, target_batch))
    count_X = X
    if X.shape[1] == args.highly_genes:
        args.highly_genes = None

    adata = sc.AnnData(X)
    adata.obs["batch"] = batch_label
    adata.obs["celltype"] = Y
    adata = normalize(adata, highly_genes=args.highly_genes, size_factors=True, normalize_input=True,
                      logtrans_input=True)
    X = adata.X.astype(np.float32)
    Y = np.array(adata.obs["celltype"])
    batch_label = np.array(adata.obs["batch"])

    if args.highly_genes != None:
        high_variable = np.array(adata.var.highly_variable.index, dtype=np.int)
        count_X = count_X[:, high_variable]
    else:
        select_genes = np.array(adata.var.index, dtype=np.int)
        select_cells = np.array(adata.obs.index, dtype=np.int)
        count_X = count_X[:, select_genes]
        count_X = count_X[select_cells]
    assert X.shape == count_X.shape
    size_factor = np.array(adata.obs.size_factors).reshape(-1, 1).astype(np.float32)

    batch_matrix = OneHotEncoder().fit_transform(batch_label.reshape(-1, 1)).toarray()

    source_x = X[batch_label == 0]
    source_raw_x = count_X[batch_label == 0]
    source_y = Y[batch_label == 0]
    source_sf = size_factor[batch_label == 0]
    source_b = batch_label[batch_label == 0]
    source_batch = batch_matrix[batch_label == 0]

    target_x = X[batch_label == 1]
    target_raw_x = count_X[batch_label == 1]
    target_y = Y[batch_label == 1]
    target_sf = size_factor[batch_label == 1]
    target_b = batch_label[batch_label == 1]
    target_batch = batch_matrix[batch_label == 1]
    num_batches = 2

    if args.structure == 0:
        model = scOpen1(X.shape[1], 32, shared_classes, total_classes, num_batches, encodeLayer=[256, 64],
                        decodeLayer=[64, 256], activation="relu", tau=args.tau)
    else:
        model = scOpen1(X.shape[1], 128, shared_classes, total_classes, num_batches, encodeLayer=[512, 256],
                        decodeLayer=[256, 512], activation="relu", tau=args.tau)
    model = model.to(device)

    source_x = torch.tensor(source_x)
    source_raw_x = torch.tensor(source_raw_x)
    source_sf = torch.tensor(source_sf)
    source_y = torch.tensor(source_y)
    source_batch = torch.tensor(source_batch).float()
    target_x = torch.tensor(target_x)
    target_raw_x = torch.tensor(target_raw_x)
    target_sf = torch.tensor(target_sf)
    target_y = torch.tensor(target_y)
    target_batch = torch.tensor(target_batch).float()

    source_dataset = TensorDataset(source_x, source_raw_x, source_sf, source_y, torch.arange(source_x.shape[0]), source_batch)
    source_dataloader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    target_dataset = TensorDataset(target_x, target_raw_x, target_sf, target_y, torch.arange(target_x.shape[0]), target_batch)
    target_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_dataset = TensorDataset(source_x, source_raw_x, source_sf, source_y, torch.arange(source_x.shape[0]), source_batch)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataset = TensorDataset(target_x, target_raw_x, target_sf, target_y, torch.arange(target_x.shape[0]), target_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    best_overall_acc = 0.
    best_seen_acc = 0.
    best_unseen_acc = 0.

    best_overall_acc2 = 0.
    best_seen_acc2 = 0.
    best_unseen_acc2 = 0.

    final_overall_acc = 0.
    final_seen_acc = 0.
    final_unseen_acc = 0.

    final_overall_acc2 = 0.
    final_seen_acc2 = 0.
    final_unseen_acc2 = 0.

    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()

    for epoch in range(args.epochs + 1):
        if epoch % args.interval == 0:
            train_embeddings, train_outputs, train_targets = extractor(model, train_dataloader, device)
            test_embeddings, test_outputs, test_targets = extractor(model, test_dataloader, device)
            if epoch <= args.pretrain:
                kmeans_ = KMeans(n_clusters=total_classes, init="k-means++", random_state=args.random_seed).fit(
                    test_embeddings.cpu().numpy())
                test_label, cluster_centers = torch.from_numpy(kmeans_.labels_).to(device), torch.from_numpy(
                    kmeans_.cluster_centers_).to(device)

                test_mask = torch.zeros_like(test_label).float()
                for i in range(total_classes):
                    test_distance = torch.sum(torch.square(test_embeddings[test_label == i].unsqueeze(1) - cluster_centers), dim=2)
                    intra_distance = test_distance[:, i]
                    inter_distance = torch.min(test_distance[:, torch.arange(test_distance.shape[1]) != i], dim=1)[0]
                    test_score = -intra_distance / inter_distance
                    test_mask[test_label == i] = torch.where(test_score >= torch.quantile(test_score, args.quantile),
                                                             torch.ones_like(test_score), torch.zeros_like(test_score))
                test_label = test_label.to(train_targets.dtype)

                label_centers = torch.zeros(shared_classes, cluster_centers.shape[1]).to(device)
                for i in range(shared_classes):
                    label_centers[i] = torch.mean(train_embeddings[train_targets == i], dim=0)
                state_dict = model.state_dict()
                state_dict['source_classifier.cluster'] = label_centers
                state_dict['target_classifier.cluster'] = cluster_centers
                model.load_state_dict(state_dict)
                cluster_mappings = cluster_match(label_centers, cluster_centers)
                overall_acc, seen_acc, unseen_acc, overall_acc2, seen_acc2, unseen_acc2 = test_new2(model,
                                                                                                    shared_classes,
                                                                                                    device,
                                                                                                    test_dataloader,
                                                                                                    cluster_mappings,
                                                                                                    epoch)
            else:
                test_label = torch.max(test_outputs, dim=1)[1].to(train_targets.dtype)
                test_mask = torch.zeros_like(test_label).float()
                label_centers = model.source_classifier.cluster.data
                cluster_centers = model.target_classifier.cluster.data
                for i in range(total_classes):
                    if torch.sum(test_label == i) >= 5:
                        test_distance = torch.sum(
                            torch.square(test_embeddings[test_label == i].unsqueeze(1) - cluster_centers), dim=2)
                        intra_distance = test_distance[:, i]
                        inter_distance = torch.min(test_distance[:, torch.arange(test_distance.shape[1]) != i], dim=1)[
                            0]
                        test_score = -intra_distance / inter_distance
                        test_mask[test_label == i] = torch.where(
                            test_score >= torch.quantile(test_score, args.quantile),
                            torch.ones_like(test_score), torch.zeros_like(test_score))
                cluster_mappings = cluster_match(label_centers, cluster_centers)
                overall_acc, seen_acc, unseen_acc, overall_acc2, seen_acc2, unseen_acc2 = test_new2(model,
                                                                                                    shared_classes,
                                                                                                    device,
                                                                                                    test_dataloader,
                                                                                                    cluster_mappings,
                                                                                                    epoch)
            if overall_acc > best_overall_acc:
                best_overall_acc = overall_acc
                best_seen_acc = seen_acc
                best_unseen_acc = unseen_acc
            if overall_acc2 > best_overall_acc2:
                best_overall_acc2 = overall_acc2
                best_seen_acc2 = seen_acc2
                best_unseen_acc2 = unseen_acc2
            print(
                "Currently, we have the best overall acc is {:.4f}, best seen acc is {:.4f}, "
                "best unseen acc is {:.4f}".format(best_overall_acc, best_seen_acc, best_unseen_acc))
            print(
                "Currently, we have the best overall acc2 is {:.4f}, best seen acc2 is {:.4f}, "
                "best unseen acc2 is {:.4f}".format(best_overall_acc2, best_seen_acc2, best_unseen_acc2))

            if epoch == args.epochs:
                final_overall_acc = overall_acc
                final_seen_acc = seen_acc
                final_unseen_acc = unseen_acc

                final_overall_acc2 = overall_acc2
                final_seen_acc2 = seen_acc2
                final_unseen_acc2 = unseen_acc2

                print(
                    "After training, we have the final overall acc is {:.4f}, final seen acc is {:.4f}, final unseen acc is {:.4f}".format(
                        final_overall_acc, final_seen_acc,
                        final_unseen_acc))
                print(
                    "After training, we have the final overall acc2 is {:.4f}, final seen acc2 is {:.4f}, final unseen acc2 is {:.4f}".format(
                        final_overall_acc2, final_seen_acc2,
                        final_unseen_acc2))


        source_dataloader_iter = cycle(source_dataloader)
        recon_losses = AverageMeter('recon_loss', ':.4e')
        ce_losses = AverageMeter('ce_loss', ':.4e')
        align_losses = AverageMeter('align_loss', ':.4e')
        pse_losses = AverageMeter('pse_loss', ':.4e')
        clu_losses = AverageMeter('clu_loss', ':.4e')
        model.train()

        for batch_idx, (x_t, raw_x_t, sf_t, y_t, index_t, batch_t) in enumerate(target_dataloader):
            (x_s, raw_x_s, sf_s, y_s, index_s, batch_s) = next(source_dataloader_iter)
            x_s, raw_x_s, sf_s, y_s, index_s, batch_s = x_s.to(device), raw_x_s.to(device), \
                                                        sf_s.to(device), y_s.to(device), \
                                                        index_s.to(device), batch_s.to(device)
            x_t, raw_x_t, sf_t, index_t, batch_t = x_t.to(device), raw_x_t.to(device), \
                                                   sf_t.to(device), index_t.to(device), batch_t.to(device)

            labeled_len = len(y_s)
            x_all = torch.cat([x_s, x_t], dim=0)
            mask_all_ = mask_generator(args.noi, x_all.cpu().numpy())
            mask_all, x_all2 = pretext_generator(mask_all_, x_all.cpu().numpy())
            mask_all = torch.from_numpy(mask_all).to(torch.float32).to(device)
            x_all2 = torch.from_numpy(x_all2).to(torch.float32).to(device)
            x_transform1 = x_all
            x_transform2 = x_all2

            z_s1, mean_s1, disp_s1, pi_s1, mask_s1, output_ss1, output_st1 = model(x_transform1[:labeled_len], batch_s)
            z_s2, mean_s2, disp_s2, pi_s2, mask_s2, output_ss2, output_st2 = model(x_transform2[:labeled_len], batch_s)
            z_t1, mean_t1, disp_t1, pi_t1, mask_t1, output_ts1, output_tt1 = model(x_transform1[labeled_len:], batch_t)
            z_t2, mean_t2, disp_t2, pi_t2, mask_t2, output_ts2, output_tt2 = model(x_transform2[labeled_len:], batch_t)

            recon_loss = ZINBLoss().to(device)(x=raw_x_s, mean=mean_s1, disp=disp_s1, pi=pi_s1, scale_factor=sf_s) \
                         + ZINBLoss().to(device)(x=raw_x_t, mean=mean_t1, disp=disp_t1, pi=pi_t1, scale_factor=sf_t)
            recon_loss = recon_loss - torch.mean(
                mask_all[:labeled_len] * torch.log(mask_s2) + (1. - mask_all[:labeled_len]) * torch.log(1. - mask_s2)) \
                         - torch.mean(
                mask_all[labeled_len:] * torch.log(mask_t2) + (1. - mask_all[labeled_len:]) * torch.log(1. - mask_t2))

            feat1 = torch.cat([z_s1, z_t1], dim=0)
            feat2 = torch.cat([z_s2, z_t2], dim=0)
            output1 = torch.cat([output_st1, output_tt1], dim=0)
            output2 = torch.cat([output_st2, output_tt2], dim=0)
            prob1 = output1 / torch.sum(output1, dim=1, keepdim=True)
            prob2 = output2 / torch.sum(output2, dim=1, keepdim=True)
            _, pred1 = prob1.max(1)
            _, pred2 = prob2.max(1)

            label_gt = (y_s.view(-1, 1) - y_s.view(1, -1)).float()
            label_gt = torch.where(label_gt == 0., torch.ones_like(label_gt), torch.zeros_like(label_gt))

            with torch.no_grad():
                pred_detach1 = pred1.detach()
                pred_detach2 = pred2.detach()
                pred_sim1 = (pred_detach1.view(-1, 1) - pred_detach1.view(1, -1)).float()
                pred_sim1 = torch.where(pred_sim1 == 0., torch.ones_like(pred_sim1), torch.zeros_like(pred_sim1))
                pred_sim2 = (pred_detach2.view(-1, 1) - pred_detach2.view(1, -1)).float()
                pred_sim2 = torch.where(pred_sim2 == 0., torch.ones_like(pred_sim2), torch.zeros_like(pred_sim2))

                feat_detach1 = feat1.detach()
                feat_sim1 = torch.zeros_like(pred_sim1).to(device)
                distance1 = torch.sum(torch.square(feat_detach1.unsqueeze(1) - feat_detach1), dim=2)
                _, idx_near1 = torch.topk(distance1, dim=-1, largest=False, k=args.top_k + 1)
                idx_near1 = idx_near1[:, 1:]
                feat_near1 = feat_detach1[idx_near1]
                distance1_ = torch.sum(torch.square(feat_near1.unsqueeze(2) - feat_detach1), dim=3)
                _, idx_near1_ = torch.topk(distance1_, dim=-1, largest=False, k=args.top_k + 1)
                idx_near1_ = idx_near1_[:, :, 1:]
                idx_new1 = torch.arange(feat_detach1.shape[0]).unsqueeze(-1).unsqueeze(-1).to(device)
                match1 = (idx_near1_ == idx_new1).sum(-1).float()
                feat_sim1 = feat_sim1.scatter_(1, idx_near1, match1)

                feat_detach2 = feat2.detach()
                feat_sim2 = torch.zeros_like(pred_sim2).to(device)
                distance2 = torch.sum(torch.square(feat_detach2.unsqueeze(1) - feat_detach2), dim=2)
                _, idx_near2 = torch.topk(distance2, dim=-1, largest=False, k=args.top_k + 1)
                idx_near2 = idx_near2[:, 1:]
                feat_near2 = feat_detach2[idx_near2]
                distance2_ = torch.sum(torch.square(feat_near2.unsqueeze(2) - feat_detach2), dim=3)
                _, idx_near2_ = torch.topk(distance2_, dim=-1, largest=False, k=args.top_k + 1)
                idx_near2_ = idx_near2_[:, :, 1:]
                idx_new2 = torch.arange(feat_detach2.shape[0]).unsqueeze(-1).unsqueeze(-1).to(device)
                match2 = (idx_near2_ == idx_new2).sum(-1).float()
                feat_sim2 = feat_sim2.scatter_(1, idx_near2, match2)

                feat_sim1 = feat_sim1 * pred_sim1
                feat_sim1[:labeled_len, :labeled_len] = label_gt.clone()
                feat_sim2 = feat_sim2 * pred_sim2
                feat_sim2[:labeled_len, :labeled_len] = label_gt.clone()

            unlabel_sim1 = torch.mm(prob1, prob1.t())
            unlabel_sim2 = torch.mm(prob2, prob2.t())
            unlabel_sim3 = torch.mm(prob1, prob2.t())
            pse_loss = torch.mean(torch.sum(-feat_sim1 * torch.log(F.sigmoid(unlabel_sim1)), dim=1)) \
                       + torch.mean(torch.sum(-feat_sim2 * torch.log(F.sigmoid(unlabel_sim2)), dim=1)) \
                       + torch.mean(torch.sum(-feat_sim1 * feat_sim2 * torch.log(F.sigmoid(unlabel_sim3)), dim=1))

            labeled_onehot = F.one_hot(train_targets)[index_s].float()
            unlabeled_onehot = F.one_hot(test_label)[index_t].float()
            unlabeled_mask = test_mask[index_t].float()

            output_ss1 = output_ss1 / torch.sum(output_ss1, dim=1, keepdim=True)
            output_ss2 = output_ss2 / torch.sum(output_ss2, dim=1, keepdim=True)

            output_st1 = output_st1 / torch.sum(output_st1, dim=1, keepdim=True)
            output_st2 = output_st2 / torch.sum(output_st2, dim=1, keepdim=True)

            output_tt1 = output_tt1 / torch.sum(output_tt1, dim=1, keepdim=True)
            output_tt2 = output_tt2 / torch.sum(output_tt2, dim=1, keepdim=True)

            ce_loss = -torch.mean(torch.sum(labeled_onehot * torch.log(torch.clamp(output_ss1, min=1e-8)), dim=1)) \
                      - torch.mean(torch.sum(labeled_onehot * torch.log(torch.clamp(output_ss2, min=1e-8)), dim=1)) \
                      - torch.mean(
                unlabeled_mask * torch.sum(unlabeled_onehot * torch.log(torch.clamp(output_tt1, min=1e-8)), dim=1)) \
                      - torch.mean(
                unlabeled_mask * torch.sum(unlabeled_onehot * torch.log(torch.clamp(output_tt2, min=1e-8)), dim=1))

            align_loss = torch.mean(
                torch.sum(-auxilarly_dis(output_st1) * torch.log(torch.clamp(output_st1, min=1e-8)), dim=1)) \
                         + torch.mean((1.0 - unlabeled_mask) * torch.sum(
                -auxilarly_dis(output_tt1) * torch.log(torch.clamp(output_tt1, min=1e-8)), dim=1)) \
                         + torch.mean(
                torch.sum(-auxilarly_dis(output_st2) * torch.log(torch.clamp(output_st2, min=1e-8)), dim=1)) \
                         + torch.mean((1.0 - unlabeled_mask) * torch.sum(
                -auxilarly_dis(output_tt2) * torch.log(torch.clamp(output_tt2, min=1e-8)), dim=1))

            label_sim1 = torch.mm(output_st1, output_st1.t())
            label_sim2 = torch.mm(output_st2, output_st2.t())
            label_sim3 = torch.mm(output_st1, output_st2.t())
            clu_loss = bce(label_sim1.view(-1, 1), label_gt.view(-1, 1)) + bce(label_sim2.view(-1, 1),
                                                                               label_gt.view(-1, 1)) \
                       + bce(label_sim3.view(-1, 1), label_gt.view(-1, 1))

            if epoch < args.pretrain:
                loss = recon_loss
            else:
                loss = recon_loss + ce_loss + align_loss + pse_loss + clu_loss

            recon_losses.update(recon_loss.item(), args.batch_size)
            ce_losses.update(ce_loss.item(), args.batch_size)
            align_losses.update(align_loss.item(), args.batch_size)
            pse_losses.update(pse_loss.item(), args.batch_size)
            clu_losses.update(clu_loss.item(), args.batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Training {}/{}, zinb loss: {:.4f}, ce loss: {:.4f}, pse loss: {:.4f}, "
              "align loss: {:.4f}, clu loss: {:.4f}".format(epoch, args.epochs, recon_losses.avg,
                                                            ce_losses.avg, pse_losses.avg, align_losses.avg,
                                                            clu_losses.avg))

    return [cluster_num, total_classes, shared_classes, ratio, mode, dropout_rate, dataseed,
           best_overall_acc, best_seen_acc, best_unseen_acc,
           best_overall_acc2, best_seen_acc2, best_unseen_acc2,
           final_overall_acc, final_seen_acc, final_unseen_acc,
            final_overall_acc2, final_seen_acc2, final_unseen_acc2]


def get_data_info(path):
    path_list = glob.glob(path + "/*")
    data_list = []
    for rec in path_list:
        data_info = rec.split("/")[-1]
        data_list.extend([data_info])
    return data_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scUDA')
    parser.add_argument('--random-seed', type=int, default=8888, metavar='S')
    parser.add_argument('--g', default='0', type=int)
    parser.add_argument('--removal', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--highly-genes', type=int, default=2000)
    parser.add_argument('--noi', type=float, default=0.3)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--deta', type=float, default=1.0)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--pretrain', type=int, default=600)
    parser.add_argument('--midtrain', type=int, default=1000)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--thres', type=float, default=5.0)
    parser.add_argument('--quantile', type=float, default=0.5)
    parser.add_argument('--structure', type=int, default=0)
    parser.add_argument("--rw", action='store_true', help='random walk or not')

    args = parser.parse_args()
    args.gpu_id = args.g
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu", args.gpu_id)

    data_list = get_data_info("../data/simulation_raw_data")
    result_list = []
    for i in range(len(data_list)):
        dataname = data_list[i]
        cluster_num = dataname.split("_")[3]
        ratio = dataname.split("_")[5]
        mode = dataname.split("_")[7]
        dropout_rate = dataname.split("_")[10]
        dataseed = dataname.split("_")[12]
        X, Y, batch_label = read_simu("../data/simulation_raw_data/" + dataname)
        result = main(args, X, Y, batch_label, device, cluster_num, ratio, mode, dropout_rate, dataseed)
        result_list.append(result)
        print(result_list)
        gc.collect()

    output = np.array(result_list)
    print(output)
    output = pd.DataFrame(output,
                          columns=["cluster number", "total classes", "shared classes", "batch ratio", "mode",
                                   "dropout rate", "data number",
                                   "overall1", "seen1", "novel1", "overall2", "seen2", "novel2",
                                   "foverall1", "fseen1", "fnovel1", "foverall2", "fseen2", "fnovel2"])
    print(output)






