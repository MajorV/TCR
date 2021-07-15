import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt


class tcrNet(nn.Module):
    def __init__(self):
        super(tcrNet, self).__init__()
        self.cnv1 = nn.Conv2d(in_channels=1, out_channels=70, kernel_size=(20,40), stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(70)
        self.cnv2 = nn.Conv2d(in_channels=70, out_channels=50, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(50)
        self.cnv3 = nn.Conv2d(in_channels=50, out_channels=30, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(30)
        self.cnv4 = nn.Conv2d(in_channels=30, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.leakyrelu = nn.LeakyReLU(0.1)

        target_filters = np.load('data/target_filters.npy')
        clutter_filters = np.load('data/clutter_filters.npy')
        # print('targets', target_filters.shape)
        # print('clutter', clutter_filters.shape)
        qcf_filter = np.concatenate((clutter_filters[:, 0:20], target_filters[:, -50:]), axis=1)
        # qcf_filter = target_filters[:, -50:]
        qcf_filter = np.swapaxes(qcf_filter, 0, 1)
        qcf_filter = qcf_filter.reshape((-1, 40, 20))
        qcf_filter = np.expand_dims(qcf_filter, axis=1)
        qcf_filter = np.swapaxes(qcf_filter, 2, 3)
        # print('qcf', qcf_filter.shape)

        # img_array = qcf_filter[20,:,:].reshape((20,40))
        # f, ax = plt.subplots(figsize=(10, 10))
        # ax.imshow(img_array)
        # ax.set_title('ljk', fontsize=10)
        # plt.show()

        layer1 = torch.tensor(qcf_filter).float()
        self.cnv1.weight = nn.Parameter(layer1)
        self.cnv1.weight.requires_grad = False

    def forward(self, x):
        x = self.leakyrelu(self.bn1(self.cnv1(x)))
        x = self.leakyrelu(self.bn2(self.cnv2(x)))
        x = self.leakyrelu(self.bn3(self.cnv3(x)))
        x = self.cnv4(x)
        # x = x **2
        return x

class tcrNet_lite(nn.Module):
    def __init__(self):
        super(tcrNet_lite, self).__init__()
        self.cnv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(20,40), stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(12)
        self.cnv2 = nn.Conv2d(in_channels=12, out_channels=5, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(5)
        self.cnv3 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(5)
        self.cnv4 = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.leakyrelu = nn.LeakyReLU(0.1)

        target_filters = np.load('data/target_filters.npy')
        clutter_filters = np.load('data/clutter_filters.npy')
        # print('targets', target_filters.shape)
        # print('clutter', clutter_filters.shape)
        qcf_filter = np.concatenate((clutter_filters[:, 0:2], target_filters[:, -10:]), axis=1)
        # qcf_filter = target_filters[:, -50:]
        qcf_filter = np.swapaxes(qcf_filter, 0, 1)
        qcf_filter = qcf_filter.reshape((-1, 40, 20))
        qcf_filter = np.expand_dims(qcf_filter, axis=1)
        qcf_filter = np.swapaxes(qcf_filter, 2, 3)
        # print('qcf', qcf_filter.shape)

        # img_array = qcf_filter[20,:,:].reshape((20,40))
        # f, ax = plt.subplots(figsize=(10, 10))
        # ax.imshow(img_array)
        # ax.set_title('ljk', fontsize=10)
        # plt.show()

        layer1 = torch.tensor(qcf_filter).float()
        self.cnv1.weight = nn.Parameter(layer1)
        self.cnv1.weight.requires_grad = False

    def forward(self, x):
        x = self.leakyrelu(self.bn1(self.cnv1(x)))
        x = self.leakyrelu(self.bn2(self.cnv2(x)))
        x = self.leakyrelu(self.bn3(self.cnv3(x)))
        x = self.cnv4(x)
        # x = x **2
        return x

class tcrLoss(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, predictions,gt):
        ctx.save_for_backward(predictions, gt)
        sum = torch.sum(gt,dim=3)
        sum = torch.sum(sum,dim=2)
        clutter_idx = torch.where(sum == 0)[0]
        target_idx = torch.where(sum != 0)[0]

        # print('clutter',clutter_idx,'targets',target_idx)

        target_response = predictions[target_idx,:,:,:].squeeze()
        clutter_response = predictions[clutter_idx,:,:,:].squeeze()

        target_response = target_response **2
        clutter_response = clutter_response **2

        target_peak = target_response[:,8,18]  #corresponds to gaussian peak in gt, detect instead of hard code later
        clutter_energy = torch.sum(clutter_response,dim=2)
        clutter_energy = torch.sum(clutter_energy,dim=1)

        # print('peak',target_peak.shape,'clutter',clutter_energy)
        n1 = target_peak.shape[0]
        n2 = clutter_energy.shape[0]
        if n1 != 0:
            loss1 = torch.log(target_peak.sum()/n1)
            # loss1 = torch.log(target_peak).sum()/n1

        else:
            loss1 = 0

        if n2 != 0:
            loss2 = torch.log(clutter_energy.sum()/n2)
            # loss2 = torch.log(clutter_energy.sum())

        else:
            loss2 = 0

        loss = loss2 - loss1

        return loss

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        predictions, gt = ctx.saved_tensors
        grad_gt = None
        grad_input = None

        sum = torch.sum(gt, dim=3)
        sum = torch.sum(sum, dim=2)

        clutter_idx = torch.where(sum == 0)[0]
        target_idx = torch.where(sum != 0)[0]

        n_samples = predictions.shape[0]

        category = torch.zeros(n_samples)
        category[target_idx] = 1


        # print('clutter',clutter_idx,'targets',target_idx)

        target_response = predictions[target_idx, :, :, :].squeeze()
        clutter_response = predictions[clutter_idx, :, :, :].squeeze()

        n_targets = target_response.shape[0]
        n_clutter = clutter_response.shape[0]

        U = torch.zeros((17,37)).cuda()
        U[8,18] = 1

        target_peak_energy = torch.zeros(n_targets)
        target_offpeak_energy = torch.zeros(n_targets)
        for i in range (n_targets):
            tmp = U * target_response[i]
            tmp = torch.flatten(tmp).unsqueeze(0)
            target_peak_energy[i] = torch.mm(tmp, torch.transpose(tmp,0,1))

            tmp = (1 - U) * target_response[i]
            tmp = torch.flatten(tmp).unsqueeze(0)
            target_offpeak_energy[i] = torch.mm(tmp, torch.transpose(tmp, 0, 1))

        clutter_response = clutter_response ** 2

        clutter_energy = torch.sum(clutter_response, dim=2)
        clutter_energy = torch.sum(clutter_energy, dim=1)

        idx_clutter = 0
        idx_target = 0
        grad_input = predictions.clone()

        for i in range(n_samples):
            if category[i] == 0:
                grad_input[i,0,:,:] = predictions[i,0,:,:]/clutter_energy[idx_clutter]
                idx_clutter += 1
            else:
                tmp = predictions[i,0,:,:]
                UY = U * tmp
                BY = (1-U)* tmp
                grad_input[i, 0, :, :] = BY/target_offpeak_energy[idx_target] -UY/(target_peak_energy[idx_target] + 1e-5)
                idx_target += 1

        return grad_input, grad_gt

