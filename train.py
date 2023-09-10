import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss
from attackers import pgd_attack


def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return lamda1*loss


def l1_penalty(var):
    return torch.mean(torch.norm(var, dim=0))

class FocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input0, input1, target0, target1):
        # print('input', input0)
        loss0 = - (1 - self.alpha) * (input0 ** self.gamma) * (1 - target0) * torch.log(1 - input0)
        loss1 = - self.alpha * ((1 - input1) ** self.gamma) * target1 * torch.log(input1)
        loss = loss0.sum() + loss1.sum()
        # print(loss0)
        print('loss0 ', loss0.sum(), 'loss1 ', loss1.sum())

        return loss

class FocalLoss2(nn.Module):

    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss2, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        loss = -self.alpha * (1 - input) ** self.gamma * (target * torch.log(input)) - \
               (1 - self.alpha) * input ** self.gamma * ((1 - target) * torch.log(1 - input))
        return loss.mean()


class SigmoidMAELoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)


class SigmoidCrossEntropyLoss(torch.nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(- torch.abs(x))
        return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))


class All_loss(torch.nn.Module):
    def __init__(self, alpha, margin):
        super(All_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.sigmoid = torch.nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.criterion = torch.nn.BCELoss()
        self.FocalLoss = FocalLoss2()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a, viz):
        label = torch.cat((nlabel, alabel), 0)
        score_abnormal = score_abnormal
        score_normal = score_normal
        #print(score_normal.shape,'score_normal')
        #print(score_abnormal.shape, 'score_abnormal')

        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze()

        label = label.cuda()

        loss_cls = self.criterion(score, label)  # BCE loss in the score space
        #loss_cls = self.FocalLoss(score, label)

        loss_abn = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1))
        loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)

        loss_um = torch.mean((loss_abn + loss_nor) ** 2)

        loss_total = loss_cls + self.alpha * loss_um

        # viz.plot_lines('magnitude loss', (self.alpha * loss_um).item())
        # viz.plot_lines('classification loss', (loss_cls).item())

        return loss_total



# def train(nloader, aloader, model, batch_size, optimizer, viz, device):
#     with torch.set_grad_enabled(True):
#         focalloss = FocalLoss()
#         model.train()
#
#         ninput, nlabel = next(nloader)
#         ainput, alabel = next(aloader)
#         nlabel = nlabel.cuda()
#         alabel = alabel.cuda()
#
#         input = torch.cat((ninput, ainput), 0).to(device)
#
#         score_abnormal2, score_normal2, feat_select_abn2, feat_select_normal2, scores2 = model(input, advbatch=False)
#         #label = torch.cat((nlabel, alabel), 0)
#         #score = torch.cat((score_normal2, score_abnormal2), 0)
#         #score = score.squeeze()
#        # print(score)
#         #label = label.cuda()
#         focal_loss = focalloss(score_normal2.squeeze(), score_abnormal2.squeeze(), nlabel, alabel)
#         print('fl: ',focal_loss)
#         optimizer.zero_grad()
#         focal_loss.backward()
#         optimizer.step()



def train(nloader, aloader, model, batch_size, optimizer, viz, device):
    with torch.set_grad_enabled(True):

        model.train()

        ninput, nlabel = next(nloader)
        ainput, alabel = next(aloader)


        for k in range(3):
            # 生成对抗样本
            adversarialfeature = pgd_attack(model, ainput, alabel)
            #print(adversarialfeature.shape)


            # 干净样本与对抗样本共同测试＋微调
            input = torch.cat((ninput, ainput), 0).to(device)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, scores = model(input,advbatch=False)  # b*32  x 2048
            scores = scores.view(batch_size * 32 * 2, -1)
            scores = scores.squeeze()
            abn_scores = scores[batch_size * 32:]  # uncomment this if you apply sparse to abnormal score only
            nlabel = nlabel[0:batch_size]
            alabel = alabel[0:batch_size]
            loss_criterion = All_loss(0.0001, 100)
            loss_criterion_clean = loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal,
                                                  feat_select_abn, viz)
            loss_sparse_clean = sparsity(abn_scores, batch_size, 8e-3)
            loss_smooth_clean = smooth(abn_scores, 8e-4)
            cost_clean = loss_criterion_clean + loss_smooth_clean + loss_sparse_clean

            input_adv = torch.cat((ninput, adversarialfeature), 0).to(device)
            score_abnormal_adv, score_normal_adv, feat_select_abn_adv, feat_select_normal_adv, scores_adv = model(
                input_adv,advbatch=True)  # b*32  x 2048
            scores_adv = scores_adv.view(batch_size * 32 * 2, -1)
            scores_adv = scores_adv.squeeze()
            abn_scores_adv = scores_adv[batch_size * 32:]  # uncomment this if you apply sparse to abnormal score only
            loss_criterion_adv = loss_criterion(score_normal, score_abnormal_adv, nlabel, alabel, feat_select_normal,
                                                feat_select_abn_adv, viz)
            loss_sparse_adv = sparsity(abn_scores_adv, batch_size, 8e-3)
            loss_smooth_adv = smooth(abn_scores_adv, 8e-4)
            cost_adv = loss_criterion_adv + loss_smooth_adv + loss_sparse_adv

            cost = 0.7 * cost_clean + 0.3 * cost_adv



           # print('loss:', cost)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # adversarialfeature = adversarialfeature.cpu().numpy()
            # for i in range(batch_size):
            #     np.save(asave[i], adversarialfeature[i])
            #
            #




# def train(nloader, aloader, model, batch_size, optimizer, viz, device):
#     with torch.set_grad_enabled(True):
#         criterion = nn.CrossEntropyLoss()
#         model.train()
#
#         ninput, nlabel = next(nloader)
#         ainput, alabel = next(aloader)
#
#         input = torch.cat((ninput, ainput), 0).to(device)
#         score_abnormal, score_normal, feat_select_abn, feat_select_normal, scores = model(input,advbatch=False)  # b*32  x 2048
#         scores = scores.view(batch_size * 32 * 2, -1)
#         scores = scores.squeeze()
#         abn_scores = scores[batch_size * 32:]  # uncomment this if you apply sparse to abnormal score only
#         nlabel = nlabel[0:batch_size]
#         alabel = alabel[0:batch_size]
#         loss_criterion = All_loss(0.0001, 100)
#         loss_criterion_clean = loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal,
#                                               feat_select_abn, viz)
#         loss_sparse_clean = sparsity(abn_scores, batch_size, 8e-3)
#         loss_smooth_clean = smooth(abn_scores, 8e-4)
#         cost = loss_criterion_clean + loss_smooth_clean + loss_sparse_clean
#         #print('loss all',loss_criterion_clean,loss_smooth_clean,loss_sparse_clean)
#
#         print('loss:', cost)
#         optimizer.zero_grad()
#         cost.backward()
#         optimizer.step()
