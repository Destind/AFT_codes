
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from torch.utils.data import DataLoader
import torch
from dataset import Dataset
import option
from config import *

args = option.parser.parse_args()
from model import Model 

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
pretrain_pkl0 = 'ckpt_results/ucf/3/final/ucf-train3-bl-bce.pkl'
model0 = Model(2048, 16)
model0 = model0.to(device)
model0.load_state_dict(torch.load(pretrain_pkl0))

pretrain_pkl1 = 'ckpt_results/ucf/3/final/ucf-train3-bl-fl-3.pkl'
model1 = Model(2048, 16)
model1 = model1.to(device)
model1.load_state_dict(torch.load(pretrain_pkl1))

def atest(dataloader, model0, model1, args, device):
    gt = np.load(args.gt)
    with torch.no_grad():
        model0.eval()
        pred0 = torch.zeros(0)
        model1.eval()
        pred1 = torch.zeros(0)
        for i, input in enumerate(dataloader):
            # print(input.shape)
            input0 = input.to(device)
            input0 = input0.permute(0, 2, 1, 3)
            score_abnormal0, score_normal0, feat_select_abn0, feat_select_normal0, logits0 = model0(inputs=input0)
            logits0 = torch.squeeze(logits0, 1)
            logits0 = torch.mean(logits0, 0)
            sig0 = logits0
            pred0 = torch.cat((pred0, sig0))

            input1 = input.to(device)
            input1 = input1.permute(0, 2, 1, 3)
            score_abnormal1, score_normal1, feat_select_abn1, feat_select_normal1, logits1 = model1(inputs=input1)
            logits1 = torch.squeeze(logits1, 1)
            logits1 = torch.mean(logits1, 0)
            sig1 = logits1
            pred1 = torch.cat((pred1, sig1))

        pred2 = torch.cat((pred0, pred1), dim=1)
        thre5 = torch.ones_like(pred2)*0.35
        dist = abs(pred2-thre5)
        choose1 = torch.max(dist,dim=1)[1]
        choose1 = choose1.unsqueeze(dim=1)
        choose0 = 1-choose1
        choose = torch.cat((choose0,choose1),dim=1)
        pred3 = pred2*choose
        pred4 = torch.max(pred3,dim=1)[0]
        pred4 = list(pred4.cpu().detach().numpy())
        pred4 = np.repeat(np.array(pred4), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred4)
        rec_auc = auc(fpr, tpr)
        print('fusion : ' + str(rec_auc))

        return  rec_auc


test_loader = DataLoader(Dataset(args, test_mode=True),
                         batch_size=1, shuffle=False,  ####
                         num_workers=0, pin_memory=False)

auc = atest(test_loader, model0, model1, args, device)

