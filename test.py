import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np

def test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)

        for i, input in enumerate(dataloader):
            # print(input.shape)
            input = input.to(device)
            input = input.permute(0,2,1,3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, logits = model(inputs=input)

            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)

            sig = logits
            #print((sig.shape))

            pred = torch.cat((pred, sig))

        #print(pred)
        gt = np.load(args.gt)
        #print(sum(gt))
        pred = list(pred.cpu().detach().numpy())

        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        return rec_auc

