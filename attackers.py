# Modified from https://github.com/Harry24k/PGD-pytorch/blob/master/PGD.ipynb 
from torch import nn
import torch


def pgd_attack(model, images, labels, eps=0.1, alpha=2 / 255, iters=5, device='cuda:0'):
    images = images.to(device)
    # print(images.shape,'image')
    labels = labels.long().to(device)
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.MSELoss()
    ori_images = images.data.to(device)
    new_ainput = torch.tensor([])
    for k in range(images.shape[0]):
        # print(k)
        # print('new:',new_ainput.shape)
        image = images[[k]]
        for i in range(iters):
            # print(image.shape)
            # image = image.view(1,10,32,2048)
            image.requires_grad = True
            output_abn = model(image)[0]
            output_nor = 1 - output_abn
            outputs = torch.cat((output_nor, output_abn), dim=1)
            model.zero_grad()
            # print('outputs',outputs.shape)
            # print('label',labels[[k]])
            cost1 = loss1(outputs, labels[[k]]).to(device)
            cost2 = loss2(image, ori_images[[k]]).to(device)
            cost = 0.9 * cost1 + 0.1 * cost2
            # print('cost:',cost1)
            # print('outputs',outputs)
            # print('labels',labels)
            cost.to(device)
            cost.backward()
            adv_images = image + alpha * image.grad.sign()
            eta = torch.clamp(adv_images - ori_images[[k]], min=-eps, max=eps)
            image = torch.clamp(ori_images[[k]] + eta, min=0, max=1).detach_()
        # print('image',image.shape)
        new_ainput = torch.cat((new_ainput, image), dim=0)

    return new_ainput


# Modified from https://github.com/Harry24k/FGSM-pytorch/blob/master/FGSM.ipynb
def fgsm_attack(model, images, labels, eps=0.1):
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    loss = nn.CrossEntropyLoss()
    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad = True

    output_abn = model(images)[0]
    output_nor = 1 - output_abn
    outputs = torch.cat((output_nor, output_abn), dim=1)

    model.zero_grad()
    # print(outputs.shape,labels.shape,'ii')
    cost = -loss(outputs, labels.long()).to(device)
    cost.backward()

    attack_images = images + eps * images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)

    return attack_images
