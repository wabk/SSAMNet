import sys
sys.path.append('..')
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm
import torchmetrics
from dataset import MRNetDataset
from model import SSAMNet

def test(model, data_loader, device):
    model.eval()

    test_auc = torchmetrics.AUROC(task="binary").to(device)
    test_acc = torchmetrics.Accuracy(average="none", task="binary").to(device)
    test_sen = torchmetrics.Recall(average='none', task="binary").to(device)
    test_spe = torchmetrics.Specificity(average="none", task="binary").to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)

    with torch.no_grad():

        for i, (image, label, weight) in enumerate(data_loader):

            image = image.to(device)
            label = label.to(device)

            prediction0 = model(image.float())
            probas0 = torch.sigmoid(prediction0)

            test_auc.update(probas0, label)
            test_acc.update(probas0, label)
            test_sen.update(probas0, label)
            test_spe.update(probas0, label)

        test_AUC = np.round(test_auc.compute().item(), 4)
        test_accuracy = np.round(test_acc.compute().item(), 4)
        test_sensitivity = np.round(test_sen.compute().item(), 4)
        test_specificity = np.round(test_spe.compute().item(), 4)

        test_auc.reset()
        test_acc.reset()
        test_sen.reset()
        test_spe.reset()

    return test_AUC, test_accuracy, test_sensitivity, test_specificity

def predict(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    test_dataset = MRNetDataset(args.data_path, args.task, args.plane, train=False, transform=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    model = SSAMNet(cla_num=1, s_top=16, c_div=8).to(device)
    model_weight_path = "/home/ubuntu/acl_1/SFMCANet_t16/results/Mrnetdata/checkpoints/acl/sagittal/r1/ours_SFCANet_acl_sagittal_36.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    test_auc, test_accuracy, test_sensitivity, test_specificity = test(model=model,
                                                                    data_loader=test_loader,
                                                                    device=device
                                                                    )
    print("test_auc {0} |  test_accuracy {1} |  test_sensitivity {2} |  test_specificity {3} ".format(
            test_auc, test_accuracy, test_sensitivity, test_specificity))
    print('-' * 30)

def parse():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--data-path', type=str, default="/home/ubuntu/datasets/MRNet-v1.0/")
    parser.add_argument('-t', '--task', type=str, choices=['abnormal', 'acl', 'meniscus'], default="acl")
    parser.add_argument('-p', '--plane', type=str, choices=['sagittal', 'coronal', 'axial'], default="sagittal")
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    predict(args)
