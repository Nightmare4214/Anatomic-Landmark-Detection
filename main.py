import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision
from packaging import version
from torch.utils.data import DataLoader
from torchvision.models import VGG19_BN_Weights
from tqdm import tqdm

import models
import train
from dataLoader import Rescale, ToTensor, LandmarksDataset
from lossFunction import fusionLossFunc_improved
from utils import setup_seed, seed_worker

# torch.set_num_threads(1)

g = torch.Generator()
g.manual_seed(0)
setup_seed(42)

plt.ion()  # interactive mode

parser = argparse.ArgumentParser()
parser.add_argument("--batchSize", type=int, default=1)
parser.add_argument("--landmarkNum", type=int, default=19)
parser.add_argument("--image_scale", default=(800, 640), type=tuple)
parser.add_argument("--use_gpu", type=int, default=0)
parser.add_argument("--spacing", type=float, default=0.1)
parser.add_argument("--R1", type=int, default=41)
parser.add_argument("--R2", type=int, default=41)
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--data_enhanceNum", type=int, default=1)
parser.add_argument("--stage", type=str, default="train")
parser.add_argument("--saveName", type=str, default="test1")
parser.add_argument("--testName", type=str, default="30cepha100_fusion_unsuper.pkl")
parser.add_argument("--dataRoot", type=str, default="/mnt/data/datasets/ceph")
parser.add_argument("--supervised_dataset_train", type=str, default="cepha/")
parser.add_argument("--supervised_dataset_test", type=str, default="cepha/")
parser.add_argument("--unsupervised_dataset", type=str, default="cepha/")
parser.add_argument("--trainingSetCsv", type=str, default="cepha_train.csv")
parser.add_argument("--testSetCsv", type=str, default="cepha_val.csv")
parser.add_argument("--unsupervisedCsv", type=str, default="cepha_val.csv")


def main():
    config = parser.parse_args()
    model_ft = models.fusionVGG19(torchvision.models.vgg19_bn(weights=VGG19_BN_Weights.DEFAULT), config).cuda(config.use_gpu)
    print("image scale ", config.image_scale)
    print("GPU: ", config.use_gpu)

    transform_origin = torchvision.transforms.Compose([
        Rescale(config.image_scale),
        ToTensor()
    ])

    train_dataset_origin = LandmarksDataset(csv_file=os.path.join(config.dataRoot, config.trainingSetCsv),
                                            root_dir=os.path.join(config.dataRoot, config.supervised_dataset_train),
                                            transform=transform_origin,
                                            landmarksNum=config.landmarkNum
                                            )

    val_dataset = LandmarksDataset(csv_file=os.path.join(config.dataRoot, config.testSetCsv),
                                   root_dir=os.path.join(config.dataRoot, config.supervised_dataset_test),
                                   transform=transform_origin,
                                   landmarksNum=config.landmarkNum
                                   )

    train_dataloader = []
    val_dataloader = []

    train_dataloader_t = DataLoader(train_dataset_origin, batch_size=config.batchSize,
                                    shuffle=True,
                                    pin_memory=False,
                                    # num_workers=40, worker_init_fn=seed_worker, generator=g
                                    )

    val_dataloader_t = DataLoader(val_dataset, batch_size=config.batchSize,
                                  shuffle=False,
                                  pin_memory=False,
                                  # num_workers=40, worker_init_fn=seed_worker, generator=g
                                  )

    # pre-load all data into memory for efficient training
    for data in tqdm(train_dataloader_t):
        train_dataloader.append(data)

    for data in tqdm(val_dataloader_t):
        val_dataloader.append(data)

    print(len(train_dataloader), len(val_dataloader))

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    para_list = list(model_ft.children())

    print("len", len(para_list))
    for idx in range(len(para_list)):
        print(idx, "-------------------->>>>", para_list[idx])

    # if use_gpu:
    model_ft = model_ft.cuda(config.use_gpu)
    if version.parse(torch.__version__) >= version.parse('2.0.0'):
        torch.set_float32_matmul_precision('high')
        model_ft = torch.compile(model_ft)
    criterion = fusionLossFunc_improved(config)

    optimizer_ft = optim.Adadelta(filter(lambda p: p.requires_grad,
                                         model_ft.parameters()), lr=1.0)

    train.train_model(model_ft, dataloaders, criterion, optimizer_ft, config)


if __name__ == "__main__":
    main()
