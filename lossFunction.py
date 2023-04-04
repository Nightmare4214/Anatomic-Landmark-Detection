import argparse
import torch
import torch.nn as nn
import numpy as np
import utils


class fusionLossFunc_improved(nn.Module):
    def __init__(self, config):
        super(fusionLossFunc_improved, self).__init__()
        # .use_gpu, R1, R2, image_scale, batchSize, landmarkNum
        self.use_gpu = config.use_gpu
        self.R1 = config.R1
        self.width = config.image_scale[1]
        self.higth = config.image_scale[0]
        self.imageNum = config.batchSize
        self.landmarkNum = config.landmarkNum

        self.binaryLoss = nn.BCEWithLogitsLoss(reduction='mean').cuda(config.use_gpu)
        self.l1Loss = torch.nn.L1Loss().cuda(config.use_gpu)

        self.HeatMap = np.zeros((self.higth * 2, self.width * 2))
        # self.mask = np.zeros((self.higth * 2, self.width * 2))

        self.offsetMapX_groundTruth = torch.zeros(self.imageNum, self.landmarkNum, self.higth, self.width).cuda(
            self.use_gpu)
        self.offsetMapY_groundTruth = torch.zeros(self.imageNum, self.landmarkNum, self.higth, self.width).cuda(
            self.use_gpu)
        self.binary_class_groundTruth1 = torch.zeros(self.imageNum, self.landmarkNum, self.higth, self.width).cuda(
            self.use_gpu)
        # self.binary_class_groundTruth2 = torch.zeros(self.imageNum, self.landmarkNum, self.higth, self.width).cuda(
        #     self.use_gpu)
        # self.offsetMask = torch.zeros(self.imageNum, self.landmarkNum, self.higth, self.width).cuda(self.use_gpu)

        rr = config.R1
        dev = 4
        referPoint = (self.higth, self.width)
        for i in range(referPoint[0] - rr, referPoint[0] + rr + 1):
            for j in range(referPoint[1] - rr, referPoint[1] + rr + 1):
                temdis = utils.Mydist(referPoint, (i, j))
                if temdis <= rr:
                    self.HeatMap[i][j] = 1
        # rr = config.R2
        # referPoint = (self.higth, self.width)
        # for i in range(referPoint[0] - rr, referPoint[0] + rr + 1):
        #     for j in range(referPoint[1] - rr, referPoint[1] + rr + 1):
        #         temdis = utils.Mydist(referPoint, (i, j))
        #         if temdis <= rr:
        #             self.mask[i][j] = 1

        self.offsetMapx = torch.arange(2 * self.higth)[:, None].expand(-1, 2 * self.width)
        self.offsetMapx = referPoint[0] - self.offsetMapx
        self.offsetMapx = self.offsetMapx.cuda(self.use_gpu).float() / config.R2

        self.offsetMapy = torch.arange(2 * self.width)[None, :].expand(2 * self.higth, -1)
        self.offsetMapy = referPoint[1] - self.offsetMapy
        self.offsetMapy = self.offsetMapy.cuda(self.use_gpu).float() / config.R2

        self.HeatMap = torch.from_numpy(self.HeatMap).cuda(self.use_gpu).float()
        # self.mask = torch.from_numpy(self.mask).cuda(self.use_gpu).float()

        # self.zeroTensor = torch.zeros((self.imageNum, self.landmarkNum, self.higth, self.width)).cuda(self.use_gpu)

    # def getOffsetMask(self, h, w, X, Y):
    #     for imageId in range(self.imageNum):
    #         for landmarkId in range(self.landmarkNum):
    #             self.offsetMask[imageId, landmarkId, :, :] = self.mask[
    #                                                          h - X[imageId][landmarkId]: 2 * h - X[imageId][landmarkId],
    #                                                          w - Y[imageId][landmarkId]: 2 * w - Y[imageId][landmarkId]]
    #     return self.offsetMask

    def forward(self, featureMaps, landmarks):  # (B,57,800,640) (B, 19, 2)
        h, w = featureMaps.size()[2], featureMaps.size()[3]
        X = np.round((landmarks[:, :, 0] * (h - 1)).numpy()).astype("int")  # (B, 19)
        Y = np.round((landmarks[:, :, 1] * (w - 1)).numpy()).astype("int")  # (B, 19)
        binary_class_groundTruth = self.binary_class_groundTruth1

        # generate heatmap and offset for every landmark
        for imageId in range(self.imageNum):
            for landmarkId in range(self.landmarkNum):
                # ~ self.binary_class_groundTruth[imageId, landmarkId, :, :] = self.HeatMap[h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]]
                binary_class_groundTruth[imageId, landmarkId, :, :] = self.HeatMap[
                                                                      h - X[imageId][landmarkId]: 2 * h - X[imageId][
                                                                          landmarkId],
                                                                      w - Y[imageId][landmarkId]: 2 * w - Y[imageId][
                                                                          landmarkId]]
                self.offsetMapX_groundTruth[imageId, landmarkId, :, :] = self.offsetMapx[
                                                                         h - X[imageId][landmarkId]: 2 * h - X[imageId][
                                                                             landmarkId],
                                                                         w - Y[imageId][landmarkId]: 2 * w - Y[imageId][
                                                                             landmarkId]]
                self.offsetMapY_groundTruth[imageId, landmarkId, :, :] = self.offsetMapy[
                                                                         h - X[imageId][landmarkId]: 2 * h - X[imageId][
                                                                             landmarkId],
                                                                         w - Y[imageId][landmarkId]: 2 * w - Y[imageId][
                                                                             landmarkId]]
        # calculate loss and mean over landmark
        indexs = binary_class_groundTruth > 0
        temloss = [
            [2 * self.binaryLoss(featureMaps[imageId][landmarkId], binary_class_groundTruth[imageId][landmarkId]),
             self.l1Loss(featureMaps[imageId][landmarkId + self.landmarkNum * 1][indexs[imageId][landmarkId]],
                         self.offsetMapX_groundTruth[imageId][landmarkId][indexs[imageId][landmarkId]]),
             self.l1Loss(featureMaps[imageId][landmarkId + self.landmarkNum * 2][indexs[imageId][landmarkId]],
                         self.offsetMapY_groundTruth[imageId][landmarkId][indexs[imageId][landmarkId]])]

            for imageId in range(self.imageNum)
            for landmarkId in range(self.landmarkNum)]
        loss1 = (sum([sum(temloss[ind]) for ind in range(self.imageNum * self.landmarkNum)])) / (
                self.imageNum * self.landmarkNum)

        return loss1


if __name__ == '__main__':
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
    config = parser.parse_args()

    loss = fusionLossFunc_improved(config)
