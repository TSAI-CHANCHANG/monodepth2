# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from layers import *
from kitti_utils import *
from utils import *
from pose_error import *
from options import MonodepthOptions
from datasets import SevenDataset
import networks

class Evaluation:
    def __init__(self, option):
        self.options = option
        self.ssim = SSIM()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        opt = self.options
        device = torch.device("cpu" if opt.no_cuda else "cuda")
        backproject_depth = {}
        project_3d = {}
        for scale in opt.scales:
            h = opt.height // (2 ** scale)
            w = opt.width // (2 ** scale)

            backproject_depth[scale] = BackprojectDepth(opt.batch_size, h, w)
            backproject_depth[scale].to(device)

            project_3d[scale] = Project3D(opt.batch_size, h, w)
            project_3d[scale].to(device)

        for scale in opt.scales:
            # disp = outputs[("disp", scale)]
            if opt.v1_multiscale:
                source_scale = scale
            else:
                # disp = F.interpolate(
                    # disp, [opt.height, opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            # _, depth = disp_to_depth(disp, opt.min_depth, opt.max_depth)

            depth = inputs["depth_gt"]
            
            for i, frame_id in enumerate(opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)
                print(T,inputs)
                cam_points = backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="zeros",align_corners=True)   
                if not opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target, depth_gt):
      """Computes reprojection loss between a batch of predicted and target images
      """
      print("pred.shape={}".format(pred.shape))
      print("target.shape={}".format(target.shape))
      print("depth_gt.shape={}".format(depth_gt.repeat(1,3,1,1).shape))
      #mask = (depth_gt.repeat(1,3,1,1) == 0)
      #depth = depth_gt[depth_gt > 0]
      #pred[mask] = target[mask]

      print("pred.shape={}".format(pred.shape))
      print("target.shape={}".format(target.shape))
      #print("depth.shape={}".format(depth.shape))
      abs_diff = torch.abs(target - pred)
      l1_loss = abs_diff.mean(1, True)

      ssim_loss = self.ssim(pred, target).mean(1, True)
      reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

      return reprojection_loss
    def evaluate(self):
        opt = self.options
        outputs = {}

        filenames = readlines(
            os.path.join(os.path.dirname(__file__), "splits", opt.split, "test_files.txt"))
        dataset = SevenDataset(opt.data_path, filenames, opt.height, opt.width,
                                opt.frame_ids, 1, is_train=False)
        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                                num_workers=opt.num_workers, pin_memory=True, drop_last=False)

        print("-> Computing pose predictions")
        reprojection_losses = []
        with torch.no_grad():
            for inputs in dataloader:
                for key, ipt in inputs.items():
                    inputs[key] = ipt.cuda()

                if opt.pose_model_type == "shared":
                    pose_feats = {f_i: features[f_i] for f_i in opt.frame_ids}
                else:
                    pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in opt.frame_ids}

                for f_i in opt.frame_ids[1:]:
                    print(f_i)
                    if f_i != "s":
                        # To maintain ordering we always pass frames in temporal order
                        fStr1 = "frame-{:06d}.pose.txt".format(inputs["index"].item())
                        gtPosePath1 = os.path.join("/content/drive/My Drive/monodepth2/splits/7scenes/chess/seq-01", "poses", fStr1)
                        gtPose1 = np.loadtxt(gtPosePath1).reshape(4, 4)
                        fStr2 = "frame-{:06d}.pose.txt".format(inputs["index"].item()+opt.frame_ids[1])
                        print("fStr1 = {}".format(fStr1))
                        print("fStr2 = {}".format(fStr2))
                        gtPosePath2 = os.path.join("/content/drive/My Drive/monodepth2/splits/7scenes/chess/seq-01", "poses", fStr2)
                        gtPose2 = np.loadtxt(gtPosePath2).reshape(4, 4)
                        gtRelativePose = calRelativePose(gtPose1, gtPose2)

                        outputs[("cam_T_cam", 0, f_i)] = torch.from_numpy(gtRelativePose.reshape(1, 4, 4).astype(np.float32)).cuda()


                self.generate_images_pred(inputs, outputs)
                pred = outputs[("color", opt.frame_ids[1], opt.scales[0])]
                target = inputs[("color", 0, opt.scales[0])]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target, inputs["depth_gt"]))
                img_2 = transforms.ToPILImage()(outputs[("color", opt.frame_ids[1], 0)].squeeze().cpu()).convert('RGB')
                img_2.save("/content/drive/My Drive/monodepth2/assets/generate_gt_{}to{}.jpg".format(opt.frame_ids[1],0)) 


        print("-> Predictions saved to")
        print(("/content/drive/My Drive/monodepth2/assets/generate_gt_{}to{}.jpg".format(opt.frame_ids[1],0)))

        


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluation = Evaluation(options.parse())
    evaluation.evaluate()
