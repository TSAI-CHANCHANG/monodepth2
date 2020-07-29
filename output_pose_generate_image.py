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
from options import MonodepthOptions
from datasets import SevenDataset
import networks

class Evaluation:
    def __init__(self, option):
        self.options = option

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
            disp = outputs[("disp", scale)]
            if opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [opt.height, opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, opt.min_depth, opt.max_depth)

            outputs[("depth", 0, scale)] = depth

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

                cam_points = backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")   
                
                if not opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def evaluate(self):
        opt = self.options
        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)


        filenames = readlines(
            os.path.join(os.path.dirname(__file__), "splits", opt.split, "test_files.txt"))

        dataset = SevenDataset(opt.data_path, filenames, opt.height, opt.width,
                                [0, 1], 4, is_train=False)
        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                                num_workers=opt.num_workers, pin_memory=True, drop_last=False)

        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
        pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

        pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
        pose_encoder.load_state_dict(torch.load(pose_encoder_path))

        pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
        pose_decoder.load_state_dict(torch.load(pose_decoder_path))

        encoder_dict = torch.load(encoder_path)
        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))
        
        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()
        pose_encoder.cuda()
        pose_encoder.eval()
        pose_decoder.cuda()
        pose_decoder.eval()

        pred_disps = []
        pred_poses = []

        print("-> Computing pose predictions")

        opt.frame_ids = [0, 1]  # pose network only takes two frames as input
        outputs = {}

        with torch.no_grad():
            for inputs in dataloader:
                for key, ipt in inputs.items():
                    inputs[key] = ipt.cuda()
                input_color = inputs[("color", 0, 0)]
                all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in opt.frame_ids], 1)

                features = [pose_encoder(all_color_aug)]
                axisangle, translation = pose_decoder(features)
                outputs = depth_decoder(encoder(input_color))
                
                pred_disp, _ = disp_to_depth(outputs[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)

                outputs[("axisangle", 0, 1)] = axisangle
                outputs[("translation", 0, 1)] = translation
                outputs[("cam_T_cam", 0, 1)] = transformation_from_parameters(axisangle[:, 0], translation[:, 0])
                pred_poses.append(transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())

                self.generate_images_pred(inputs, outputs)
                print(inputs[("color", 1, 0)].shape)
                print(inputs[("color", 1, 1)].shape)
                print(inputs[("color", 1, 2)].shape)
                print(inputs[("color", 1, 3)].shape)
                #picture = outputs[("color", 1, 1)].squeeze().cpu().view(480,640,3).numpy()
                img_2 = transforms.ToPILImage()(outputs[("color", 1, 0)].squeeze().cpu()).convert('RGB')
                img_2.save("/content/drive/My Drive/monodepth2/generate.jpg") 

        pred_disps = np.concatenate(pred_disps)
        pred_poses = np.concatenate(pred_poses)

        print("-> Predictions saved to")
        print(pred_poses)


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluation = Evaluation(options.parse())
    evaluation.evaluate()
