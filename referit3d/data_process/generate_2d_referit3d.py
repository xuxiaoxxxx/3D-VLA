#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('/data/xuxiaoxu/code/3dvg/3D-VLA')

from referit3d.data_process.in_out.arguments import parse_arguments
from referit3d.data_process.in_out.neural_net_oriented import load_scan_related_data, load_referential_data
from referit3d.data_process.in_out.neural_net_oriented import compute_auxiliary_data, trim_scans_per_referit3d_data
from referit3d.data_process.in_out.pt_datasets.listening_dataset import make_data_loaders


if __name__ == '__main__':

    # Parse arguments
    args = parse_arguments()

    args.path_3d = '/data/xuxiaoxu/dataset/scannet/scans'
    args.path_2d = '/data/xuxiaoxu/test/scannet_img_base'
    args.path_saved_proj_2d = '/data/xuxiaoxu/code/3dvg/3D-VLA/referit3d/data_process/test_pic'

    # Read the scan related information
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(args.scannet_file)

    # Read the linguistic data of ReferIt3D
    referit_data = load_referential_data(args, args.referit3D_file, scans_split)

    # Prepare data & compute auxiliary meta-information.
    all_scans_in_dict = trim_scans_per_referit3d_data(referit_data, all_scans_in_dict)
    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict, args)
    data_loaders = make_data_loaders(args, referit_data, vocab, class_to_idx, all_scans_in_dict, mean_rgb)

    splits = ['train']
    for split in splits:
        loader = data_loaders[split]
        for res in loader:
            object_ids = res

    print("Process the image over!")