import numpy as np
import os
import imageio
import cv2 as cv
import math
import skimage.transform as sktf
from torch.utils.data import Dataset
from functools import partial
from .utils import dataset_to_dataloader, max_io_workers

# the following will be shared on other datasets too if not, they should become part of the ListeningDataset
# maybe make SegmentedScanDataset with only static functions and then inherit.
from .utils import check_segmented_object_order, sample_scan_object, pad_samples, objects_bboxes
from .utils import instance_labels_of_context, mean_rgb_unit_norm_transform


def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic

class ListeningDataset(Dataset):
    def __init__(self, references, scans, vocab, max_seq_len, points_per_object, max_distractors,
                 class_to_idx=None, object_transformation=None,
                 visualization=False, path_3d=None, path_2d=None, path_saved_proj_2d=None):

        self.references = references
        self.scans = scans
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.points_per_object = points_per_object
        self.max_distractors = max_distractors
        self.max_context_size = self.max_distractors + 1  # to account for the target.
        self.class_to_idx = class_to_idx
        self.visualization = visualization
        self.object_transformation = object_transformation

        self.path_2d = path_2d
        self.save_pic = path_saved_proj_2d
        
        ########### judge the obj_id is already exited #########
        # path = '/data/xuxiaoxu/dataset/scannet/scans'
        path = path_3d
        l = os.listdir(path)
        l.sort()
        self.ref_list = {}
        for name in l:
            self.ref_list[name] = []

        ##########  matrix ##################
        fx = 577.870605
        fy = 577.870605
        mx = 319.5
        my = 239.5
        intricsic = np.eye(4)
        intricsic[0][0] = fx
        intricsic[1][1] = fy
        intricsic[0][2] = mx
        intricsic[1][2] = my

        self.imageDim = [640, 480]
        self.intricsic = adjust_intrinsic(intricsic, intrinsic_image_dim=[640, 480], image_dim=self.imageDim)

        if not check_segmented_object_order(scans):
            raise ValueError

    def __len__(self):
        return len(self.references)

    def get_reference_data(self, index):
        ref = self.references.loc[index]
        scan = self.scans[ref['scan_id']]
        target = scan.three_d_objects[ref['target_id']]
        tokens = np.array(self.vocab.encode(ref['tokens'], self.max_seq_len), dtype=np.long)
        is_nr3d = ref['dataset'] == 'nr3d'

        return scan, target, tokens, is_nr3d

    def prepare_distractors(self, scan, target):
        target_label = target.instance_label

        # First add all objects with the same instance-label as the target
        distractors = [o for o in scan.three_d_objects if
                       (o.instance_label == target_label and (o != target))]

        # Then all more objects up to max-number of distractors
        already_included = {target_label}
        clutter = [o for o in scan.three_d_objects if o.instance_label not in already_included]
        np.random.shuffle(clutter)

        distractors.extend(clutter)
        distractors = distractors[:self.max_distractors]
        np.random.shuffle(distractors)

        return distractors

    def __getitem__(self, index):
        res = dict()
        scan, target, tokens, is_nr3d = self.get_reference_data(index)

        target_id = target.object_id
        scan_id = scan.scan_id
        # path_2d_base = '/data/xuxiaoxu/dataset/scannet/scannet_frames_25k'
        path_2d_base = self.path_2d
        # save_pic = '/data/xuxiaoxu/code/3dvg/referit3d_2d_data/test_pic'
        save_pic = self.save_pic
        # import pdb; pdb.set_trace()
        if target_id not in self.ref_list[scan_id]: # check the object whether has obtained

            if not os.path.exists(os.path.join(save_pic, scan_id)):
                os.makedirs(os.path.join(save_pic, scan_id))

            coords = target.get_pc()

            # pic load
            path_2D = os.path.join(path_2d_base, scan_id, 'color')
            pic_name = os.listdir(path_2D)
            pic_name.sort()

            path_pose = os.path.join(path_2d_base, scan_id, 'pose')
            pose_name = os.listdir(path_pose)
            pose_name.sort()

            path_depth = os.path.join(path_2d_base, scan_id, 'depth')
            depth_name = os.listdir(path_depth)
            depth_name.sort()

            # path_label = os.path.join(path_2d_base, scan_id, 'label')
            # label_name = os.listdir(path_label)
            # label_name.sort()

            l = len(pose_name)
            Max_points = 0
            Max_points_index = -1
            Max_points_x_min = 0
            Max_points_x_max = 0
            Max_points_y_min = 0
            Max_points_y_max = 0
            proj_2d_Max = None

            for i in range(l):
                posePath = os.path.join(path_pose, pose_name[i])
                depthPath = os.path.join(path_depth, depth_name[i])
                depth = imageio.imread(depthPath) / 1000.0

                pose = np.asarray(
                    [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
                    (x.split(" ") for x in open(posePath).read().splitlines())]
                )

                link = np.zeros((3, coords.shape[0]))
                coordsNew = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T

                assert coordsNew.shape[0] == 4, "[!] Shape error"

                world_to_camera = np.linalg.inv(pose)
                p = np.matmul(world_to_camera, coordsNew)
                p[0] = (p[0] * self.intricsic[0][0]) / p[2] + self.intricsic[0][2]
                p[1] = (p[1] * self.intricsic[1][1]) / p[2] + self.intricsic[1][2]
                pi = np.round(p).astype(np.int32)
                inside_mask = (pi[0] >= 0) * (pi[1] >= 0) \
                            * (pi[0] < self.imageDim[0]) * (pi[1] < self.imageDim[1])
                occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]]
                                        - p[2][inside_mask]) <= 0.05
                inside_mask[inside_mask == True] = occlusion_mask
                link[0][inside_mask] = pi[1][inside_mask]
                link[1][inside_mask] = pi[0][inside_mask]
                link[2][inside_mask] = 1
                link = link[:, link[2, :] != 0].T

                proj_2d = link[:, :2].astype(np.int32)
                proj_2d = proj_2d[:, ::-1]

                # save the crop pic
                proj_points_num = proj_2d.shape[0]
                if proj_points_num > 0:
                    bound_x = int(self.imageDim[0] * 0.1)
                    bound_y = int(self.imageDim[1] * 0.1)
                    x_min = max(min(proj_2d[:, 0]) - bound_x, 0)
                    x_max = min(max(proj_2d[:, 0]) + bound_x, self.imageDim[0])
                    y_min = max(min(proj_2d[:, 1]) - bound_y, 0)
                    y_max = min(max(proj_2d[:, 1]) + bound_y, self.imageDim[1])

                    if proj_points_num > Max_points:
                        Max_points = proj_points_num
                        Max_points_index = i
                        Max_points_x_min = x_min
                        Max_points_x_max = x_max
                        Max_points_y_min = y_min
                        Max_points_y_max = y_max
                        proj_2d_Max = proj_2d

            pic_max_path = os.path.join(path_2D, pic_name[Max_points_index])

            img = imageio.imread(pic_max_path)
            img = img[:, :, ::-1].astype(np.uint8).copy()
            img = sktf.resize(img, [480, 640], order=0, preserve_range=True)
            # import pdb; pdb.set_trace()
            crop_img = img[Max_points_y_min:Max_points_y_max, Max_points_x_min:Max_points_x_max]
            print("scan_id:", scan_id, target_id, target.instance_label, pic_max_path, crop_img.shape)
            save_path = os.path.join(os.path.join(save_pic, scan_id), str(target_id) + '.jpg')

            # for i in range(proj_2d_Max.shape[0]):
            #     cv.circle(img, (int(proj_2d_Max[i][0]), int(proj_2d_Max[i][1])), 2, (0, 255, 0), 2)
            cv.imwrite(save_path, crop_img)
            self.ref_list[scan_id].append(target_id)
            

        return res


def make_data_loaders(args, referit_data, vocab, class_to_idx, scans, mean_rgb):
    n_workers = args.n_workers
    if n_workers == -1:
        n_workers = max_io_workers()

    data_loaders = dict()
    is_train = referit_data['is_train']
    splits = ['train', 'test']

    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb,
                                    unit_norm=args.unit_sphere_norm)
    for split in splits:
        mask = is_train if split == 'train' else ~is_train
        d_set = referit_data[mask]
        d_set.reset_index(drop=True, inplace=True)

        max_distractors = args.max_distractors if split == 'train' else args.max_test_objects - 1
        ## this is a silly small bug -> not the minus-1.

        dataset = ListeningDataset(references=d_set,
                                   scans=scans,
                                   vocab=vocab,
                                   max_seq_len=args.max_seq_len,
                                   points_per_object=args.points_per_object,
                                   max_distractors=max_distractors,
                                   class_to_idx=class_to_idx,
                                   object_transformation=object_transformation,
                                   visualization=args.mode == 'evaluate',
                                   path_3d=args.path_3d,
                                   path_2d=args.path_2d,
                                   path_saved_proj_2d=args.path_saved_proj_2d)

        seed = None
        data_loaders[split] = dataset_to_dataloader(dataset, split, args.batch_size, n_workers, seed=seed)

    return data_loaders
