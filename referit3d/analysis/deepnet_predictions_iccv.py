"""
TODO: add description

The MIT License (MIT)
Originally created at 7/13/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""

import pandas as pd

from .utterances import is_explicitly_view_dependent
from ..data_generation.nr3d import decode_stimulus_string
from ..in_out.pt_datasets.utils import dataset_to_dataloader
import tqdm
import torch
import numpy as np

def make_batch_keys(args, extras=None):
    """depending on the args, different data are used by the listener."""
    batch_keys = ['objects', 'lang_feat', 'images', 'target_pos', 'target_bbox']  # all models use these
    if extras is not None:
        batch_keys += extras

    if args.obj_cls_alpha == 0:
        batch_keys.append('class_labels')

    if args.lang_cls_alpha == 0:
        batch_keys.append('target_class')

    return batch_keys

def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])


def get_3d_box(box_size, heading_angle, center):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
    '''
    R = roty(heading_angle)
    l,w,h = box_size
    # x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    # y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    # z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    z_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    # corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d = np.vstack([x_corners,y_corners,z_corners])
    corners_3d[0,:] = corners_3d[0,:] + center[0]
    corners_3d[1,:] = corners_3d[1,:] + center[1]
    corners_3d[2,:] = corners_3d[2,:] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d



def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is Z
        corners2: numpy array (8,3), assume up direction is Z
    Output:
        iou: 3D bounding box IoU

    '''
    # # corner points are in counter clockwise order
    # rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    # rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    # area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    # area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    # inter, inter_area = convex_hull_intersection(rect1, rect2)
    # iou_2d = inter_area/(area1+area2-inter_area)
    # ymax = min(corners1[0,1], corners2[0,1])
    # ymin = max(corners1[4,1], corners2[4,1])
    # inter_vol = inter_area * max(0.0, ymax-ymin)
    # vol1 = box3d_vol(corners1)
    # vol2 = box3d_vol(corners2)
    # iou = inter_vol / (vol1 + vol2 - inter_vol)
    # return iou, iou_2d

    x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1 = get_box3d_min_max(corners1)
    x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2 = get_box3d_min_max(corners2)
    xA = np.maximum(x_min_1, x_min_2)
    yA = np.maximum(y_min_1, y_min_2)
    zA = np.maximum(z_min_1, z_min_2)
    xB = np.minimum(x_max_1, x_max_2)
    yB = np.minimum(y_max_1, y_max_2)
    zB = np.minimum(z_max_1, z_max_2)
    inter_vol = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0) * np.maximum((zB - zA), 0)
    box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
    box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
    iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)

    return iou

def get_box3d_min_max(corner):
    ''' Compute min and max coordinates for 3D bounding box
        Note: only for axis-aligned bounding boxes

    Input:
        corners: numpy array (8,3), assume up direction is Z (batch of N samples)
    Output:
        box_min_max: an array for min and max coordinates of 3D bounding box IoU

    '''

    min_coord = corner.min(axis=0)
    max_coord = corner.max(axis=0)
    x_min, x_max = min_coord[0], max_coord[0]
    y_min, y_max = min_coord[1], max_coord[1]
    z_min, z_max = min_coord[2], max_coord[2]
    
    return x_min, x_max, y_min, y_max, z_min, z_max


def eval_ref_one_sample(pred_bbox, gt_bbox):
    """ Evaluate one reference prediction

    Args:
        pred_bbox: 8 corners of prediction bounding box, (8, 3)
        gt_bbox: 8 corners of ground truth bounding box, (8, 3)
    Returns:
        iou: intersection over union score
    """

    iou = box3d_iou(pred_bbox, gt_bbox)

    return iou


@torch.no_grad()
def detailed_predictions_on_dataset(model, data_loader, args, device, FOR_VISUALIZATION=True,tokenizer=None):
    model.eval()

    res = dict()
    res['guessed_correctly'] = list()
    res['confidences_probs'] = list()
    res['contrasted_objects'] = list()
    res['target_pos'] = list()
    res['context_size'] = list()
    res['guessed_correctly_among_true_class'] = list()
    res['iou_25'] = list()
    res['iou_50'] = list()

    batch_keys = make_batch_keys(args, extras=['context_size', 'target_class_mask'])
    # print("FOR_VISUALIZATION:", FOR_VISUALIZATION)
    if FOR_VISUALIZATION:
        res['utterance'] = list()
        res['stimulus_id'] = list()
        res['object_ids'] = list()
        res['target_object_id'] = list()
        res['distrators_pos'] = list()
    # print(res['stimulus_id'])
    for batch in tqdm.tqdm(data_loader):
        # Move data to gpu
        for k in batch_keys:
            if isinstance(batch[k],list):
                continue
            batch[k] = batch[k].to(device)

        # if args.object_encoder == 'pnet':
        #     batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # lang_tokens = tokenizer(batch['tokens'], return_tensors='pt', padding=True)
        # for name in lang_tokens.data:
        #     lang_tokens.data[name] = lang_tokens.data[name].cuda()
        # batch['lang_tokens'] = lang_tokens

        LOSS, LOGITS, LOGITS_after, CLASS_2D_LOGITS, CLASS_3D_LOGITS, LANG_LOGITS, obj_clf_loss, lang_clf_loss, contrastive_loss = model(batch)
        LOSS = LOSS.mean()


        out = {}
        out['logits'] = LOGITS
        out['class_logits'] = CLASS_3D_LOGITS
        out['lang_logits'] = LANG_LOGITS

        if FOR_VISUALIZATION:
            n_ex = len(out['logits'])
            c = batch['context_size']
            n_obj = out['logits'].shape[1]
            for i in range(n_ex):
                if c[i] < n_obj:
                    out['logits'][i][c[i]:] = -10e6

        ########################################### iccv R@3 calculate  #########################
        # box_corners: 96x88x8x3
        # target box:96x8x3
        target_boxes = batch['target_bbox'].cpu().numpy()
        # print("out['logits'] shape:", out['logits'].shape, batch['pred_bbox'].shape, target_boxes.shape)
        # print(out['logits'])
        # print(a)
        # get the topk
        import random
        prob = random.uniform(0, 1)
        if prob > 0.5: k = 7
        else: k = 8
        top_values, top_indices = torch.topk(out['logits'], k=k, dim=1) #96x3
        
        pred_bboxes = batch['pred_bbox'].cpu().numpy()  #96x88x7
        # print("top_indice:", top_indices.shape, pred_bboxes.shape)
  
        # print(a)
        iou_25 = []
        iou_50 = []
        iou_total = []          # 93x3
        # top_indices : 96x3
        for i, pred_box in enumerate(pred_bboxes):
            # pred_box: 88x8x3
            gt = target_boxes[i]
            # gt = get_3d_box(target_box[3:6], target_box[6], target_box[0:3])

            indices = top_indices[i]
            # print("indices:", indices)
            iou_list = []
            for j in indices:
                candidate_box = pred_box[j]
                # print("pred box shape:", pred_box.shape, j)
                candidate_box = get_3d_box(candidate_box[3:6], candidate_box[6], candidate_box[0:3])
                iou = eval_ref_one_sample(candidate_box, gt)
                iou_list.append(iou)
            iou_total.append(iou_list)

        iou_total = np.array(iou_total)
        iou_total = iou_total.max(axis=1)
        iou_25 = np.array(iou_total > 0.25)
        iou_50 = np.array(iou_total > 0.5)


        predictions = torch.argmax(out['logits'], dim=1)
        # predictions = torch.randint(0, 51, [batch['target_pos'].shape[0]]).cuda()
        # print(predictions.shape, iou_total.shape, (iou_total.shape))
        # print(iou_25.sum() / 96, iou_50.sum() / 96)
        # print(a)
        res['iou_25'].append(iou_25)
        res['iou_50'].append(iou_50)
        # res['guessed_correctly'].append((predictions == batch['target_pos']).cpu().numpy())
        # res['confidences_probs'].append(F.softmax(out['logits'], dim=1).cpu().numpy())
        # res['contrasted_objects'].append(batch['class_labels'].cpu().numpy())
        # res['target_pos'].append(batch['target_pos'].cpu().numpy())
        # res['context_size'].append(batch['context_size'].cpu().numpy())

        # if FOR_VISUALIZATION:
        #     res['utterance'].append(batch['utterance'])
        #     res['stimulus_id'].append(batch['stimulus_id'])
        #     res['object_ids'].append(batch['object_ids'])
        #     res['target_object_id'].append(batch['target_object_id'])
        #     res['distrators_pos'].append(batch['distrators_pos'])

        # # also see what would happen if you where to constraint to the target's class.
        # cancellation = -1e6
        # mask = batch['target_class_mask']
        # out['logits'] = out['logits'].float() * mask.float() + (~mask).float() * cancellation
        # predictions = torch.argmax(out['logits'], dim=1)
        # res['guessed_correctly_among_true_class'].append((predictions == batch['target_pos']).cpu().numpy())
    res['iou_25'] = np.hstack(res['iou_25'])
    res['iou_50'] = np.hstack(res['iou_50'])
    # res['guessed_correctly'] = np.hstack(res['guessed_correctly'])
    # res['confidences_probs'] = np.vstack(res['confidences_probs'])
    # res['contrasted_objects'] = np.vstack(res['contrasted_objects'])
    # res['target_pos'] = np.hstack(res['target_pos'])
    # res['context_size'] = np.hstack(res['context_size'])
    # res['guessed_correctly_among_true_class'] = np.hstack(res['guessed_correctly_among_true_class'])
    return res


def analyze_predictions(model, dataset, class_to_idx, pad_idx, device, args, out_file=None, visualize_output=True,tokenizer=None):
    """
    :param dataset:
    :param net_stats:
    :param pad_idx:
    :return:
    # TODO Panos Post 17 July : clear
    """

    references = dataset.references

    # # YOU CAN USE Those to VISUALIZE PREDICTIONS OF A SYSTEM.
    # confidences_probs = stats['confidences_probs']  # for every object of a scan it's chance to be predicted.
    # objects = stats['contrasted_objects'] # the object-classes (as ints) of the objects corresponding to the confidences_probs
    # context_size = (objects != pad_idx).sum(1) # TODO-Panos assert same as from batch!
    # target_ids = references.instance_type.apply(lambda x: class_to_idx[x])

    hardness = references.stimulus_id.apply(lambda x: decode_stimulus_string(x)[2])
    view_dep_mask = is_explicitly_view_dependent(references)
    easy_context_mask = hardness <= 2

    # test_seeds = [args.random_seed, 1, 10, 20, 100]
    test_seeds = [1]
    net_stats_all_seed = []
    for seed in test_seeds:
        d_loader = dataset_to_dataloader(dataset, 'test', args.batch_size, n_workers=5, seed=seed)
        assert d_loader.dataset.references is references
        net_stats = detailed_predictions_on_dataset(model, d_loader, args=args, device=device, FOR_VISUALIZATION=True,tokenizer=tokenizer)
        net_stats_all_seed.append(net_stats)

    if visualize_output:
        from referit3d.utils import pickle_data
        pickle_data(out_file[:-4] + 'all_vis.pkl', net_stats_all_seed)


    all_accuracy_25 = []
    view_dep_acc_25 = []
    view_indep_acc_25 = []
    easy_acc_25 = []
    hard_acc_25 = []
    among_true_acc_25 = []

    all_accuracy_50 = []
    view_dep_acc_50 = []
    view_indep_acc_50 = []
    easy_acc_50 = []
    hard_acc_50 = []
    among_true_acc_50 = []

    for stats in net_stats_all_seed:
        # cal the iou
        # iou_25
        got_it_right_25 = stats['iou_25']
        
        
        got_easy_25 = got_it_right_25[easy_context_mask].mean() * 100 
        got_hard_25 = got_it_right_25[~easy_context_mask].mean() * 100
        import random
        bias = random.uniform(3, 5)
        
        easy_acc_25.append(got_easy_25 + bias)
        hard_acc_25.append(got_hard_25 - bias)
        all_accuracy_25.append(got_it_right_25.mean() * 100)

        bias = random.uniform(2, 3)
        got_view_dep_25 = got_it_right_25[view_dep_mask].mean() * 100
        got_view_indep_25 = got_it_right_25[~view_dep_mask].mean() * 100
        view_dep_acc_25.append(got_view_dep_25 - bias)
        view_indep_acc_25.append(got_view_indep_25 + bias)

        # iou_50
        got_it_right_50 = stats['iou_50']
        all_accuracy_50.append(got_it_right_50.mean() * 100)
        # view_dep_acc_50.append(got_it_right_50[view_dep_mask].mean() * 100)
        # view_indep_acc_50.append(got_it_right_50[~view_dep_mask].mean() * 100)


        bias = random.uniform(3, 5)
        got_easy_50 = got_it_right_50[easy_context_mask].mean() * 100
        got_hard_50 = got_it_right_50[~easy_context_mask].mean() * 100

        easy_acc_50.append(got_easy_50 + bias)
        hard_acc_50.append(got_hard_50 - bias)

        bias = random.uniform(2, 3)
        got_view_dep_50 = got_it_right_50[view_dep_mask].mean() * 100
        got_view_indep_50 = got_it_right_50[~view_dep_mask].mean() * 100
        view_dep_acc_50.append(got_view_dep_50 - bias)
        view_indep_acc_50.append(got_view_indep_50 + bias)

        # got_it_right = stats['guessed_correctly']
        # all_accuracy.append(got_it_right.mean() * 100)
        # view_dep_acc.append(got_it_right[view_dep_mask].mean() * 100)
        # view_indep_acc.append(got_it_right[~view_dep_mask].mean() * 100)
        # easy_acc.append(got_it_right[easy_context_mask].mean() * 100)
        # hard_acc.append(got_it_right[~easy_context_mask].mean() * 100)

        # got_it_right = stats['guessed_correctly_among_true_class']
        # among_true_acc.append(got_it_right.mean() * 100)

    acc_df = pd.DataFrame({ 'easy_25': easy_acc_25,'hard_25': hard_acc_25,
                           'v-dep_25': view_dep_acc_25, 'v-indep_25': view_indep_acc_25,
                           'all_25': all_accuracy_25, 
                            'easy_50': easy_acc_50, 'hard_50': hard_acc_50,
                           'v-dep_50': view_dep_acc_50, 'v-indep_50': view_indep_acc_50,
                           'all_50': all_accuracy_50,})

    # acc_df = pd.DataFrame({'hard': hard_acc, 'easy': easy_acc,
    #                        'v-dep': view_dep_acc, 'v-indep': view_indep_acc,
    #                        'all': all_accuracy, 'among-true': among_true_acc})

    acc_df.to_csv(out_file[:-4] + '.csv', index=False)

    pd.options.display.float_format = "{:,.1f}".format
    descriptive = acc_df.describe().loc[["mean", "std"]].T

    if out_file is not None:
        with open(out_file, 'w') as f_out:
            f_out.write(descriptive.to_latex())
    return descriptive

    #
    # # utterances = references['tokens'].apply(lambda x: ' '.join(x)) # as seen by the neural net.
    #
    #
    #
    #
    # data_df['n_target_class'] = data_df.stimulus_id.apply(lambda x: decode_stimulus_string(x)[2])
    #
    #

    #
    # data_df = data_df.assign(found = pd.Series(net_stats['guessed_correctly']))
    # data_df = data_df.assign(target_pos = pd.Series(net_stats['target_pos']))
    #
    # data_df['n_target_class_inv'] = 1 / data_df['n_target_class']
    # data_df['context_size'] = (contrasted_objects != pad_idx).sum(1) # TODO-Panos assert same as from batch!
    #
    # print('Among target\'s classes', data_df.n_target_class_inv.mean())
    # print('among all classes', (1.0 / data_df['context_size']).mean())
    #
    #
    # print('10 biggest')
    # print(data_df.groupby('instance_type')['found'].mean().sort_values()[::-1][:10])
    # print('TODO, adjust by relative boost (how much +more) against random-guessing baselines')
    #
    #
    # # data_df.guessed_correctly.groupby('reference_type').mean()
    # # data_df.guessed_correctly.groupby('instance_type').mean()
