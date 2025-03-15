import torch
import argparse
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import sys
sys.path.append('/data/xuxiaoxu/code/3dvg/MVT-ws')
import referit3d.clip as clip
from . import DGCNN
from .utils import get_siamese_features, my_get_siamese_features
from ..in_out.vocabulary import Vocabulary
import math
try:
    from . import PointNetPP
except ImportError:
    PointNetPP = None

from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from transformers import BertTokenizer, BertModel, BertConfig
from referit3d.models import MLP
import time


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            # nn.GELU(),

            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
            # nn.GELU(),
        )

        # self.fc = nn.Sequential(
        #     nn.Conv1d(c_in, c_in // 2, 3, padding=1),
        #     nn.BatchNorm1d(c_in // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(c_in // 2, c_in, 3, padding=1),
        #     nn.BatchNorm1d(c_in),
        # )

    def forward(self, x):
        x = self.fc(x)
        return x


class ReferIt3DNet_transformer(nn.Module):

    def __init__(self,
                 args,
                 n_obj_classes,
                 ignore_index,
                 device=None):

        super().__init__()

        self.bert_pretrain_path = args.bert_pretrain_path

        self.view_number = args.view_number
        self.rotate_number = args.rotate_number

        self.label_lang_sup = args.label_lang_sup
        self.aggregate_type = args.aggregate_type

        self.encoder_layer_num = args.encoder_layer_num
        self.decoder_layer_num = args.decoder_layer_num
        self.decoder_nhead_num = args.decoder_nhead_num

        self.object_dim = args.object_latent_dim
        self.inner_dim = args.inner_dim
        
        self.dropout_rate = args.dropout_rate
        self.lang_cls_alpha = args.lang_cls_alpha
        self.obj_cls_alpha = args.obj_cls_alpha

        self.contrastive_alpha = args.contrastive_alpha

        self.object_encoder = PointNetPP(sa_n_points=[32, 16, None],
                                        sa_n_samples=[[32], [32], [None]],
                                        sa_radii=[[0.2], [0.4], [None]],
                                        sa_mlps=[[[3, 64, 64, 128]],
                                                [[128, 128, 128, 256]],
                                                [[256, 256, self.object_dim, self.object_dim]]])

        self.refer_encoder = nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(d_model=self.inner_dim, 
            nhead=self.decoder_nhead_num, dim_feedforward=2048, activation="gelu"), num_layers=self.decoder_layer_num)

        # Classifier heads
        self.language_clf = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim), 
                                        nn.ReLU(), nn.Dropout(self.dropout_rate), 
                                        nn.Linear(self.inner_dim, n_obj_classes))

        self.object_language_clf = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim), 
                                                nn.ReLU(), nn.Dropout(self.dropout_rate), 
                                                nn.Linear(self.inner_dim, 1))

        if not self.label_lang_sup:
            self.obj_clf = MLP(self.inner_dim, [self.object_dim, self.object_dim, n_obj_classes], dropout_rate=self.dropout_rate)

        self.obj_feature_mapping = nn.Sequential(
            nn.Linear(self.object_dim, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
        )

        self.box_feature_mapping = nn.Sequential(
            nn.Linear(4, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
        )


        self.model, _ = clip.load("ViT-B/32", device=device)

        self.lang_feat_projection = nn.Linear(512, self.inner_dim)

        self.logit_loss = nn.CrossEntropyLoss()
        self.lang_logits_loss = nn.CrossEntropyLoss()
        self.class_logits_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)


        self.image_adapter = Adapter(512)
        self.pc_adapter = Adapter(512)
        self.text_adapter = Adapter(512)

        self.text_mapping = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim), 
                                                nn.ReLU())
        self.image_mapping = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim), 
                                                nn.ReLU())
        self.pc_mapping = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim), 
                                                nn.ReLU())

        self.image_obj_clf = MLP(self.inner_dim, [self.object_dim, self.object_dim, n_obj_classes], dropout_rate=self.dropout_rate)
        self.pc_obj_clf = MLP(self.inner_dim, [self.object_dim, self.object_dim, n_obj_classes], dropout_rate=self.dropout_rate)
        self.class_lang_features = torch.load("/data/xuxiaoxu/code/3dvg/MVT-ws/referit3d/scripts/cls_feature_eot.pth").to(device).float()[:-1]

    @torch.no_grad()
    def aug_input(self, input_points, box_infos):
        input_points = input_points.float().to(self.device)
        box_infos = box_infos.float().to(self.device)
        xyz = input_points[:, :, :, :3]
        bxyz = box_infos[:,:,:3] # B,N,3
        B,N,P = xyz.shape[:3]
        rotate_theta_arr = torch.Tensor([i*2.0*np.pi/self.rotate_number for i in range(self.rotate_number)]).to(self.device)
        view_theta_arr = torch.Tensor([i*2.0*np.pi/self.view_number for i in range(self.view_number)]).to(self.device)
        
        # rotation
        if self.training:
            # theta = torch.rand(1) * 2 * np.pi  # random direction rotate aug
            theta = rotate_theta_arr[torch.randint(0,self.rotate_number,(B,))]  # 4 direction rotate aug
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rotate_matrix = torch.Tensor([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,1.0]]).to(self.device)[None].repeat(B,1,1)
            rotate_matrix[:, 0, 0] = cos_theta
            rotate_matrix[:, 0, 1] = -sin_theta
            rotate_matrix[:, 1, 0] = sin_theta
            rotate_matrix[:, 1, 1] = cos_theta

            input_points[:, :, :, :3] = torch.matmul(xyz.reshape(B,N*P,3), rotate_matrix).reshape(B,N,P,3)
            bxyz = torch.matmul(bxyz.reshape(B,N,3), rotate_matrix).reshape(B,N,3)
        # return input_points, bxyz
        # multi-view
        bsize = box_infos[:,:,-1:]
        boxs=[]
        for theta in view_theta_arr:
            # print("theta:", theta, bxyz[0].sum())
            rotate_matrix = torch.Tensor([[math.cos(theta), -math.sin(theta), 0.0],
                                        [math.sin(theta), math.cos(theta),  0.0],
                                        [0.0,           0.0,            1.0]]).to(self.device)
            rxyz = torch.matmul(bxyz.reshape(B*N, 3),rotate_matrix).reshape(B,N,3)

            boxs = torch.cat([rxyz,bsize],dim=-1)

        return input_points, boxs

    def compute_loss(self, batch, CLASS_LOGITS, LANG_LOGITS, LOGITS, AUX_LOGITS=None):

        referential_loss = self.logit_loss(LOGITS, batch['target_pos'])
        obj_clf_loss = self.class_logits_loss(CLASS_LOGITS.transpose(2, 1), batch['class_labels'])
        if self.lang_cls_alpha > 0:
            lang_clf_loss = self.lang_logits_loss(LANG_LOGITS, batch['target_class'])
            total_loss = referential_loss + self.obj_cls_alpha * obj_clf_loss + self.lang_cls_alpha * lang_clf_loss
        else:
            total_loss = referential_loss + self.obj_cls_alpha * obj_clf_loss
        
        return total_loss, referential_loss, self.obj_cls_alpha * obj_clf_loss, self.lang_cls_alpha * lang_clf_loss

    def compute_contrastive_loss(self, objects_features_before_2d, objects_features_before):
        # contrastive loss
        # 2d :bxN2xC 3d : bxN3xC 
        # 2d: bx52x512 3d : bx52x512
        feature_2d = objects_features_before_2d
        feature_3d = objects_features_before
        temperature = 10
        l = feature_2d.shape[1]

        logit = torch.matmul(feature_2d, feature_3d.transpose(-1, -2)) / temperature # BxN3xN2

        ##### 2d to 3d contrastive loss #####
        label= torch.tensor([i for i in range(l)]).reshape(1, -1).repeat(logit.shape[0], 1).cuda()

        softmax = nn.LogSoftmax(dim=-1)
        loss_func = nn.NLLLoss()

        x_log = softmax(logit)
        loss_contrastive_2d = loss_func(x_log,label)

        x_log = softmax(logit.transpose(-1, -2))
        loss_contrastive_3d = loss_func(x_log,label)
        total_loss = loss_contrastive_2d + loss_contrastive_3d

        return total_loss


    def forward(self, batch: dict, epoch=None):
        # batch['class_labels']: GT class of each obj
        # batch['target_class']：GT class of target obj
        # batch['target_pos']: GT id

        self.device = self.obj_feature_mapping[0].weight.device

        #################### lang clf ###################
        with torch.no_grad():
            text = clip.tokenize(batch['text']).to(batch['objects'].device)
            lang_infos = self.model.encode_text(text).float().cuda()
        
        text_adapter_fts = self.text_adapter(lang_infos)
        text_adapter_fts = text_adapter_fts * 0.99 + lang_infos * 0.01
        LANG_LOGITS = self.language_clf(text_adapter_fts)

        lang_cls_pred = torch.argmax(LANG_LOGITS, dim=1)

        # use the top2 cls
        lang_logit = LANG_LOGITS.clone()
        lang_logit[lang_logit == lang_cls_pred.reshape(-1, 1)] = -10e6
        lang_cls_pred_top2 = torch.argmax(lang_logit, dim=1)
        
        
        # use the top3 cls
        lang_logit = LANG_LOGITS.clone()
        lang_logit[lang_logit == lang_cls_pred.reshape(-1, 1)] = -10e6
        lang_logit[lang_logit == lang_cls_pred_top2.reshape(-1, 1)] = -10e6
        lang_cls_pred_top3 = torch.argmax(lang_logit, dim=1)

        ######################  main backbone ##############
        ## rotation augmentation and multi_view generation
        obj_points, boxs = self.aug_input(batch['objects'], batch['box_info'])
        B,N,P = obj_points.shape[:3]
        
        ## obj_encoding
        objects_features = get_siamese_features(self.object_encoder, obj_points, aggregator=torch.stack) # BxNxC
        obj_3d_feats = self.obj_feature_mapping(objects_features)# BxNxC blood!!!
        box_infos = self.box_feature_mapping(boxs)# BxNxC
        obj_3d_feats = obj_3d_feats + box_infos

        cat_infos = obj_3d_feats.reshape(B, -1, self.inner_dim)

        out_feats = self.refer_encoder(cat_infos.transpose(0, 1), cat_infos.transpose(0, 1)).transpose(0, 1).reshape(B, -1, self.inner_dim)


        obj_2d_feats = batch['images'].float()
        ## view_aggregation
        refer_feat = out_feats

        # ##### adapter
        image_adapter_fts = self.image_adapter(obj_2d_feats.reshape(B*N,-1)).reshape(B,N,-1)
        pc_adapter_fts = self.pc_adapter(refer_feat.reshape(B*N,-1)).reshape(B,N,-1)

        # <LOSS>: obj_cls
        class_lang_features = self.class_lang_features

        ratio = 0.2
        obj_res_add = obj_2d_feats * (1 - ratio) + image_adapter_fts * ratio
        pc_res_add = refer_feat * (1 - ratio) + pc_adapter_fts * ratio

        ratio_class = 0.01
        class_adapter_fts = self.text_adapter(class_lang_features)
        class_lang_features = class_lang_features * (1 - ratio_class) + class_adapter_fts * ratio_class

        CLASS_2D_LOGITS = torch.matmul(obj_res_add.reshape(B*N,-1), class_lang_features.permute(1,0)).reshape(B,N,-1)   
        CLASS_3D_LOGITS = torch.matmul(pc_res_add.reshape(B*N,-1), class_lang_features.permute(1,0)).reshape(B,N,-1)   

        ########################  filter the obj(compare the obj clf pred bbox class) ############

        batch_class = CLASS_3D_LOGITS.argmax(dim=-1)
        batch_class_mask = torch.zeros(batch_class.shape)
        
        target_class = lang_cls_pred.reshape(-1, 1)
        target_class_top_2 = lang_cls_pred_top2.reshape(-1, 1)
        target_class_top_3 = lang_cls_pred_top3.reshape(-1, 1)

        # generate the mask
        batch_class_mask[batch_class == target_class_top_3] = 1
        batch_class_mask[batch_class == target_class_top_2] = 1
        batch_class_mask[batch_class == target_class] = 1

        batch_class_mask = batch_class_mask.bool()
        num_obj = batch_class_mask.sum(dim=-1)

        ################################ inference #################
        with torch.no_grad():
            LOGITS = torch.einsum("ijk,ik->ij", refer_feat,lang_infos)
            LOGITS_after = torch.einsum("ijk,ik->ij", obj_2d_feats,lang_infos)
            batch_size = LOGITS.shape[0]

            for b in range(batch_size):
                if num_obj[b] == 0:
                    continue
                mask = batch_class_mask[b]
                LOGITS[b][mask == 0] = -10e6
        
        # <LOSS>: contrastive loss
        contrastive_loss_L1 = self.compute_contrastive_loss(obj_2d_feats, refer_feat)
        contrastive_loss_L2 = self.compute_contrastive_loss(image_adapter_fts, pc_adapter_fts)

        # <LOSS>: class logit loss
        obj_2d_clf_loss = self.class_logits_loss(CLASS_2D_LOGITS.transpose(2, 1), batch['class_labels'])
        obj_3d_clf_loss = self.class_logits_loss(CLASS_3D_LOGITS.transpose(2, 1), batch['class_labels'])

        # <LOSS>: text loss 
        text_clf_loss = self.lang_logits_loss(LANG_LOGITS, batch['target_class'])

        LOSS = (contrastive_loss_L1 + contrastive_loss_L2) * 0.5 + obj_2d_clf_loss + obj_3d_clf_loss + text_clf_loss

        return LOSS, LOGITS, LOGITS_after, CLASS_2D_LOGITS, CLASS_3D_LOGITS, LANG_LOGITS, obj_2d_clf_loss + obj_3d_clf_loss, text_clf_loss, contrastive_loss_L2