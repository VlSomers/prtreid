from __future__ import division, print_function, absolute_import
from collections import OrderedDict

import os.path as osp
import torch
import numpy as np
from monai.losses import DiceLoss, FocalLoss
from tabulate import tabulate
from torchreid.losses import init_body_parts_triplet_loss, CrossEntropyLoss, TripletLoss
from ..engine import Engine
from ... import metrics
from ...metrics.distance import compute_distance_matrix_using_bp_features
from ...utils import plot_body_parts_pairs_distance_distribution, \
    plot_pairs_distance_distribution, re_ranking
from torchreid.utils.constants import *
from torch import nn
from ...utils.torchtools import collate
from ...utils.visualization.feature_map_visualization import display_feature_maps


class ImagePartBasedEngine(Engine):
    r"""Triplet-loss engine for part-based image-reid.
    """
    # TODO remove this class: add poses to "imgs" tensor and move "compute_distance_matrix" logic
    def __init__(
            self,
            datamanager,
            model,
            optimizer,
            writer,
            loss_name,
            config,
            dist_combine_strat,
            batch_size_pairwise_dist_matrix,
            engine_state,
            margin=0.3,
            scheduler=None,
            use_gpu=True,
            save_model_flag=False,
            mask_filtering_training=False,
            mask_filtering_testing=False
    ):
        super(ImagePartBasedEngine, self).__init__(config, datamanager, writer, engine_state, use_gpu=use_gpu, save_model_flag=save_model_flag, detailed_ranking=config.test.detailed_ranking)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parts_num = self.config.data.parts_num
        self.register_model('model', model, optimizer, scheduler)
        self.part_triplet_loss = init_body_parts_triplet_loss(loss_name, margin=margin, batch_size_pairwise_dist_matrix=batch_size_pairwise_dist_matrix)
        self.simple_triplet_loss = TripletLoss(margin=margin)
        self.identity_loss = CrossEntropyLoss(self.datamanager.num_train_pids, use_gpu=use_gpu)

        if self.config.loss.part_based.ppl == 'cl':
            self.part_prediction_loss = CrossEntropyLoss(self.parts_num+1, use_gpu=use_gpu, eps=0.1)
        elif self.config.loss.part_based.ppl == 'fl':
            self.part_prediction_loss = FocalLoss(to_onehot_y=True, gamma=1.0)
        elif self.config.loss.part_based.ppl == 'dl':
            self.part_prediction_loss = DiceLoss(to_onehot_y=True, softmax=True)
        else:
            raise ValueError("Loss {} for part prediction is not supported".format(self.config.loss.part_based.ppl))
        # TODO https://docs.monai.io/en/stable/losses.html#generalizeddiceloss
        self.losses_weights = self.config.loss.part_based.weights

        self.mask_filtering_training = mask_filtering_training
        self.mask_filtering_testing = mask_filtering_testing
        self.dist_combine_strat = dist_combine_strat
        self.batch_size_pairwise_dist_matrix = batch_size_pairwise_dist_matrix

        self.feature_extraction_timer = self.writer.feature_extraction_timer
        self.loss_timer = self.writer.loss_timer
        self.optimizer_timer = self.writer.optimizer_timer

    def forward_backward(self, data):
        imgs, target_masks, pids, imgs_path = self.parse_data_for_train(data)

        # feature extraction
        self.feature_extraction_timer.start()
        embeddings_dict, visibility_scores_dict, id_cls_scores_dict, pixels_cls_scores, spatial_features, part_masks = self.model(imgs, external_parts_masks=target_masks)
        display_feature_maps(embeddings_dict, spatial_features, part_masks, imgs_path, pids)
        self.feature_extraction_timer.stop()

        # loss
        self.loss_timer.start()
        loss, loss_summary = self.combine_losses(visibility_scores_dict, embeddings_dict, id_cls_scores_dict, pixels_cls_scores, target_masks, pids)
        self.loss_timer.stop()

        # optimization step
        self.optimizer_timer.start()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer_timer.stop()

        return loss, loss_summary

    def combine_losses(self, visibility_scores_dict, embeddings_dict, id_cls_scores_dict, pixels_cls_scores, target_masks, pids):
        # TODO visiblity = torch.ones(): first test if same results with both codes (because codes will be different)
        loss_summary = {}
        losses = []

        # global, foreground and parts embeddings id loss
        for key in [GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS]:
            dict = OrderedDict() if key not in loss_summary else loss_summary[key]
            ce_w = self.losses_weights[key].ce
            if ce_w > 0:
                # if len(id_cls_scores_dict[key].shape) == 3:
                #     parts_id_loss, parts_id_accuracy = self.compute_id_cls_loss(id_cls_scores_dict[key][:, 0, :],
                #                                                                 visibility_scores_dict[key][:, 0], pids,
                #                                                                 self.mask_filtering_training)
                # else:
                #     parts_id_loss, parts_id_accuracy = self.compute_id_cls_loss(id_cls_scores_dict[key],
                #                                                                 visibility_scores_dict[key], pids,
                #                                                                 self.mask_filtering_training)

                parts_id_loss, parts_id_accuracy = self.compute_id_cls_loss(id_cls_scores_dict[key],
                                                                            visibility_scores_dict[key], pids,
                                                                            self.mask_filtering_training)

                losses.append((ce_w, parts_id_loss))
                dict['c'] = parts_id_loss
                dict['a'] = parts_id_accuracy

            loss_summary[key] = dict

        # global, foreground and parts embeddings triplet loss
        for key in [GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS]:
            dict = OrderedDict() if key not in loss_summary else loss_summary[key]
            tr_w = self.losses_weights[key].tr
            if tr_w > 0:
                parts_triplet_loss, parts_trivial_triplets_ratio, parts_valid_triplets_ratio = \
                    self.compute_triplet_loss(embeddings_dict[key], visibility_scores_dict[key], pids,
                                              self.mask_filtering_training)
                losses.append((tr_w, parts_triplet_loss))
                dict['t'] = parts_triplet_loss
                dict['tt'] = parts_trivial_triplets_ratio
                dict['vt'] = parts_valid_triplets_ratio

            loss_summary[key] = dict

        # pixels classification loss
        loss_summary[PIXELS] = OrderedDict()
        if pixels_cls_scores is not None\
                and target_masks is not None\
                and self.losses_weights[PIXELS]['ce'] > 0:
            pixels_cls_loss, pixels_cls_accuracy = self.compute_pixels_cls_loss(target_masks, pixels_cls_scores)
            losses.append((self.losses_weights[PIXELS].ce, pixels_cls_loss))
            loss_summary[PIXELS]['c'] = pixels_cls_loss
            loss_summary[PIXELS]['a'] = pixels_cls_accuracy

        loss = torch.stack([weight * loss for weight, loss in losses]).sum()

        return loss, loss_summary

    def compute_pixels_cls_loss(self, target_masks, pixels_cls_scores):
        if pixels_cls_scores.is_cuda:
            target_masks = target_masks.cuda()
        target_masks = nn.functional.interpolate(target_masks, pixels_cls_scores.shape[2::], mode='bilinear', align_corners=True)  # Best perf with bilinear here and nearest in resize transform
        pixels_cls_score_targets = target_masks.argmax(dim=1)  # [N, Hf, Wf]
        pixels_cls_score_targets = pixels_cls_score_targets.flatten()  # [N*Hf*Wf]
        pixels_cls_scores = pixels_cls_scores.permute(0, 2, 3, 1).flatten(0, 2)  # [N*Hf*Wf, M]
        loss = self.part_prediction_loss(pixels_cls_scores, pixels_cls_score_targets)
        accuracy = metrics.accuracy(pixels_cls_scores, pixels_cls_score_targets)[0]
        return loss, accuracy.item()

    def compute_triplet_loss(self, embeddings, visibility_scores, pids, use_visibility_scores):
        if use_visibility_scores:
            visibility = visibility_scores if len(visibility_scores.shape) == 2 else visibility_scores.unsqueeze(1)
        else:
            visibility = None
        embeddings = embeddings if len(embeddings.shape) == 3 else embeddings.unsqueeze(1)
        triplet_loss, trivial_triplets_ratio, valid_triplets_ratio = self.part_triplet_loss(embeddings, pids, parts_visibility=visibility)
        return triplet_loss, trivial_triplets_ratio, valid_triplets_ratio

    def compute_id_cls_loss(self, id_cls_scores, visibility_scores, pids, use_visibility_scores):
        if len(id_cls_scores.shape) == 3:
            M = id_cls_scores.shape[1]
            id_cls_scores = id_cls_scores.flatten(0, 1)
            pids = pids.unsqueeze(1).expand(-1, M).flatten(0, 1)
            visibility_scores = visibility_scores.flatten(0, 1)
        weights = None
        if use_visibility_scores and visibility_scores.dtype is torch.bool:
            id_cls_scores = id_cls_scores[visibility_scores]
            pids = pids[visibility_scores]
        elif use_visibility_scores and visibility_scores.dtype is not torch.bool:
            weights = visibility_scores
        cls_loss = self.identity_loss(id_cls_scores, pids, weights)
        accuracy = metrics.accuracy(id_cls_scores, pids)[0]
        return cls_loss, accuracy

    def _feature_extraction(self, data_loader):
        f_, pids_, camids_, parts_visibility_, p_masks_, pxl_scores_, anns = [], [], [], [], [], [], []
        for batch_idx, data in enumerate(data_loader):
            imgs, masks, pids, camids = self.parse_data_for_eval(data)
            if self.use_gpu:
                if masks is not None:
                    masks = masks.cuda()
                imgs = imgs.cuda()
            self.writer.test_batch_timer.start()
            features, visibility_scores, parts_masks, pixels_cls_scores = self.model(imgs, external_parts_masks=masks)
            self.writer.test_batch_timer.stop()
            if self.mask_filtering_testing:
                parts_visibility = visibility_scores
                parts_visibility = parts_visibility.cpu()
                parts_visibility_.append(parts_visibility)
            else:
                parts_visibility_ = None
            features = features.data.cpu()
            parts_masks = parts_masks.data.cpu()
            f_.append(features)
            p_masks_.append(parts_masks)
            pxl_scores_.append(pixels_cls_scores)
            pids_.extend(pids)
            camids_.extend(camids)
            anns.append(data)
        if self.mask_filtering_testing:
            parts_visibility_ = torch.cat(parts_visibility_, 0)
        f_ = torch.cat(f_, 0)
        p_masks_ = torch.cat(p_masks_, 0)
        pxl_scores_ = torch.cat(pxl_scores_, 0) if pxl_scores_[0] is not None else None
        pids_ = np.asarray(pids_)
        camids_ = np.asarray(camids_)
        anns = collate(anns)
        return f_, pids_, camids_, parts_visibility_, p_masks_, pxl_scores_, anns

    @torch.no_grad()
    def _evaluate(
        self,
        epoch,
        dataset_name='',
        query_loader=None,
        gallery_loader=None,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        visrank_q_idx_list=[],
        visrank_count=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False,
        save_features=False
    ):
        print('Extracting features from query set ...')
        qf, q_pids, q_camids, qf_parts_visibility, q_parts_masks, q_pxl_scores_, q_anns = self._feature_extraction(query_loader)
        print('Done, obtained {} tensor'.format(qf.shape))

        print('Extracting features from gallery set ...')
        gf, g_pids, g_camids, gf_parts_visibility, g_parts_masks, g_pxl_scores_, g_anns = self._feature_extraction(gallery_loader)
        print('Done, obtained {} tensor'.format(gf.shape))

        print('Test batch feature extraction speed: {:.4f} sec/batch'.format(self.writer.test_batch_timer.avg))

        if save_features:
            features_dir = osp.join(save_dir, 'features')
            print('Saving features to : ' + features_dir)
            # TODO create if doesn't exist
            torch.save(gf, osp.join(features_dir, 'gallery_features_' + dataset_name + '.pt'))
            torch.save(qf, osp.join(features_dir, 'query_features_' + dataset_name + '.pt'))
            # save pids, camids and feature length

        self.writer.performance_evaluation_timer.start()
        if normalize_feature:
            print('Normalizing features with L2 norm ...')
            qf = self.normalize(qf)
            gf = self.normalize(gf)
        print('Computing distance matrix with metric={} ...'.format(dist_metric))
        distmat, body_parts_distmat = compute_distance_matrix_using_bp_features(qf, gf, qf_parts_visibility,
                                                                                      gf_parts_visibility,
                                                                                      self.dist_combine_strat,
                                                                                      self.batch_size_pairwise_dist_matrix,
                                                                                      self.use_gpu, dist_metric)
        # display_attributes_ranking(body_parts_distmat, distmat, g_camids, g_pids, q_camids, q_pids, use_metric_cuhk03)
        distmat = distmat.numpy()
        body_parts_distmat = body_parts_distmat.numpy()
        if rerank:
            print('Applying person re-ranking ...')
            distmat_qq, body_parts_distmat_qq = compute_distance_matrix_using_bp_features(qf, qf, qf_parts_visibility, qf_parts_visibility,
                                                                    self.dist_combine_strat, self.batch_size_pairwise_dist_matrix,
                                                                    self.use_gpu, dist_metric)
            distmat_gg, body_parts_distmat_gg = compute_distance_matrix_using_bp_features(gf, gf, gf_parts_visibility, gf_parts_visibility,
                                                                     self.dist_combine_strat, self.batch_size_pairwise_dist_matrix,
                                                                     self.use_gpu, dist_metric)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        eval_metric = self.datamanager.test_loader[dataset_name]['query'].dataset.eval_metric

        print('Computing CMC and mAP ...')
        eval_metrics = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            q_anns=q_anns,
            g_anns=g_anns,
            eval_metric=eval_metric
        )

        mAP = eval_metrics['mAP']
        cmc = eval_metrics['cmc']
        print('** Results **')
        print('mAP: {:.2%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))

        for metric in eval_metrics.keys():
            if metric != 'mAP' and metric != 'cmc':
                val, size = eval_metrics[metric]
                if val is not None:
                    print('{:<20}: {:.2%} ({})'.format(metric, val, size))
                else:
                    print('{:<20}: not provided'.format(metric))

        # Parts ranking
        if self.detailed_ranking:
            self.display_individual_parts_ranking_performances(body_parts_distmat, cmc, g_camids, g_pids, mAP,
                                                               q_camids, q_pids, eval_metric)
        # TODO move below to writer
        plot_body_parts_pairs_distance_distribution(body_parts_distmat, q_pids, g_pids, "Query-gallery")
        print('Evaluate distribution of distances of pairs with same id vs different ids')
        same_ids_dist_mean, same_ids_dist_std, different_ids_dist_mean, different_ids_dist_std, ssmd = \
            plot_pairs_distance_distribution(distmat, q_pids, g_pids,
                                             "Query-gallery")  # TODO separate ssmd from plot, put plot in writer
        print("Positive pairs distance distribution mean: {:.3f}".format(same_ids_dist_mean))
        print("Positive pairs distance distribution standard deviation: {:.3f}".format(same_ids_dist_std))
        print("Negative pairs distance distribution mean: {:.3f}".format(different_ids_dist_mean))
        print("Negative pairs distance distribution standard deviation: {:.3f}".format(
            different_ids_dist_std))
        print("SSMD = {:.4f}".format(ssmd))

        # if groundtruth target body masks are provided, compute part prediction accuracy
        avg_pxl_pred_accuracy = 0.0
        if 'mask' in q_anns and 'mask' in g_anns and q_pxl_scores_ is not None and g_pxl_scores_ is not None:
            _, q_pxl_pred_accuracy = self.compute_pixels_cls_loss(torch.from_numpy(q_anns['mask']), q_pxl_scores_)
            _, g_pxl_pred_accuracy = self.compute_pixels_cls_loss(torch.from_numpy(g_anns['mask']), g_pxl_scores_)
            avg_pxl_pred_accuracy = (q_pxl_pred_accuracy * len(q_parts_masks) + g_pxl_pred_accuracy * len(g_parts_masks)) /\
                                    (len(q_parts_masks) + len(g_parts_masks))
            print("Pixel prediction accuracy for query = {:.2f}% and for gallery = {:.2f}% and on average = {:.2f}%"
                  .format(q_pxl_pred_accuracy, g_pxl_pred_accuracy, avg_pxl_pred_accuracy))

        if visrank:
            self.writer.visualize_rank(self.datamanager.test_loader[dataset_name], dataset_name, distmat, save_dir,
                                       visrank_topk, visrank_q_idx_list, visrank_count,
                                       body_parts_distmat, qf_parts_visibility, gf_parts_visibility, q_parts_masks,
                                       g_parts_masks, mAP, cmc[0])

        self.writer.visualize_embeddings(qf, gf, q_pids, g_pids, self.datamanager.test_loader[dataset_name],
                                         dataset_name,
                                         qf_parts_visibility, gf_parts_visibility, mAP, cmc[0])
        self.writer.performance_evaluation_timer.stop()
        return cmc, mAP, ssmd, avg_pxl_pred_accuracy

    def display_individual_parts_ranking_performances(self, body_parts_distmat, cmc, g_camids, g_pids, mAP, q_camids,
                                                      q_pids, eval_metric):
        print('Parts embeddings individual rankings :')
        bp_offset = 0
        if GLOBAL in self.config.model.bpbreid.test_embeddings:
            bp_offset += 1
        if FOREGROUND in self.config.model.bpbreid.test_embeddings:
            bp_offset += 1
        table = []
        for bp in range(0, body_parts_distmat.shape[0]):  # TODO DO NOT TAKE INTO ACCOUNT -1 DISTANCES!!!!
            perf_metrics = metrics.evaluate_rank(
                body_parts_distmat[bp],
                q_pids,
                g_pids,
                q_camids,
                g_camids,
                eval_metric=eval_metric
            )
            title = 'p {}'.format(bp - bp_offset)
            if bp < bp_offset:
                if bp == 0:
                    if GLOBAL in self.config.model.bpbreid.test_embeddings:
                        title = GLOBAL
                    else:
                        title = FOREGROUND
                if bp == 1:
                    title = FOREGROUND
            mAP = perf_metrics['mAP']
            cmc = perf_metrics['cmc']
            table.append([title, mAP, cmc[0], cmc[4], cmc[9]])
        headers = ["embed", "mAP", "R-1", "R-5", "R-10"]
        print(tabulate(table, headers, tablefmt="fancy_grid", floatfmt=".3f"))

    def parse_data_for_train(self, data):
        imgs = data['image']
        imgs_path = data['img_path']
        masks = data['mask'] if 'mask' in data else None
        pids = data['pid']

        if self.use_gpu:
            imgs = imgs.cuda()
            if masks is not None:
                masks = masks.cuda()
            pids = pids.cuda()

        if masks is not None:
            assert masks.shape[1] == (self.config.data.parts_num + 1)

        return imgs, masks, pids, imgs_path

    def parse_data_for_eval(self, data):
        imgs = data['image']
        masks = data['mask'] if 'mask' in data else None
        pids = data['pid']
        camids = data['camid']
        return imgs, masks, pids, camids

