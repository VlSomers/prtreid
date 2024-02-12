from __future__ import division, print_function, absolute_import

import os.path as osp
import torch
import numpy as np
from tabulate import tabulate
from torch import nn
from tqdm import tqdm
from pathlib import Path
import pickle

from ...losses import CrossEntropyLoss
from ..engine import Engine
from ... import metrics
from ...losses.GiLt_loss import GiLtLoss
from ...losses.body_part_attention_loss import BodyPartAttentionLoss
from ...metrics.distance import compute_distance_matrix_using_bp_features
from ...utils import (
    plot_body_parts_pairs_distance_distribution,
    plot_pairs_distance_distribution,
    re_ranking,
)
from prtreid.utils.constants import *

from ...utils.tools import extract_test_embeddings
from ...utils.torchtools import collate
from ...utils.visualization.feature_map_visualization import display_feature_maps

################ focal loss for role classification #############################
import torch
import torch.nn as nn
import torch.nn.functional as F
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
####################################################################

class ImagePartBasedEngine(Engine):
    r"""Training/testing engine for part-based image-reid.
    """

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
        mask_filtering_testing=False,
    ):
        super(ImagePartBasedEngine, self).__init__(
            config,
            datamanager,
            writer,
            engine_state,
            use_gpu=use_gpu,
            save_model_flag=save_model_flag,
            detailed_ranking=config.test.detailed_ranking,
        )

        self.model = model
        self.register_model("model", model, optimizer, scheduler)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parts_num = self.config.model.bpbreid.masks.parts_num
        self.mask_filtering_training = mask_filtering_training
        self.mask_filtering_testing = mask_filtering_testing
        self.dist_combine_strat = dist_combine_strat
        self.batch_size_pairwise_dist_matrix = batch_size_pairwise_dist_matrix
        self.losses_weights = self.config.loss.part_based.weights

        # Losses
        self.GiLt = GiLtLoss(
            self.losses_weights,
            use_visibility_scores=self.mask_filtering_training,
            triplet_margin=margin,
            loss_name=loss_name,
            writer=self.writer,
            use_gpu=self.use_gpu,
        )

        self.GiLt_team = GiLtLoss(
            self.losses_weights,
            use_visibility_scores=self.mask_filtering_training,
            triplet_margin=0.4,
            loss_name=loss_name,
            writer=self.writer,
            use_gpu=self.use_gpu,
        )

        self.role_loss = FocalLoss()
        # self.role_loss = CrossEntropyLoss(label_smooth=True)

        self.body_part_attention_loss = BodyPartAttentionLoss(
            loss_type=self.config.loss.part_based.ppl, use_gpu=self.use_gpu
        )

        # Timers
        self.feature_extraction_timer = self.writer.feature_extraction_timer
        self.loss_timer = self.writer.loss_timer
        self.optimizer_timer = self.writer.optimizer_timer

    def forward_backward(self, data):
        imgs, target_masks, pids, teams, roles, imgs_path = self.parse_data_for_train(data)

        # feature extraction
        self.feature_extraction_timer.start()
        (
            embeddings_dict,
            visibility_scores_dict,
            id_cls_scores_dict,
            team_cls_scores_dict,
            role_cls_scores_dict,
            pixels_cls_scores,
            spatial_features,
            masks,
        ) = self.model(imgs, external_parts_masks=target_masks)
        display_feature_maps(
            embeddings_dict, spatial_features, masks[PARTS], imgs_path, pids
        )
        self.feature_extraction_timer.stop()

        # loss
        self.loss_timer.start()
        loss, loss_summary = self.combine_losses(
            visibility_scores_dict,
            embeddings_dict,
            id_cls_scores_dict,
            team_cls_scores_dict,
            role_cls_scores_dict,
            pids,
            teams,
            roles,
            pixels_cls_scores,
            target_masks,
            bpa_weight=self.losses_weights[PIXELS]["ce"],
        )
        self.loss_timer.stop()

        # optimization step
        self.optimizer_timer.start()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer_timer.stop()

        return loss, loss_summary

    def combine_losses(
        self,
        visibility_scores_dict,
        embeddings_dict,
        id_cls_scores_dict,
        team_cls_scores_dict,
        role_cls_scores_dict,
        pids,
        teams,
        roles,
        pixels_cls_scores=None,
        target_masks=None,
        bpa_weight=0,
    ):
        # 1. ReID objective:
        # GiLt loss on holistic and part-based embeddings
        reid_loss, reid_loss_summary = self.GiLt(
            embeddings_dict, visibility_scores_dict, id_cls_scores_dict, pids)
        loss_summary = reid_loss_summary

        # #########################################################
        #
        role_loss = self.role_loss(role_cls_scores_dict[GLOBAL], roles)

        ################ team affiliation is applied just on players(not refree, goalkeeper) ###############
        for k in embeddings_dict.keys():
            embeddings_dict[k] = embeddings_dict[k][:24]
        # embeddings_dict[PARTS] = embeddings_dict[PARTS][:, 1:4]
        for k in visibility_scores_dict.keys():
            visibility_scores_dict[k] = visibility_scores_dict[k][:24]
        # visibility_scores_dict[PARTS] = visibility_scores_dict[PARTS][:, 1:4]
        for k in team_cls_scores_dict.keys():
            team_cls_scores_dict[k] = team_cls_scores_dict[k][:24]
        # team_cls_scores_dict[PARTS] = team_cls_scores_dict[PARTS][:, 1:4]
        team_loss, team_loss_summary = self.GiLt_team(
            embeddings_dict, visibility_scores_dict, team_cls_scores_dict, teams[:24], 1)

        #################### multi-task objective ##################
        loss = reid_loss + 0.1 * team_loss + 1.5 * role_loss
        ############################################################

        # 2. Part prediction objective:
        # Body part attention loss on spatial feature map
        if (
            pixels_cls_scores is not None
            and target_masks is not None
            and bpa_weight > 0
        ):
            # resize external masks to fit feature map size
            target_masks = nn.functional.interpolate(
                target_masks,
                pixels_cls_scores.shape[2::],
                mode="bilinear",
                align_corners=True,
            )
            # compute target part index for each spatial location, i.e. each spatial location (pixel) value indicate
            # the (body) part that spatial location belong to, or 0 for background.
            pixels_cls_score_targets = target_masks.argmax(dim=1)  # [N, Hf, Wf]
            # compute the classification loss for each pixel
            bpa_loss, bpa_loss_summary = self.body_part_attention_loss(
                pixels_cls_scores, pixels_cls_score_targets
            )
            loss += bpa_weight * bpa_loss
            loss_summary = {**loss_summary, **bpa_loss_summary}

        return loss, loss_summary

    def _feature_extraction(self, data_loader):
        f_, pids_, camids_, teams_, gids_, roles_, parts_visibility_, p_masks_, pxl_scores_, role_scores_, anns = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for batch_idx, data in enumerate(tqdm(data_loader, desc=f"Batches processed")):
            imgs, masks, pids, teams, camids, gids, roles = self.parse_data_for_eval(data)
            if self.use_gpu:
                if masks is not None:
                    masks = masks.cuda()
                imgs = imgs.cuda()
            self.writer.test_batch_timer.start()
            model_output = self.model(imgs, external_parts_masks=masks)
            (
                features,
                visibility_scores,
                parts_masks,
                pixels_cls_scores,
                role_cls_scores,
            ) = extract_test_embeddings(
                model_output, self.config.model.bpbreid.test_embeddings
            )
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
            pxl_scores_.append(pixels_cls_scores.data.cpu() if pixels_cls_scores is not None else None)
            role_scores_.append(role_cls_scores[GLOBAL].cpu() if role_cls_scores is not None else None)
            pids_.extend(pids)
            teams_.extend(teams)
            camids_.extend(camids)
            gids_.extend(gids)
            roles_.extend(roles)
            anns.append(data)
        if self.mask_filtering_testing:
            parts_visibility_ = torch.cat(parts_visibility_, 0)
        f_ = torch.cat(f_, 0)
        p_masks_ = torch.cat(p_masks_, 0)
        pxl_scores_ = torch.cat(pxl_scores_, 0) if pxl_scores_[0] is not None else None
        role_scores_ = torch.cat(role_scores_, 0) if role_scores_[0] is not None else None
        pids_ = np.asarray(pids_)
        teams_ = np.asarray(teams_)
        camids_ = np.asarray(camids_)
        gids_ = np.asarray(gids_)
        roles_ = np.asarray(roles_)
        anns = collate(anns)
        return f_, pids_, teams_, camids_, gids_, roles_, parts_visibility_, p_masks_, pxl_scores_, role_scores_, anns

    @torch.no_grad()
    def _evaluate(
        self,
        epoch,
        dataset_name="",
        query_loader=None,
        gallery_loader=None,
        dist_metric="euclidean",
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        visrank_q_idx_list=[],
        visrank_count=10,
        save_dir="",
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False,
        save_features=False,
    ):
        print("Extracting features from query set ...")
        (
            qf,
            q_pids,
            q_teams,
            q_camids,
            q_gids,
            q_roles,
            qf_parts_visibility,
            q_parts_masks,
            q_pxl_scores_,
            q_role_scores_,
            q_anns,
        ) = self._feature_extraction(query_loader)
        print("Done, obtained {} tensor".format(qf.shape))

        print("Extracting features from gallery set ...")
        (
            gf,
            g_pids,
            g_teams,
            g_camids,
            g_gids,
            g_roles,
            gf_parts_visibility,
            g_parts_masks,
            g_pxl_scores_,
            g_role_scores_,
            g_anns,
        ) = self._feature_extraction(gallery_loader)

        print("Done, obtained {} tensor".format(gf.shape))

        print(
            "Test batch feature extraction speed: {:.4f} sec/batch".format(
                self.writer.test_batch_timer.avg
            )
        )

        if save_features:
            dict = {'features': gf, 'cids': g_camids, 'team': g_teams, 'role': g_roles, 'game_id': g_gids, 'vis_scores': gf_parts_visibility}
            features_dir = osp.join(save_dir, 'features')
            print('Saving features to : ' + features_dir)
            # TODO create if doesn't exist
            torch.save(gf, osp.join(features_dir, 'gallery_features_' + dataset_name + '.pt'))
            torch.save(qf, osp.join(features_dir, 'query_features_' + dataset_name + '.pt'))
            # save pids, camids and feature length

        self.writer.performance_evaluation_timer.start()
        if normalize_feature:
            print("Normalizing features with L2 norm ...")
            qf = self.normalize(qf)
            gf = self.normalize(gf)
        print("Computing distance matrix with metric={} ...".format(dist_metric))
        distmat, body_parts_distmat = compute_distance_matrix_using_bp_features(
            qf,
            gf,
            qf_parts_visibility,
            gf_parts_visibility,
            self.dist_combine_strat,
            self.batch_size_pairwise_dist_matrix,
            self.use_gpu,
            dist_metric,
        )
        distmat = distmat.numpy()
        body_parts_distmat = body_parts_distmat.numpy()
        if rerank:
            print("Applying person re-ranking ...")
            (
                distmat_qq,
                body_parts_distmat_qq,
            ) = compute_distance_matrix_using_bp_features(
                qf,
                qf,
                qf_parts_visibility,
                qf_parts_visibility,
                self.dist_combine_strat,
                self.batch_size_pairwise_dist_matrix,
                self.use_gpu,
                dist_metric,
            )
            (
                distmat_gg,
                body_parts_distmat_gg,
            ) = compute_distance_matrix_using_bp_features(
                gf,
                gf,
                gf_parts_visibility,
                gf_parts_visibility,
                self.dist_combine_strat,
                self.batch_size_pairwise_dist_matrix,
                self.use_gpu,
                dist_metric,
            )
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        eval_metric = self.datamanager.test_loader[dataset_name][
            "query"
        ].dataset.eval_metric

        print("Computing CMC and mAP for eval Team Affiliation metric '{}' ...".format(eval_metric))
        # filter our non-players, i.e. team = -1
        f_distmat = distmat[q_teams != -1][:, g_teams != -1]
        f_q_teams = q_teams[q_teams != -1]
        f_g_teams = g_teams[g_teams != -1]
        f_q_camids = q_camids[q_teams != -1]
        f_g_camids = g_camids[g_teams != -1]
        eval_metrics = metrics.evaluate_rank(
            f_distmat,
            f_q_teams,
            f_g_teams,
            f_q_camids,
            f_g_camids,
            q_anns=None,
            g_anns=None,
            eval_metric=eval_metric,
            max_rank=np.array(ranks).max(),
        )

        mAP = eval_metrics["mAP"]
        cmc = eval_metrics["cmc"]
        print("** Results **")
        print("mAP: {:.2%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))

        print("Computing CMC and mAP for eval REID metric '{}' ...".format(eval_metric))
        eval_metrics = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            q_anns=q_anns,
            g_anns=g_anns,
            eval_metric=eval_metric,
            max_rank=np.array(ranks).max(),
        )

        mAP = eval_metrics["mAP"]
        cmc = eval_metrics["cmc"]
        print("** Results **")
        print("mAP: {:.2%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))

        for metric in eval_metrics.keys():
            if metric != "mAP" and metric != "cmc":
                val, size = eval_metrics[metric]
                if val is not None:
                    print("{:<20}: {:.2%} ({})".format(metric, val, size))
                else:
                    print("{:<20}: not provided".format(metric))

        # Parts ranking
        if self.detailed_ranking:
            self.display_individual_parts_ranking_performances(
                body_parts_distmat,
                cmc,
                g_camids,
                g_pids,
                mAP,
                q_camids,
                q_pids,
                eval_metric,
            )
        # TODO move below to writer
        plot_body_parts_pairs_distance_distribution(
            body_parts_distmat, q_pids, g_pids, "Query-gallery"
        )
        print(
            "Evaluate distribution of distances of pairs with same id vs different ids"
        )
        (
            same_ids_dist_mean,
            same_ids_dist_std,
            different_ids_dist_mean,
            different_ids_dist_std,
            ssmd,
        ) = plot_pairs_distance_distribution(
            distmat, q_pids, g_pids, "Query-gallery"
        )  # TODO separate ssmd from plot, put plot in writer
        print(
            "Positive pairs distance distribution mean: {:.3f}".format(
                same_ids_dist_mean
            )
        )
        print(
            "Positive pairs distance distribution standard deviation: {:.3f}".format(
                same_ids_dist_std
            )
        )
        print(
            "Negative pairs distance distribution mean: {:.3f}".format(
                different_ids_dist_mean
            )
        )
        print(
            "Negative pairs distance distribution standard deviation: {:.3f}".format(
                different_ids_dist_std
            )
        )
        print("SSMD = {:.4f}".format(ssmd))

        # if groundtruth target body masks are provided, compute part prediction accuracy
        avg_pxl_pred_accuracy = 0.0
        if (
            "mask" in q_anns
            and "mask" in g_anns
            and q_pxl_scores_ is not None
            and g_pxl_scores_ is not None
        ):
            q_pxl_pred_accuracy = self.compute_pixels_cls_accuracy(
                torch.from_numpy(q_anns["mask"]), q_pxl_scores_
            )
            g_pxl_pred_accuracy = self.compute_pixels_cls_accuracy(
                torch.from_numpy(g_anns["mask"]), g_pxl_scores_
            )
            avg_pxl_pred_accuracy = (
                q_pxl_pred_accuracy * len(q_parts_masks)
                + g_pxl_pred_accuracy * len(g_parts_masks)
            ) / (len(q_parts_masks) + len(g_parts_masks))
            print(
                "Pixel prediction accuracy for query = {:.2f}% and for gallery = {:.2f}% and on average = {:.2f}%".format(
                    q_pxl_pred_accuracy, g_pxl_pred_accuracy, avg_pxl_pred_accuracy
                )
            )

        ###################################################
        avg_pxl_pred_accuracy = 0.0
        q_pxl_pred_accuracy = self.compute_role_cls_accuracy(
                torch.from_numpy(q_roles), q_role_scores_
            )
        g_pxl_pred_accuracy = self.compute_role_cls_accuracy(
                torch.from_numpy(g_roles), g_role_scores_
            )
        avg_pxl_pred_accuracy = (
                q_pxl_pred_accuracy * q_roles.shape[0]
                + g_pxl_pred_accuracy * g_roles.shape[0]
            ) / (q_roles.shape[0] + g_roles.shape[0])
        print(
                "Role prediction accuracy for query = {:.2f}% and for gallery = {:.2f}% and on average = {:.2f}%".format(
                    100*q_pxl_pred_accuracy, 100*g_pxl_pred_accuracy, 100*avg_pxl_pred_accuracy
                )
            )
         ######################################################

        if visrank:
            self.writer.visualize_rank(
                self.datamanager.test_loader[dataset_name],
                dataset_name,
                distmat,
                save_dir,
                visrank_topk,
                visrank_q_idx_list,
                visrank_count,
                body_parts_distmat,
                qf_parts_visibility,
                gf_parts_visibility,
                q_parts_masks,
                g_parts_masks,
                mAP,
                cmc[0],
            )

        self.writer.visualize_embeddings(
            qf,
            gf,
            q_pids,
            g_pids,
            self.datamanager.test_loader[dataset_name],
            dataset_name,
            qf_parts_visibility,
            gf_parts_visibility,
            mAP,
            cmc[0],
        )
        self.writer.performance_evaluation_timer.stop()
        return cmc, mAP, ssmd, avg_pxl_pred_accuracy

    def compute_pixels_cls_accuracy(self, target_masks, pixels_cls_scores):
        if pixels_cls_scores.is_cuda:
            target_masks = target_masks.cuda()
        target_masks = nn.functional.interpolate(
            target_masks,
            pixels_cls_scores.shape[2::],
            mode="bilinear",
            align_corners=True,
        )  # Best perf with bilinear here and nearest in resize transform
        pixels_cls_score_targets = target_masks.argmax(dim=1)  # [N, Hf, Wf]
        pixels_cls_score_targets = pixels_cls_score_targets.flatten()  # [N*Hf*Wf]
        pixels_cls_scores = pixels_cls_scores.permute(0, 2, 3, 1).flatten(
            0, 2
        )  # [N*Hf*Wf, M]
        accuracy = metrics.accuracy(pixels_cls_scores, pixels_cls_score_targets)[0]
        return accuracy.item()

    def compute_role_cls_accuracy(self, target_masks, pixels_cls_scores):
        if pixels_cls_scores.is_cuda:
            target_masks = target_masks.cuda()
        pixels_cls_scores = pixels_cls_scores.argmax(dim=1)
        accuracy = (pixels_cls_scores == target_masks).sum()/target_masks.shape[0]
        return accuracy

    def display_individual_parts_ranking_performances(
        self,
        body_parts_distmat,
        cmc,
        g_camids,
        g_pids,
        mAP,
        q_camids,
        q_pids,
        eval_metric,
    ):
        print("Parts embeddings individual rankings :")
        bp_offset = 0
        if GLOBAL in self.config.model.bpbreid.test_embeddings:
            bp_offset += 1
        if FOREGROUND in self.config.model.bpbreid.test_embeddings:
            bp_offset += 1
        table = []
        for bp in range(
            0, body_parts_distmat.shape[0]
        ):  # TODO DO NOT TAKE INTO ACCOUNT -1 DISTANCES!!!!
            perf_metrics = metrics.evaluate_rank(
                body_parts_distmat[bp],
                q_pids,
                g_pids,
                q_camids,
                g_camids,
                eval_metric=eval_metric,
                max_rank=10,
            )
            title = "p {}".format(bp - bp_offset)
            if bp < bp_offset:
                if bp == 0:
                    if GLOBAL in self.config.model.bpbreid.test_embeddings:
                        title = GLOBAL
                    else:
                        title = FOREGROUND
                if bp == 1:
                    title = FOREGROUND
            mAP = perf_metrics["mAP"]
            cmc = perf_metrics["cmc"]
            table.append([title, mAP, cmc[0], cmc[4], cmc[9]])
        headers = ["embed", "mAP", "R-1", "R-5", "R-10"]
        print(tabulate(table, headers, tablefmt="fancy_grid", floatfmt=".3f"))

    def parse_data_for_train(self, data):
        imgs = data["image"]
        imgs_path = data["img_path"]
        masks = data["mask"] if "mask" in data else None
        pids = data["pid"]
        teams = data["team"]
        roles = data["role"]

        if self.use_gpu:
            imgs = imgs.cuda()
            if masks is not None:
                masks = masks.cuda()
            pids = pids.cuda()
            teams = teams.cuda()
            roles = roles.cuda()

        if masks is not None:
            assert masks.shape[1] == (
                self.config.model.bpbreid.masks.parts_num + 1
            ), f"masks.shape[1] ({masks.shape[1]}) != parts_num ({self.config.model.bpbreid.masks.parts_num})"

        return imgs, masks, pids, teams, roles, imgs_path

    def parse_data_for_eval(self, data):
        imgs = data["image"]
        masks = data["mask"] if "mask" in data else None
        pids = data["pid"]
        camids = data["camid"]
        teams = data["team"]
        roles = data["role"]
        game_ids = data["video_id"]

        return imgs, masks, pids, teams, camids, game_ids, roles
