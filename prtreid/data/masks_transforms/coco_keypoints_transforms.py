import numpy as np

from prtreid.data.masks_transforms.mask_transform import MaskGroupingTransform

COCO_KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                  "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
                  "right_knee", "left_ankle", "right_ankle"]

COCO_KEYPOINTS_MAP = {k: i for i, k in enumerate(COCO_KEYPOINTS)}

COCO_JOINTS = [
    'head',
    'torso',
    'right_upperarm',
    'left_upperarm',
    'right_forearm',
    'left_forearm',
    'right_femur',
    'left_femur',
    'right_tibia',
    'left_tibia',
]

COCO_JOINTS_MAP = {k: i for i, k in enumerate(COCO_JOINTS)}


class CocoToSixBodyMasks(MaskGroupingTransform):
    parts_grouping = {
        "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
        "torso": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        "left_arm": ["left_elbow", "left_wrist"],
        "right_arm": ["right_elbow", "right_wrist"],
        "left_leg": ["left_knee", "left_ankle"],
        "right_leg": ["right_knee", "right_ankle"],
    }

    def __init__(self):
        super().__init__(self.parts_grouping, COCO_KEYPOINTS_MAP)


class CocoJointsToSixBodyMasks(MaskGroupingTransform):
    parts_grouping = {
        "head": ["head"],
        "torso": ["torso"],
        "left_arm": ["left_upperarm", "left_forearm"],
        "right_arm": ["right_upperarm", "right_forearm"],
        "left_leg": ["left_femur", "left_tibia"],
        "right_leg": ["right_femur", "right_tibia"],
    }

    def coco_joints_to_body_part_visibility_scores(self, coco_joints_visibility_scores):
        visibility_scores = []
        for i, part in enumerate(self.parts_names):
            visibility_scores.append(coco_joints_visibility_scores[[self.parts_map[k] for k in self.parts_grouping[part]]].mean())
        return np.array(visibility_scores)

    def __init__(self):
        super().__init__(self.parts_grouping, COCO_JOINTS_MAP)
