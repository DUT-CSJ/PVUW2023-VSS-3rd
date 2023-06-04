# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ADE20KDataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = (
        'wall', 'ceiling', 'door', 'stair', 'ladder', 'escalator', 'Playground_slide', 'handrail_or_fence', 'window',
        'rail', 'goal', 'pillar', 'pole', 'floor', 'ground', 'grass', 'sand', 'athletic_field', 'road',
         'path', 'crosswalk', 'building', 'house', 'bridge', 'tower', 'windmill', 'well_or_well_lid',
         'other_construction', 'sky', 'mountain', 'stone', 'wood', 'ice', 'snowfield', 'grandstand', 'sea', 'river',
         'lake', 'waterfall', 'water', 'billboard_or_Bulletin_Board', 'sculpture', 'pipeline', 'flag',
         'parasol_or_umbrella', 'cushion_or_carpet', 'tent', 'roadblock', 'car', 'bus', 'truck', 'bicycle',
         'motorcycle', 'wheeled_machine', 'ship_or_boat', 'raft', 'airplane', 'tyre', 'traffic_light', 'lamp', 'person',
         'cat', 'dog', 'horse', 'cattle', 'other_animal', 'tree', 'flower', 'other_plant', 'toy', 'ball_net',
         'backboard', 'skateboard', 'bat', 'ball', 'cupboard_or_showcase_or_storage_rack', 'box',
         'traveling_case_or_trolley_case', 'basket', 'bag_or_package', 'trash_can', 'cage', 'plate',
         'tub_or_bowl_or_pot', 'bottle_or_cup', 'barrel', 'fishbowl', 'bed', 'pillow', 'table_or_desk', 'chair_or_seat',
         'bench', 'sofa', 'shelf', 'bathtub', 'gun', 'commode', 'roaster', 'other_machine', 'refrigerator',
         'washing_machine', 'Microwave_oven', 'fan', 'curtain', 'textiles', 'clothes', 'painting_or_poster', 'mirror',
         'flower_pot_or_vase', 'clock', 'book', 'tool', 'blackboard', 'tissue', 'screen_or_television', 'computer',
         'printer', 'Mobile_phone', 'keyboard', 'other_electronic_product', 'fruit', 'food', 'instrument', 'train'
    )

    PALETTE = [[128, 0, 0],
               [0, 128, 0],
               [128, 128, 0],
               [0, 0, 128],
               [128, 0, 128],
               [0, 128, 128],
               [128, 128, 128],
               [64, 0, 0],
               [191, 0, 0],
               [64, 128, 0],
               [191, 128, 0],
               [64, 0, 128],
               [191, 0, 128],
               [64, 128, 128],
               [191, 128, 128],
               [0, 64, 0],
               [128, 64, 0],
               [0, 191, 0],
               [128, 191, 0],
               [0, 64, 128],
               [128, 64, 128],
               [22, 22, 22],
               [23, 23, 23],
               [24, 24, 24],
               [25, 25, 25],
               [26, 26, 26],
               [27, 27, 27],
               [28, 28, 28],
               [29, 29, 29],
               [30, 30, 30],
               [31, 31, 31],
               [32, 32, 32],
               [33, 33, 33],
               [34, 34, 34],
               [35, 35, 35],
               [36, 36, 36],
               [37, 37, 37],
               [38, 38, 38],
               [39, 39, 39],
               [40, 40, 40],
               [41, 41, 41],
               [42, 42, 42],
               [43, 43, 43],
               [44, 44, 44],
               [45, 45, 45],
               [46, 46, 46],
               [47, 47, 47],
               [48, 48, 48],
               [49, 49, 49],
               [50, 50, 50],
               [51, 51, 51],
               [52, 52, 52],
               [53, 53, 53],
               [54, 54, 54],
               [55, 55, 55],
               [56, 56, 56],
               [57, 57, 57],
               [58, 58, 58],
               [59, 59, 59],
               [60, 60, 60],
               [61, 61, 61],
               [62, 62, 62],
               [63, 63, 63],
               [64, 64, 64],
               [65, 65, 65],
               [66, 66, 66],
               [67, 67, 67],
               [68, 68, 68],
               [69, 69, 69],
               [70, 70, 70],
               [71, 71, 71],
               [72, 72, 72],
               [73, 73, 73],
               [74, 74, 74],
               [75, 75, 75],
               [76, 76, 76],
               [77, 77, 77],
               [78, 78, 78],
               [79, 79, 79],
               [80, 80, 80],
               [81, 81, 81],
               [82, 82, 82],
               [83, 83, 83],
               [84, 84, 84],
               [85, 85, 85],
               [86, 86, 86],
               [87, 87, 87],
               [88, 88, 88],
               [89, 89, 89],
               [90, 90, 90],
               [91, 91, 91],
               [92, 92, 92],
               [93, 93, 93],
               [94, 94, 94],
               [95, 95, 95],
               [96, 96, 96],
               [97, 97, 97],
               [98, 98, 98],
               [99, 99, 99],
               [100, 100, 100],
               [101, 101, 101],
               [102, 102, 102],
               [103, 103, 103],
               [104, 104, 104],
               [105, 105, 105],
               [106, 106, 106],
               [107, 107, 107],
               [108, 108, 108],
               [109, 109, 109],
               [110, 110, 110],
               [111, 111, 111],
               [112, 112, 112],
               [113, 113, 113],
               [114, 114, 114],
               [115, 115, 115],
               [116, 116, 116],
               [117, 117, 117],
               [118, 118, 118],
               [119, 119, 119],
               [120, 120, 120],
               [121, 121, 121],
               [122, 122, 122],
               [123, 123, 123],
               [124, 124, 124]]

    def __init__(self, **kwargs):
        super(ADE20KDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)
        self._palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0,
                0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128, 191, 0, 128, 64, 128, 128, 191, 128, 128, 0, 64, 0,
                128, 64, 0, 0, 191, 0, 128, 191, 0, 0, 64, 128, 128, 64, 128, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25,
                25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33,
                34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42,
                42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51,
                51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59,
                60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68,
                68, 69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77,
                77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85,
                86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94,
                94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102, 102,
                102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109,
                109, 110, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116,
                116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123,
                123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130,
                130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136, 137, 137,
                137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143, 144, 144,
                144, 145, 145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 149, 149, 149, 150, 150, 150, 151, 151,
                151, 152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156, 157, 157, 157, 158, 158,
                158, 159, 159, 159, 160, 160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165,
                165, 166, 166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171, 172, 172,
                172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178, 179, 179,
                179, 180, 180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186, 186,
                186, 187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191, 192, 192, 192, 193, 193,
                193, 194, 194, 194, 195, 195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200,
                200, 201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206, 207, 207,
                207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213, 214, 214,
                214, 215, 215, 215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221,
                221, 222, 222, 222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226, 227, 227, 227, 228, 228,
                228, 229, 229, 229, 230, 230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235,
                235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 242, 242,
                242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249,
                249, 250, 250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255]

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            # result = result + 1

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            output.putpalette(self._palette)
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)
        return result_files
