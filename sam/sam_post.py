import torch
import mmcv
import cv2
import os
from PIL import Image
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pycocotools.mask as maskUtils
from tqdm import tqdm
# from mmdet.core.visualization.image import imshow_det_bboxes

from vspw_id2label import CONFIG as id2label


class SSA:
    def __init__(self):
        sam = sam_model_registry["vit_h"](checkpoint='/home/csj/desk2t/Code/segment-anything/checkpoint/sam_vit_h_4b8939.pth').cuda()
        self.mask_branch_model = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=64,
            # Foggy driving (zero-shot evaluate) is more challenging than other dataset, so we use a larger points_per_side
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
            output_mode='coco_rle',
        )
        print('[Model loaded] Mask branch (SAM) is loaded.')
        self._palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
                         64, 0,
                         0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128, 191, 0, 128, 64, 128, 128, 191, 128, 128, 0,
                         64, 0,
                         128, 64, 0, 0, 191, 0, 128, 191, 0, 0, 64, 128, 128, 64, 128, 22, 22, 22, 23, 23, 23, 24, 24,
                         24, 25,
                         25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33,
                         33, 33,
                         34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41,
                         42, 42,
                         42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50,
                         50, 51,
                         51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58, 58, 58, 59,
                         59, 59,
                         60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67,
                         68, 68,
                         68, 69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76,
                         76, 77,
                         77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85,
                         85, 85,
                         86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93,
                         94, 94,
                         94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101,
                         102, 102,
                         102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108,
                         109, 109,
                         109, 110, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115,
                         116, 116,
                         116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122,
                         123, 123,
                         123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129,
                         130, 130,
                         130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136,
                         137, 137,
                         137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143,
                         144, 144,
                         144, 145, 145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 149, 149, 149, 150, 150, 150,
                         151, 151,
                         151, 152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156, 157, 157, 157,
                         158, 158,
                         158, 159, 159, 159, 160, 160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 164, 164, 164,
                         165, 165,
                         165, 166, 166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171,
                         172, 172,
                         172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178,
                         179, 179,
                         179, 180, 180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185,
                         186, 186,
                         186, 187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191, 192, 192, 192,
                         193, 193,
                         193, 194, 194, 194, 195, 195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199,
                         200, 200,
                         200, 201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206,
                         207, 207,
                         207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213,
                         214, 214,
                         214, 215, 215, 215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220,
                         221, 221,
                         221, 222, 222, 222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226, 227, 227, 227,
                         228, 228,
                         228, 229, 229, 229, 230, 230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234,
                         235, 235,
                         235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241,
                         242, 242,
                         242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248,
                         249, 249,
                         249, 250, 250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255]

    def vspw_post_processing(self, img_path=None, output_path=None, prediction_path=None):
        #  class_id generate
        prediction = np.array(Image.open(prediction_path).convert('P'))
        class_ids = torch.from_numpy(prediction).cuda()
        semantc_mask = class_ids.clone()

        #  SAM mask generate
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        anns = {'annotations': self.mask_branch_model.generate(img)}
        anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)

        #  processing anns
        class_names = []
        for ann in anns['annotations']:
            valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
            # get the class ids of the valid pixels
            propose_classes_ids = class_ids[valid_mask]
            num_class_proposals = len(torch.unique(propose_classes_ids))
            if num_class_proposals == 1:
                semantc_mask[valid_mask] = propose_classes_ids[0]
                ann['class_name'] = id2label['id2label'][str(propose_classes_ids[0].item())]
                ann['class_proposals'] = id2label['id2label'][str(propose_classes_ids[0].item())]
                class_names.append(ann['class_name'])
                # bitmasks.append(maskUtils.decode(ann['segmentation']))
                continue
            top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
            top_1_propose_class_names = [id2label['id2label'][str(class_id.item())] for class_id in
                                         top_1_propose_class_ids]

            semantc_mask[valid_mask] = top_1_propose_class_ids.byte()
            ann['class_name'] = top_1_propose_class_names[0]
            ann['class_proposals'] = top_1_propose_class_names[0]
            class_names.append(ann['class_name'])
            # bitmasks.append(maskUtils.decode(ann['segmentation']))

            del valid_mask
            del propose_classes_ids
            del num_class_proposals
            del top_1_propose_class_ids
            del top_1_propose_class_names

        sematic_class_in_img = torch.unique(semantc_mask)
        semantic_bitmasks, semantic_class_names = [], []

        # semantic prediction
        anns['semantic_mask'] = {}
        for i in range(len(sematic_class_in_img)):
            class_name = id2label['id2label'][str(sematic_class_in_img[i].item())]
            class_mask = semantc_mask == sematic_class_in_img[i]
            class_mask = class_mask.cpu().numpy().astype(np.uint8)
            semantic_class_names.append(class_name)
            semantic_bitmasks.append(class_mask)
            anns['semantic_mask'][str(sematic_class_in_img[i].item())] = maskUtils.encode(
                np.array((semantc_mask == sematic_class_in_img[i]).cpu().numpy(), order='F', dtype=np.uint8))
            anns['semantic_mask'][str(sematic_class_in_img[i].item())]['counts'] = \
                anns['semantic_mask'][str(sematic_class_in_img[i].item())]['counts'].decode('utf-8')

        final_predition = np.zeros((h, w))
        sematic_class_in_img = sematic_class_in_img.cpu().numpy()
        for i in range(len(semantic_bitmasks)):
            final_predition = final_predition + semantic_bitmasks[i] * sematic_class_in_img[i]

        output = Image.fromarray(final_predition.astype(np.uint8)).convert('P')
        output.putpalette(self._palette)
        output.save(output_path)

        # imshow_det_bboxes(img,
        #                   bboxes=None,
        #                   labels=np.arange(len(sematic_class_in_img)),
        #                   segms=np.stack(semantic_bitmasks),
        #                   class_names=semantic_class_names,
        #                   font_size=25,
        #                   show=False,
        #                   out_file=os.path.join(output_path))
        # print('[Save] save SSA prediction: ', os.path.join(output_path))
        # mmcv.dump(anns, os.path.join(output_path, filename + '_semantic.json'))
        del img
        del anns
        del class_ids
        del semantc_mask
        # del bitmasks
        del class_names
        del semantic_bitmasks
        del semantic_class_names


images_path = '/home/csj/desk2t/Code/mmVSPW/datasets/images/test'
predictions_path = '/home/csj/desk2t/Code/mmVSPW/work_dirs/mask2former5175'
output_path = '/home/csj/desk2t/Code/mmVSPW/work_dirs/sam2'
images = os.listdir(images_path)
images = sorted(images)
predictions = os.listdir(predictions_path)
predictions = sorted(predictions)
model = SSA()
for i in tqdm(range(len(images))[26367:]):#[3016:9589]
    img_path = os.path.join(images_path, images[i])
    pre_path = os.path.join(predictions_path, predictions[i])
    out_path = os.path.join(output_path, predictions[i])
    model.vspw_post_processing(img_path, out_path, pre_path)
