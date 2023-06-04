# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv.runner import load_checkpoint, wrap_fp16_model
from PIL import Image

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


@torch.no_grad()
def main(args):
    _palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0,
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
    models = []
    gpu_ids = args.gpus
    configs = args.config
    ckpts = args.checkpoint

    cfg = mmcv.Config.fromfile(configs[0])

    if args.aug_test:
        cfg.data.test.pipeline[1].img_ratios = [
            1.0# 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
        ]
        cfg.data.test.pipeline[1].flip = True
    else:
        cfg.data.test.pipeline[1].img_ratios = [1.0]
        cfg.data.test.pipeline[1].flip = False

    torch.backends.cudnn.benchmark = True

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        dist=False,
        shuffle=False,
    )

    for idx, (config, ckpt) in enumerate(zip(configs, ckpts)):
        cfg = mmcv.Config.fromfile(config)
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True

        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        if cfg.get('fp16', None):
            wrap_fp16_model(model)
        load_checkpoint(model, ckpt, map_location='cpu')
        torch.cuda.empty_cache()
        tmpdir = args.out
        mmcv.mkdir_or_exist(tmpdir)
        model = MMDataParallel(model, device_ids=[gpu_ids[idx % len(gpu_ids)]])
        model.eval()
        models.append(model)

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler
    for batch_indices, data in zip(loader_indices, data_loader):
        result = []

        for model in models:
            x, _ = scatter_kwargs(
                inputs=data, kwargs=None, target_gpus=model.device_ids)
            if args.aug_test:
                logits = model.module.aug_test_logits(**x[0])
            else:
                logits = model.module.simple_test_logits(**x[0])
            result.append(logits)

        result_logits = 0
        for logit in result:
            result_logits += logit
        # for i in range(len(result)):
        #     if i == 0:
        #         result_logits += result[i]
        #     else:
        #         result_logits += result[i] * 0.5

        pred = result_logits.argmax(axis=1).squeeze()
        img_info = dataset.img_infos[batch_indices[0]]
        file_name = os.path.join(
            tmpdir, img_info['ann']['seg_map'].split(os.path.sep)[-1])
        # Image.fromarray(pred.astype(np.uint8)).save(file_name)
        pred = Image.fromarray(pred.astype(np.uint8)).convert('P')
        pred.putpalette(_palette)
        pred.save(file_name)
        prog_bar.update()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Model Ensemble with logits result')
    parser.add_argument(
        '--config', type=str, nargs='+', help='ensemble config files path')
    parser.add_argument(
        '--checkpoint',
        type=str,
        nargs='+',
        help='ensemble checkpoint files path')
    parser.add_argument(
        '--aug-test',
        action='store_true',
        help='control ensemble aug-result or single-result (default)')
    parser.add_argument(
        '--out', type=str, default='results', help='the dir to save result')
    parser.add_argument(
        '--gpus', type=int, nargs='+', default=[0], help='id of gpu to use')

    args = parser.parse_args()
    assert len(args.config) == len(args.checkpoint), \
        f'len(config) must equal len(checkpoint), ' \
        f'but len(config) = {len(args.config)} and' \
        f'len(checkpoint) = {len(args.checkpoint)}'
    assert args.out, "ensemble result out-dir can't be None"
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
