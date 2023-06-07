import os
import shutil
import time

def dataset_process(source_path, target_path):
    # with open(os.path.join(source_path, 'train.txt'), 'r') as f:
    #     train_samples = f.read().splitlines()
    #
    # with open(os.path.join(source_path, 'val.txt'), 'r') as f:
    #     val_samples = f.read().splitlines()

    with open(os.path.join(source_path, 'test.txt'), 'r') as f:
        test_samples = f.read().splitlines()

    os.makedirs(os.path.join(target_path, 'annotations', 'training'), exist_ok=True)
    os.makedirs(os.path.join(target_path, 'annotations', 'validation'), exist_ok=True)
    os.makedirs(os.path.join(target_path, 'images', 'training'), exist_ok=True)
    os.makedirs(os.path.join(target_path, 'images', 'validation'), exist_ok=True)
    os.makedirs(os.path.join(target_path, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(target_path, 'annotations', 'test'), exist_ok=True)


    video_clips = os.listdir(os.path.join(source_path, 'data'))
    for video in video_clips:
        start = time.time()
    #     if video in train_samples:
    #         masks = os.listdir(os.path.join(source_path, 'data', video, 'mask'))
    #         for mask in masks:
    #             name = video + mask
    #             shutil.copy2(os.path.join(source_path, 'data', video, 'mask', mask),
    #                          os.path.join(target_path, 'annotations', 'training', name))
    #         images = os.listdir(os.path.join(source_path, 'data', video, 'origin'))
    #         for image in images:
    #             name = video + image
    #             shutil.copy2(os.path.join(source_path, 'data', video, 'origin', image),
    #                          os.path.join(target_path, 'images', 'training', name))
    #
    #     if video in val_samples:
    #         masks = os.listdir(os.path.join(source_path, 'data', video, 'mask'))
    #         for mask in masks:
    #             name = video + mask
    #             shutil.copy2(os.path.join(source_path, 'data', video, 'mask', mask),
    #                          os.path.join(target_path, 'annotations', 'validation', name))
    #         images = os.listdir(os.path.join(source_path, 'data', video, 'origin'))
    #         for image in images:
    #             name = video + image
    #             shutil.copy2(os.path.join(source_path, 'data', video, 'origin', image),
    #                          os.path.join(target_path, 'images', 'validation', name))

        if video in test_samples:
            images = os.listdir(os.path.join(source_path, 'data', video, 'origin'))
            for image in images:
                name = video + '--' + image
                shutil.copy2(os.path.join(source_path, 'data', video, 'origin', image),
                             os.path.join(target_path, 'images', 'test', name))

            masks = os.listdir(os.path.join('/media/vos/Data/hzq/mmsegmentation/datasets/result_submission', video))
            for mask in masks:
                name = video + '--' + mask
                shutil.copy2(os.path.join('/media/vos/Data/hzq/mmsegmentation/datasets/result_submission', video, mask),
                             os.path.join(target_path, 'annotations', 'test', name))

        end = time.time()
        print("一次循环时间为：%.6f秒", (end - start))

def result_process(result_path, video_path):
    os.makedirs(video_path, exist_ok=True)
    results = os.listdir(result_path)
    i = 0
    for res in results:
        index = res.find('--')
        video = res[:index]
        frame = res[index + 2 :]
        os.makedirs(os.path.join(video_path, video), exist_ok=True)
        shutil.copy2(os.path.join(result_path, res), os.path.join(video_path, video, frame))
        print(i)
        print('\n')
        i += 1


if __name__ == '__main__':
    vspw_path = '/media/vos/Data/hzq/datasets/VSPW_480p'
    mmseg_path = '/media/vos/Data/hzq/mmsegmentation/datasets'
    result_path = '/home/csj/desk2t/Code/mmVSPW/work_dirs/mask2former5175'
    video_path = '/home/csj/desk2t/Code/mmVSPW/work_dirs/result_submission'
    # dataset_process(vspw_path, mmseg_path)
    result_process(result_path, video_path)