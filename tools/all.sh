
bash tools/dist_train.sh  ./configs/swin/upernet_swinl.py 2 --work-dir ./work_dirs/swin --load-from ./checkpoint/upernet_swin_large.pth
bash tools/dist_train.sh  ./configs/convnext/upernet_convnextl.py 2 --work-dir ./work_dirs/convnext --load-from ./checkpoint/upernet_convnext_large.pth
bash tools/dist_train.sh  ./configs/beit/upernet_beit.py 2 --work-dir ./work_dirs/beit --load-from ./checkpoint/upernet_beit_base.pth
python tools/test.py ./configs/convnext/upernet_convnextl.py /home/data1/mmVSPW/work_dirs/convnext2a100/iter_25000.pth --format-only
python tools/test.py ./configs/swin/upernet_swinl.py /home/data1/mmVSPW/work_dirs/swin2a1002/ensemble.pth --format-only --tmpoutdir ./work_dirs/weightaverage

python tools/model_ensemble.py \
  --config ./configs/swin/upernet_swinl.py ./configs/convnext/upernet_convnextl.py ./configs/beit/upernet_beit.py \
  --checkpoint ./work_dirs/swin2a100/iter_16000.pth ./work_dirs/convnext2a100/iter_25000.pth work_dirs/beit1a100/iter_16500.pth\
  --aug-test \
  --out ./work_dirs/ensembletest\
  --gpus 0\

/home/dut/anaconda3/envs/openmmlab/bin/python tools/model_ensemble.py \
  --config ./configs/swin/upernet_swinl.py ./configs/convnext/upernet_convnextxl.py \
  --checkpoint ./work_dirs/swin2a100/iter_16000.pth ./work_dirs/convnexta100/latest.pth \
  --out ./work_dirs/test \
  --gpus 0\

/home/dut/anaconda3/envs/vspw/bin/python tools/model_ensemble.py \
  --config ./configs/swin/upernet_swinl.py ./configs/convnext/upernet_convnextl.py \
  --checkpoint ./work_dirs/swin2a1002/latest.pth ./work_dirs/convnext2a100/latest.pth \
  --out ./work_dirs/ensembletest\
  --gpus 0\


CUDA_VISIBLE_DEVICES=3 /home/dut/anaconda3/envs/vspw/bin/python3.8 tools/test.py ./configs/internimage/mask2former_internimageh.py ./work_dirs/mask2former/latest.pth --tmpoutdir ./work_dirs/mask2former190ktta --format-only --aug-test
/home/dut/anaconda3/envs/vspw/bin/python tools/test.py ./configs/internimage/upernet_internimagexl.py ./checkpoint/internimage.pth --format-only --tmpoutdir ./work_dirs/internimage
/home/dut/anaconda3/envs/vspw/bin/python tools/test.py ./configs/internimage/upernet_internimageh.py ./work_dirs/internimage/iter48000.pth --format-only --tmpoutdir ./work_dirs/internimageH
CUDA_VISIBLE_DEVICES=4 /home/dut/anaconda3/envs/vspw/bin/python3.8 tools/test.py ./configs/internimage/upernet_internimageh.py ./work_dirs/internimage/iter60000.pth --aug-test --format-only --tmpoutdir ./work_dirs/internimageH60ktta
bash tools/dist_test.sh ./configs/internimage/upernet_internimageh.py ./work_dirs/internimage/iter48000.pth --tmpoutdir ./work_dirs/internimageH48k
CUDA_VISIBLE_DEVICES=2,3 bash tools/dist_test.sh ./configs/internimage/mask2former_internimageh.py ./work_dirs/mask2former/best_mIoU_iter_62000.pth 2 --tmpoutdir ./work_dirs/mask2former62k --format-only
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh ./configs/internimage/upernet_coco.py 3 --work-dir ./work_dirs/coco_pretrain
CUDA_VISIBLE_DEVICES=2,3 /home/dut/anaconda3/envs/vspw/bin/python3.8 tools/model_ensemble.py \
  --config ./configs/internimage/mask2former_internimageh.py ./configs/internimage/upernet_internimageh.py \
  --checkpoint ./work_dirs/mask2former/latest.pth ./work_dirs/internimage/iter160000.pth \
  --aug-test \
  --out ./work_dirs/ensembletta \
  --gpus 1 \
python tools/dist_test.sh ./configs/internimage/mask2former_internimageh.py ./work_dirs/internimage/best_mIoU_iter_62000.pth --tmpoutdir ./work_dirs/mask2former62k --format-only

CUDA_VISIBLE_DEVICES=4 /home/dut/anaconda3/envs/vspw/bin/python3.8 tools/model_ensemble.py \
  --config ./configs/internimage/upernet_internimageh.py ./configs/swin/upernet_swinl.py \
  --checkpoint ./work_dirs/internimage/iter60000.pth ./work_dirs/swin2a1002/latest.pth \
  --out ./work_dirs/ensemble \
  --gpus 0 \

/home/dut/anaconda3/envs/vspw/bin/python tools/model_ensemble.py \
  --config ./configs/convnext/upernet_convnextxl.py ./configs/swin/upernet_swinl.py ./configs/internimage/upernet_internimagexl.py \
  --checkpoint ./work_dirs/convnexta100/latest.pth ./work_dirs/swin2a1002/latest.pth ./checkpoint/internimage.pth \
  --out ./work_dirs/enselblesingle \
  --gpus 0 \


python tools/model_ensemble.py \
  --config ./configs/convnext/upernet_convnextxl.py ./configs/swin/upernet_swinl.py ./configs/internimage/upernet_internimagexl.py \
  --checkpoint ./work_dirs/convnexta100/latest.pth ./work_dirs/swin2a1002/latest.pth ./checkpoint/internimage.pth \
  --aug-test \
  --out ./work_dirs/ensembletta \
  --gpus 0 \

CUDA_VISIBLE_DEVICES=1,2,3 python3.8 tools/model_ensemble.py \
  --config ./configs/internimage/mask2former_internimageh.py ./configs/internimage/mask2former_internimageh.py \
  --checkpoint ./work_dirs/mask2former/best_mIoU_iter_62000.pth ./work_dirs/mask2formercoco/iter_12500.pth \
  --out ./work_dirs/ensemblemask2former \
  --gpus 1,2,3 \

CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh ./configs/internimage/mask2former_internimageh.py  ./work_dirs/mask2formercoco/iter_12500.pth --format-only --tmpoutdir ./work_dirs/mask2formercoco12k
CUDA_VISIBLE_DEVICES=1 python tools/train.py ./configs/internimage/mask2former_internimageh.py --work-dir ./work_dirs/mask2formerval --load-from ./work_dirs/mask2former/iter_12500.pth
