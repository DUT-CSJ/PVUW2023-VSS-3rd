# SwinTransformer
python tools/train.py  ./configs/swin/vizwiz.py  --work-dir ./sod_dirs/swint
python tools/test.py ./configs/swin/vizwiz.py ./sod_dirs/swint/latest.pth --format-only --tmpoutdir ./sod_dirs/swint_test

# ConvNext
python tools/train.py  ./configs/convnext/vizwiz.py  --work-dir ./sod_dirs/convnextt
python tools/test.py ./configs/convnext/vizwiz.py ./sod_dirs/convnextt/latest.pth --format-only --tmpoutdir ./sod_dirs/convnextt_test
python tools/test.py ./configs/convnext/vizwizs.py ./sod_dirs/convnexts/latest.pth --format-only --tmpoutdir ./sod_dirs/convnexts_test

# Internimage
python tools/train.py  ./configs/internimage/vizwizt.py  --work-dir ./sod_dirs/internimaget
python tools/train.py  ./configs/internimage/vizwizt_mask2former.py  --work-dir ./sod_dirs/internimaget_mask2former
python tools/test.py ./configs/internimage/vizwizt.py ./sod_dirs/internimaget/latest.pth --format-only --tmpoutdir ./sod_dirs/inference/internimaget_flip
python tools/test.py ./configs/internimage/vizwizs.py ./sod_dirs/internimages/latest.pth

# ensemble
python tools/sod_ensemble.py \
  --config ./configs/internimage/vizwizt.py ./configs/convnext/vizwizs.py \
  --checkpoint ./sod_dirs/internimaget/latest.pth ./sod_dirs/convnexts/latest.pth \
  --out ./sod_dirs/ensemble \
  --gpus 0\
