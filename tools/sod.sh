# SwinTransformer
python tools/train.py  ./configs/swin/vizwiz.py  --work-dir ./sod_dirs/swint
python tools/test.py ./configs/swin/vizwiz.py ./sod_dirs/swint/latest.pth --format-only --tmpoutdir ./sod_dirs/swint_test
python tools/test.py ./configs/swin/vizwizb.py ./sod_dirs/swinb384val/latest.pth --format-only --tmpoutdir ./sod_dirs/swinb384val

# ConvNext
python tools/train.py  ./configs/convnext/vizwiz.py  --work-dir ./sod_dirs/convnextt
python tools/test.py ./configs/convnext/vizwiz.py ./sod_dirs/convnextt/latest.pth --format-only --tmpoutdir ./sod_dirs/convnextt_test
python tools/test.py ./configs/convnext/vizwizs.py ./sod_dirs/convnexts/latest.pth --format-only --tmpoutdir ./sod_dirs/convnexts_test
python tools/test.py ./configs/convnext/vizwizb.py ./sod_dirs/convnextb384val/latest.pth --format-only --tmpoutdir ./sod_dirs/inference/convnextb384val

# Internimage
python tools/train.py  ./configs/internimage/vizwizt.py  --work-dir ./sod_dirs/internimaget512val
python tools/train.py  ./configs/internimage/vizwizt_mask2former.py  --work-dir ./sod_dirs/internimaget_mask2former
python tools/test.py ./configs/internimage/vizwizt.py ./sod_dirs/internimaget_load/latest.pth --format-only --tmpoutdir ./sod_dirs/inference/internimaget_load
python tools/test.py ./configs/internimage/vizwizt.py ./sod_dirs/internimaget512val/latest.pth --format-only --tmpoutdir ./sod_dirs/inference/internimaget512val
python tools/test.py ./configs/internimage/vizwizb.py ./sod_dirs/internimageb384val/latest.pth --format-only --tmpoutdir ./sod_dirs/inference/internimageb384val60k

# beit
python tools/train.py ./configs/beit/vizwizb.py  --work-dir ./sod_dirs/beitb

# ensemble
python tools/sod_ensemble.py \
  --config ./configs/internimage/vizwizt.py ./configs/convnext/vizwizs.py \
  --checkpoint ./sod_dirs/internimaget/latest.pth ./sod_dirs/convnexts/latest.pth \
  --out ./sod_dirs/ensemble \
  --gpus 0\

python tools/sod_ensemble.py \
  --config ./configs/internimage/vizwizb.py ./configs/convnext/vizwizb.py ./configs/swin/vizwizb.py  \
  --checkpoint ./sod_dirs/internimageb384val/latest.pth ./sod_dirs/convnextb384val/latest.pth ./sod_dirs/swinb384val/latest.pth \
  --out ./sod_dirs/inference/ensemble3model \
  --gpus 0\
