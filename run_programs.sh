#!/bin/bash

nohup python -m training.main_test \
    --train-data "./Body_Parts_XRay/train_square.csv" \
    --val-data "./Body_Parts_XRay/val_square.csv" \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --warmup 1000 \
    --logs "/mnt/g/Logtemp/open_clip/Body_Parts_XRay" \
    --batch-size 32 \
    --aug-cfg "augment_dir='/home/yilu/Development/open_clip/Body_Parts_XRay/'" \
    --lr 2e-6 \
    --wd 0.1 \
    --epochs 300 \
    --workers 4 \
    --model "coca_ViT-L-14" \
    --save-frequency 3 \
    --pretrained "mscoco_finetuned_laion2b_s13b_b90k" \
    --report-to "tensorboard" \
    --log-every-n-steps 200 \
    --grad-checkpointing \
    --local-loss \
    --gather-with-grad \
    --accum-freq 4 \
    > "/mnt/g/Logtemp/open_clip/Body_Parts_XRay/coca_ViT-L_14_clahe.log" 2>&1

nohup python ./Body_Parts_XRay/test_Body_Parts_XRay.py > test_Body_Parts_XRay.log 2>&1