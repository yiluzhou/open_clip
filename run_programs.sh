#!/bin/bash

nohup python -m training.main_test \
    --train-data "./Knee_OA/train.csv" \
    --val-data "./Knee_OA/val.csv" \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --warmup 1000 \
    --logs "/mnt/g/Logtemp/open_clip/Knee_OA" \
    --batch-size 32 \
    --aug-cfg "augment_dir='/home/yilu/Development/open_clip/Knee_OA/'" \
    --lr 5e-6 \
    --wd 0.1 \
    --epochs 400 \
    --workers 4 \
    --model "coca_ViT-L-14" \
    --save-frequency 4 \
    --pretrained "mscoco_finetuned_laion2b_s13b_b90k" \
    --report-to "tensorboard" \
    --log-every-n-steps 200 \
    --grad-checkpointing \
    --local-loss \
    --gather-with-grad \
    --accum-freq 4 \
    > "/mnt/g/Logtemp/open_clip/Knee_OA/coca_ViT-L_14_normalization.log" 2>&1 &


    --aug-cfg "augment_dir='/home/yilu/Development/open_clip/Knee_OA/'" \
nohup python ./Knee_OA/test_Knee_OA.py > ./Knee_OA/test_Knee_OA.log 2>&1
