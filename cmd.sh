nohup python -m training.main_test \
    --train-data "./Whale_Dolphin_Identification/train.csv" \
    --val-data "./Whale_Dolphin_Identification/val.csv" \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --warmup 1000 \
    --logs "/mnt/g/Logtemp/open_clip/Whale_Dolphin_Identification" \
    --batch-size 32 \
    --lr 1e-6 \
    --wd 0.1 \
    --epochs 100 \
    --workers 4 \
    --model "coca_ViT-L-14" \
    --save-frequency 1 \
    --pretrained "mscoco_finetuned_laion2B-s13B-b90k" \
    --report-to "tensorboard" \
    --log-every-n-steps 100 \
    --grad-checkpointing \
    --local-loss \
    --gather-with-grad \
    --accum-freq 4 \
    > "/mnt/g/Logtemp/open_clip/Whale_Dolphin_Identification/coca_ViT-L_14_1.txt" 2>&1 &


nohup python -m training.main_test \
    --train-data "./Body_Parts_XRay/train_square.csv" \
    --val-data "./Body_Parts_XRay/val_square.csv" \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --warmup 1000 \
    --logs "/mnt/g/Logtemp/open_clip/Body_Parts_XRay" \
    --batch-size 32 \
    --aug-cfg "augment_dir='/home/yilu/Development/open_clip/Body_Parts_XRay/'" \
    --lr 1e-6 \
    --wd 0.1 \
    --epochs 200 \
    --workers 4 \
    --model "coca_ViT-L-14" \
    --save-frequency 2 \
    --pretrained "mscoco_finetuned_laion2B-s13B-b90k" \
    --report-to "tensorboard" \
    --log-every-n-steps 200 \
    --grad-checkpointing \
    --local-loss \
    --gather-with-grad \
    --accum-freq 4 \
    > "/mnt/g/Logtemp/open_clip/Body_Parts_XRay/coca_ViT-L_14_aug.txt" 2>&1 &


nohup torchrun --nproc_per_node 3 -m training.main_test \
--train-data "./Body_Parts_XRay/train.csv" \
--val-data "./Body_Parts_XRay/val.csv" \
--csv-img-key filepath \
--csv-caption-key caption \
--warmup 1000 \
--logs "/mnt/x/Log/open_clip_2920X/Body_Parts_XRay/" \
--batch-size 16 \
--lr 1e-6 \
--wd 0.1 \
--epochs 100 \
--workers 4 \
--model "coca_ViT-B-32" \
--save-frequency 1 \
--pretrained "mscoco_finetuned_laion2b_s13b_b90k" \
--report-to "tensorboard" \
--log-every-n-steps 100 \
--grad-checkpointing \
--local-loss \
--gather-with-grad \
--accum-freq 4 \
> "/mnt/x/Log/open_clip_2920X/Body_Parts_XRay/coca_ViT-B-32_1.txt" 2>&1 &



nohup python -m training.main_test \
    --train-data "./Body_Parts_XRay/train_square.csv" \
    --val-data "./Body_Parts_XRay/val_square.csv" \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --warmup 1000 \
    --logs "/mnt/g/Logtemp/open_clip/Body_Parts_XRay" \
    --batch-size 32 \
    --aug-cfg "augment_dir='/home/yilu/Development/open_clip/Body_Parts_XRay/'" \
    --lr 1e-6 \
    --wd 0.1 \
    --epochs 200 \
    --workers 4 \
    --model "coca_ViT-L-14" \
    --save-frequency 2 \
    --pretrained "/mnt/g/Logtemp/open_clip/Body_Parts_XRay/2023_08_21-23_03_36-model_coca_ViT-L-14-lr_1e-06-b_32-j_4-p_amp/checkpoints/epoch_200.pt" \
    --report-to "tensorboard" \
    --log-every-n-steps 200 \
    --grad-checkpointing \
    --local-loss \
    --gather-with-grad \
    --accum-freq 4 \
    > "/mnt/g/Logtemp/open_clip/Body_Parts_XRay/coca_ViT-L_14_aug2.txt" 2>&1 &


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
    > "/mnt/g/Logtemp/open_clip/Body_Parts_XRay/coca_ViT-L_14_clahe.log" 2>&1 &

nohup python ./Body_Parts_XRay/test_Body_Parts_XRay.py > test_Body_Parts_XRay.log 2>&1 &


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
    --pretrained "/mnt/g/Logtemp/open_clip/Body_Parts_XRay/2023_08_23-07_20_30-model_coca_ViT-L-14-lr_2e-06-b_32-j_4-p_amp/checkpoints/epoch_300.pt" \
    --report-to "tensorboard" \
    --log-every-n-steps 200 \
    --grad-checkpointing \
    --local-loss \
    --gather-with-grad \
    --accum-freq 4 \
    > "/mnt/g/Logtemp/open_clip/Body_Parts_XRay/coca_ViT-L_14_clahe1.log" 2>&1
nohup python ./Body_Parts_XRay/test_Body_Parts_XRay.py > test_Body_Parts_XRay.log 2>&1
