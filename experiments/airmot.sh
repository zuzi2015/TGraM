cd src
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_air.py mot --exp_id 210622_airmot_tgram_sydney_11_2 --arch dla_34 --num_epochs 1000 --batch_size 1 --gpus 0 --lr 7e-4 --down_ratio 4 --load_model '../exp/airmot/210622_airmot_tgram_sydney_11/model_last.pth' --data_cfg '../src/lib/cfg/air_mot.json' --data_dir '/workspace/tgram/src/data/' --num_frames 3 --dataloader jde --val_intervals 30 --num_workers 0
cd ..

