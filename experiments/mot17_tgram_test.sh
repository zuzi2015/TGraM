cd src
CUDA_VISIBLE_DEVICES=0,1 python train_air.py mot --exp_id 210525_mot17_tgrammbv3 --arch tgrammbv3 --batch_size 30 --gpus 0,1 --lr 1.25e-4 --down_ratio 4 --load_model '' --data_cfg '../src/lib/cfg/mot17.json' --data_dir '/workspace/fairmot/src/data/' --num_frames 3 --dataloader tgram  --num_epochs 150 --val_intervals 30 --num_workers 0
cd ..

