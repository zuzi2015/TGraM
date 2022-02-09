cd src
CUDA_VISIBLE_DEVICES=0,1 python train_air.py mot --resume --exp_id 210529_mot17_tgrammbseg --arch tgrammbseg --batch_size 32 --gpus 0,1 --lr 7e-5 --down_ratio 4 --load_model '../exp/mot/210529_mot17_tgrammbseg/model_last.pth' --data_cfg '../src/lib/cfg/mot17.json' --data_dir '/workspace/fairmot/src/data/' --num_frames 3 --dataloader tgram  --num_epochs 210 --val_intervals 30 --num_workers 0
cd ..

