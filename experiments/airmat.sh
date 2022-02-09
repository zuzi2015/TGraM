cd src
CUDA_VISIBLE_DEVICES=0,1 python train_air.py mot --exp_id 210613_airmat_fairmot --arch dla_34 --batch_size 32 --gpus 0,1 --lr 7e-4 --down_ratio 4 --load_model '../models/mix_mot17_half_dla34.pth' --data_cfg '../src/lib/cfg/air_mat.json' --data_dir '/workspace/fairmot/src/data/' --num_frames 3 --dataloader jde  --num_epochs 210 --val_intervals 30 --num_workers 0
cd ..

