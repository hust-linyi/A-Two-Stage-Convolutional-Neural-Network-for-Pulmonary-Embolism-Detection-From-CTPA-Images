CUDA_VISIBLE_DEVICES=1 python main.py\
  --model res18\
  --config config_training\
  -b 4\
  --gpu 1\
  -j 1\
  --data all\
  --save-freq 1\
  --epochs 100\
  --lr 0.0001\
  --resume /home/linyi/Code/pytorch/pe_detection_clean/model/best_public.ckpt




