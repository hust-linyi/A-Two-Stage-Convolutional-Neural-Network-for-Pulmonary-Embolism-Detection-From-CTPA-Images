#!/bin/bash
ckpt_path=/home/linyi/Code/pytorch/pe_detection/detector/results/train/allres18-1220-1251/
main_path=/home/linyi/Code/pytorch/pe_detection/detector/
eval_path=/home/linyi/Code/pytorch/pe_detection/evaluationScript/


CUDA_VISIBLE_DEVICES=1 python $main_path/lin_main.py\
  --model res18\
  --config config_training_new_test\
  -b 8\
  --gpu 1\
  -j 1\
  --data all\
  --save-freq 1\
  --epochs 50\
  --lr 0.0001\
  --save-dir $ckpt_path\
  --resume ../best_public.ckpt


for i in $(ls $ckpt_path | grep ckpt)
do
echo "start"
CUDA_VISIBLE_DEVICES=1 python $main_path/lin_main.py\
  --resume $ckpt_path$i\
  --model res18\
  --config config_training_new_test\
  -b 1\
  --gpu 1\
  -j 1\
  --test 1\
  --testthresh -10
echo "end"
done


python $eval_path/lin_getfroc_all_ckpt.py

