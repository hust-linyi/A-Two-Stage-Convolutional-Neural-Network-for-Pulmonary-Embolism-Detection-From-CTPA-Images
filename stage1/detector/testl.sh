#!/bin/bash
ckpt_path=
main_path=
eval_path=


for i in $(ls $ckpt_path | grep ckpt)
do
echo "start"
CUDA_VISIBLE_DEVICES=1 python $main_path/lin_main.py\
  --resume $ckpt_path$i\
  --model res18\
  --config config_training\
  -b 4\
  --gpu 2\
  -j 4\
  --test 1\
  --testthresh -8
echo "end"
done


python $eval_path/evaluate.py

