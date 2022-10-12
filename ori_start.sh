#### HR_shanghai 
#standard 
# 01 
# 0.81~0.83
# python stc_train_eval.py --dataset 'HR_shanghai' --dataset_sence '01' --only_normal True --level high --seg_len 7 --pred_next_frame 4 --filter_bbox 2200 --filter_border 5 --filter_cover False --filter_independent False --ae_lr 0.001 --ae_batch_size 32

# 02
# 0.807
# python stc_train_eval.py --dataset 'HR_shanghai' --dataset_sence 02 --level high --seg_len 7 --pred_next_frame 5 --filter_bbox 2200 --filter_border 5
# 
# python stc_train_eval.py --dataset 'HR_shanghai' --dataset_sence 02 --level high --seg_len 7 --pred_next_frame 5 --filter_bbox 2000 #--filter_border -1

# 04
# 0.75
# python stc_train_eval.py --dataset 'HR_shanghai' --dataset_sence 04 --level high --seg_len 7 --pred_next_frame 4 --filter_bbox 2000 --filter_border -1
# 0.75
# python stc_train_eval.py --dataset 'HR_shanghai' --dataset_sence 04 --level high --seg_len 7 --pred_next_frame 5 --filter_bbox 2000 --filter_border 2
# 0.80
# python stc_train_eval.py --dataset 'HR_shanghai' --dataset_sence 04 --level high --seg_len 6 --pred_next_frame 2 --filter_bbox 2200 --filter_border 2

# 06
# 0.864
# python stc_train_eval.py --dataset 'HR_shanghai' --dataset_sence 06 --level high --seg_len 7 --pred_next_frame 4 --filter_bbox 2200 --filter_border 2


# 08
# 0.70
# python stc_train_eval.py --dataset 'HR_shanghai' --dataset_sence 08 --level high --seg_len 7 --pred_next_frame 4 --filter_bbox 2200 --filter_border 2


# 09
# 0.949
# python stc_train_eval.py --dataset 'HR_shanghai' --dataset_sence 09 --level high --seg_len 7 --pred_next_frame 4 --filter_bbox 2200 --filter_border 2


# 10，有固定不动的电动车异常
# 0.78
# python stc_train_eval.py --dataset 'HR_shanghai' --dataset_sence 10 --level high --seg_len 7 --pred_next_frame 4 --filter_bbox 2200 --filter_border 2
# 0.80
# python stc_train_eval.py --dataset 'HR_shanghai' --dataset_sence 10 --level high --seg_len 7 --pred_next_frame 4 # --filter_bbox 200 --filter_border 2


# 11
# 0.98
# python stc_train_eval.py --dataset 'HR_shanghai' --dataset_sence 11 --level high --seg_len 7 --pred_next_frame 4 --filter_bbox 2200 # --filter_border 2

# 12
# 0.77 不稳，大部分是0.65
# python stc_train_eval.py --dataset 'HR_shanghai' --dataset_sence 12 --level high --seg_len 7 --pred_next_frame 4 --filter_bbox 200 # --filter_border 2


# 02 03 05 07 08
# 0.846
# python stc_train_eval.py --dataset 'HR_shanghai' --train_dataset_sence 01 02 03 04 05 06 07 08 09 10 11 12 13 --dataset_sence 02 03 05 07 08 --level low --seg_len 7 --pred_next_frame 2 --filter_bbox 2200 --filter_border 5 --ae_batch_size 256

# 02 03 05 07 08 new_process
# 0.826
# python stc_train_eval.py --dataset 'HR_shanghai' --train_dataset_sence 01 02 03 04 05 06 07 08 09 10 11 12 13 --dataset_sence 02 03 05 07 08 --level low --seg_len 7 --pred_next_frame 2 --filter_bbox 2200 --filter_border 5 --ae_batch_size 256 --new_preprocess True


# 02 03 04 05 07 08 12
# 0.724
# python stc_train_eval.py --dataset 'HR_shanghai' --train_dataset_sence 01 02 03 04 05 06 07 08 09 10 11 12 13 --dataset_sence 02 03 04 05 07 08 12 --level low --seg_len 7 --pred_next_frame 2 --filter_bbox 2200 --filter_border 5 --ae_batch_size 128


# 02 03 05 07 08 12
# 
# python stc_train_eval.py --dataset 'HR_shanghai' --train_dataset_sence 01 02 03 04 05 06 07 08 09 10 11 12 13 --dataset_sence 02 03 05 07 08 12 --level low --seg_len 7 --pred_next_frame 2 --filter_bbox 2200 --filter_border 5 --ae_batch_size 256

# 02 03 04 05 07 08
# 0.808
# python stc_train_eval.py --dataset 'HR_shanghai' --train_dataset_sence 01 02 03 04 05 06 07 08 09 10 11 12 13 --dataset_sence 02 03 04 05 07 08 --level low --seg_len 7 --pred_next_frame 2 --filter_bbox 2200 --filter_border 5 --ae_batch_size 256


#### avenue
# 奔跑组 01_02 01_03 01_04 
# 抛掷组 02_05 02_10 02_11 02_12 02_13 02_14 02_17 02_18 02_20
# 跳跃组 03_07 03_08 03_21
# 位置相关组 01_01 02_06 02_09  02_15 02_19
# HR 02_16

# standard 01 02 03 06 16

# all
# 0.63
# python stc_train_eval.py --dataset 'avenue' --dataset_sence 01 --vid_res 640 360 --level high --seg_len 7 --pred_next_frame 4 --filter_border 2  #--filter_bbox 2200
# 0.656
# python stc_train_eval.py --dataset 'avenue' --dataset_sence 01 --vid_res 640 360 --level low --seg_len 7 --pred_next_frame 2 --filter_border 2   --ae_batch_size 256 #--filter_bbox 2200
# 0.67
# python stc_train_eval.py --dataset 'avenue' --dataset_sence 01 --vid_res 640 360 --level low --seg_len 4 --pred_next_frame 1 --filter_border 2   --ae_batch_size 256 #--filter_bbox 2200

# 01
# 0.982
# python stc_train_eval.py --dataset 'HR_avenue' --vid_res 640 360 --train_dataset_sence 01 --dataset_sence 01 --level high --seg_len 7 --pred_next_frame 5 --filter_border 2

#### HR_avenue
# 05_03
# 0.99
python stc_train_eval.py --dataset 'HR_avenue' --vid_res 640 360 --train_dataset_sence 01 --dataset_sence 05 --level high --seg_len 7 --pred_next_frame 5 --filter_border 2
# 06
# 0.99
# python stc_train_eval.py --dataset 'HR_avenue' --vid_res 640 360 --train_dataset_sence 01 --dataset_sence 06 --level high --seg_len 7 --pred_next_frame 5 --filter_border 2


# 03
# 0.962
# python stc_train_eval.py --dataset 'HR_avenue' --vid_res 640 360 --train_dataset_sence 01 --dataset_sence 03 --level high --seg_len 7 --pred_next_frame 5 --filter_border 2

# 02 hard
# 0.67
# python stc_train_eval.py --dataset 'HR_avenue' --vid_res 640 360 --train_dataset_sence 01 --dataset_sence 02 --level low --seg_len 6 --pred_next_frame 1 --filter_border 2 --ae_batch_size 512
# 0.736
# python stc_train_eval.py --dataset 'HR_avenue' --vid_res 640 360 --train_dataset_sence 01 --dataset_sence 02 --level low --seg_len 7 --pred_next_frame 2 --filter_border 2 --ae_batch_size 512
# new preprocess有效 0.75
# python stc_train_eval.py --dataset 'HR_avenue' --vid_res 640 360 --train_dataset_sence 01 --dataset_sence 02 --level low --seg_len 7 --pred_next_frame 2 --filter_border 2 --ae_batch_size 512  --new_preprocess True

#### HR_avenue soft
### 01 02 03
# python stc_train_eval.py --dataset 'HR_avenue' --vid_res 640 360 --train_dataset_sence 01 --dataset_sence 01 02 03 --level low --seg_len 7 --pred_next_frame 2 --filter_border 2 --ae_batch_size 512

# 04
# 
# python stc_train_eval.py --dataset 'HR_avenue' --vid_res 640 360 --train_dataset_sence 01 --dataset_sence 04 --level low --seg_len 7 --pred_next_frame 2 --filter_border 2 --ae_batch_size 512

# 02 04
# 
# python stc_train_eval.py --dataset 'HR_avenue' --vid_res 640 360 --train_dataset_sence 01 --dataset_sence 02 04 --level high --seg_len 7 --pred_next_frame 5 --ae_batch_size 512
# 0.64
# python stc_train_eval.py --dataset 'HR_avenue' --vid_res 640 360 --train_dataset_sence 01 --dataset_sence 02 04 --level low --seg_len 7 --pred_next_frame 2 --filter_border 2 --ae_batch_size 512
# 0.656
# python stc_train_eval.py --dataset 'HR_avenue' --vid_res 640 360 --train_dataset_sence 01 --dataset_sence 02 04 --level low --seg_len 7 --pred_next_frame 2 --filter_border 2 --ae_batch_size 512  --new_preprocess True



#### avenue
# 02 04
# 0.60
# python stc_train_eval.py --dataset 'avenue' --vid_res 640 360 --train_dataset_sence 01 --dataset_sence 02 04 --level high --seg_len 7 --pred_next_frame 5 --filter_border 2 --ae_batch_size 512
#
# python stc_train_eval.py --dataset 'avenue' --vid_res 640 360 --train_dataset_sence 01 --dataset_sence 02 04 --level low --seg_len 7 --pred_next_frame 2 --filter_border 2



#### HR_corridor    
# except 02 03 07 09

#04 05 11
# 0.785
# python stc_train_eval.py --dataset 'corridor' --vid_res 1920 1080 --train_dataset_sence 01 --dataset_sence 04 05 11 --level high --seg_len 7 --pred_next_frame 5 --filter_bbox 5000

#01 06 08 10 
# 0.86
# python stc_train_eval.py --dataset 'corridor' --vid_res 1920 1080 --train_dataset_sence 01 --dataset_sence 01 06 08 10 --level low --seg_len 7 --pred_next_frame 2 --filter_bbox 5000 --ae_batch_size 512  #--new_preprocess True --filter_border 2 

#01 06 08 10 02 03 07 09
# 0.66
# python stc_train_eval.py --dataset 'corridor' --vid_res 1920 1080 --train_dataset_sence 01 --dataset_sence 01 06 08 10 02 03 07 09 --level high --seg_len 7 --pred_next_frame 2 --filter_bbox 5000 --ae_batch_size 512  #--new_preprocess True --filter_border 2 
# 0.747
# python stc_train_eval.py --dataset 'corridor' --vid_res 1920 1080 --train_dataset_sence 01 --dataset_sence 01 06 08 10 02 03 07 09 --level low --seg_len 7 --pred_next_frame 2 --filter_bbox 5000 --ae_batch_size 512  #--new_preprocess True --filter_border 2 















#### ucsdped1
# 
# python stc_train_eval.py --dataset 'ucsdped1' --dataset_sence 01 --vid_res 238 158 --level high --seg_len 7 --pred_next_frame 4 --filter_border 2  #--filter_bbox 150


#### ucsdped2
# HR 04
# 0.75
# python stc_train_eval.py --dataset 'ucsdped2' --dataset_sence 01 --vid_res 360 240 --level high --seg_len 7 --pred_next_frame 4 --filter_border 2  #--filter_bbox 150
# 0.81
# python stc_train_eval.py --dataset 'ucsdped2' --dataset_sence 01 --vid_res 360 240 --level high --seg_len 14 --pred_next_frame 7 --filter_border 2  #--filter_bbox 150
# 
# python stc_train_eval.py --dataset 'ucsdped2' --dataset_sence 01 --vid_res 360 240 --level high --seg_len 14 --pred_next_frame 10 --filter_border 2  #--filter_bbox 150
# 
# python stc_train_eval.py --dataset 'ucsdped2' --dataset_sence 01 --vid_res 360 240 --level low --seg_len 7 --pred_next_frame 2 --filter_border 2  --ae_batch_size 512

# 0.75
# python stc_train_eval.py --dataset 'ucsdped2' --dataset_sence 01 --vid_res 360 240 --level high --seg_len 8 --pred_next_frame 6 --filter_border 2  --use_bbox_high_level True

# 分组测试
# 12356
# python stc_train_eval.py --dataset 'ucsdped2' --dataset_sence 01 --vid_res 360 240 --level high --seg_len 7 --pred_next_frame 4 --filter_border 2  #--filter_bbox 150


