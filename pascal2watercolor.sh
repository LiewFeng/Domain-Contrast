echo 'pascal2watercolor'

echo 'base detector training'
python trainval_net.py --dataset pascal_voc_water_0712 --net res101 --cuda --bs 1 --nw 8 --lr 1e-3 --lr_decay_step 5 --s 1 --log_ckpt_name pascal2watercolor --epochs 6

echo 'S2T contrastive training'
python train_S2T_contrast.py --dataset pascal_voc_water_0712 --dataset_t dt_watercolor --net res101 --cuda --nw 8 --bs 8 --r True --checksession 1 --checkepoch 6 --checkpoint 22977 --lr 0.000005 --s 2 --epochs 5 --lr_decay_step 5 --log_ckpt_name pascal2watercolor --t 0.5 --lambd 0.5

echo 'T2S contrastive training'
python train_T2S_contrast.py --dataset watercolor --dataset_t watercolor2VOC07 --net res101 --cuda --nw 8 --bs 8 --r True --checksession 2 --checkepoch 5 --checkpoint 2872 --lr 0.0000001 --s 5 --epochs 5 --lr_decay_step 5 --log_ckpt_name pascal2watercolor --t 0.5 --lambd 1.0 --mode image_level 

echo 'Pseudo Label'
python pseudo_label.py --dataset watercolor2VOC07 --net res101 --cuda  --checksession 3 --checkepoch 5 --checkpoint 249 --save_dir data/pl_watercolor/VOC2007 --log_ckpt_name pascal2watercolor --conf_thresh 0.99
# # Remember to remove old pkl file if it's not the first time of PL
# rm data/cache/pl_watercolor_2007_train_gt_roidb.pkl


echo 'fine tuning on Pseudo Labels'
python train_detector.py --dataset pl_watercolor --net res101 --cuda --nw 8 --bs 1 --r True --checksession 3 --checkepoch 5 --checkpoint 249 --lr 0.00002 --s 4 --epochs 3 --lr_decay_step 5 --log_ckpt_name pascal2watercolor
echo 'test on Pseudo_Labels_model'
# Remember to change the checkpoint according to your training
python test.py --dataset watercolor --net res101 --cuda  --checksession 4 --checkepoch 3 --checkpoint $checkpoint --log_ckpt_name pascal2watercolor

