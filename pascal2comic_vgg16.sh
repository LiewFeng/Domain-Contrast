echo 'pascal2comic_vgg16'

echo 'base detector training'
python train_detector.py --dataset pascal_voc_water_0712 --net vgg16 --cuda --bs 1 --nw 8 --lr 1e-3 --lr_decay_step 5 --s 1 --log_ckpt_name pascal2comic --epochs 7

echo 'S2T contrastive training'
python train_S2T_contrast.py --dataset pascal_voc_water_0712 --dataset_t dt_comic --net vgg16 --cuda --nw 8 --bs 8 --r True --checksession 1 --checkepoch 7 --checkpoint 22977 --lr 0.000001 --s 2 --epochs 5 --lr_decay_step 5 --log_ckpt_name pascal2comic --t 0.5 --lambd 0.5

echo 'T2S contrastive training'
python train_T2S_contrast.py --dataset comic --dataset_t comic2VOC07 --net vgg16 --cuda --nw 8 --bs 8 --r True --checksession 2 --checkepoch 5 --checkpoint 2872 --lr 0.000005 --s 3 --epochs 5 --lr_decay_step 5 --log_ckpt_name pascal2comic --t 0.5 --lambd 1.0 --mode image_level 

echo 'Pseudo Label'
python pseudo_label.py --dataset comic2VOC07 --net vgg16 --cuda  --checksession 3 --checkepoch 5 --checkpoint 249 --save_dir data/pl_comic/VOC2007 --log_ckpt_name pascal2comic --conf_thresh 0.95
# # Remember to remove old pkl file if it's not the first time of PL
# rm data/cache/pl_comic_2007_train_gt_roidb.pkl

echo 'fine tuning on Pseudo Labels'
python train_detector.py --dataset pl_comic --net vgg16 --cuda --nw 8 --bs 1 --r True --checksession 3 --checkepoch 5 --checkpoint 249 --lr 0.00001 --s 4 --epochs 3 --lr_decay_step 5 --log_ckpt_name pascal2comic
echo 'test on Pseudo_Labels_model'
# Remember to change the checkpoint according to your training
python test.py --dataset comic --net vgg16 --cuda  --checksession 4 --checkepoch 3 --checkpoint $checkpoint --log_ckpt_name pascal2comic3




