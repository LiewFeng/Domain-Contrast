echo 'pascal2clipart'

echo 'base detector training'
python train_detector.py --dataset pascal_voc_0712 --net res101 --cuda --bs 1 --nw 8 --lr 1e-3 --lr_decay_step 5 --s 1 --log_ckpt_name pascal2clipart --epochs 7
echo 'test on base_net on pascal voc07'
python test.py --dataset pascal_voc_0712 --net res101 --cuda  --checksession 1 --checkepoch 7 --checkpoint 33101 --log_ckpt_name pascal2clipart
echo 'test on base detector on clipart'
python test.py --dataset clipart --net res101 --cuda  --checksession 1 --checkepoch 7 --checkpoint 33101 --log_ckpt_name pascal2clipart


echo 'S2T contrastive training'
python train_S2T_contrast.py --dataset pascal_voc_0712 --dataset_t dt_clipart --net res101 --cuda --nw 8 --bs 8 --r True --checksession 1 --checkepoch 7 --checkpoint 33101 --lr 0.000005 --s 2 --epochs 5 --lr_decay_step 5 --log_ckpt_name pascal2clipart --t 0.5 --lambd 0.5 
echo 'test on S2T model'
python test.py --dataset clipart --net res101 --cuda  --checksession 2 --checkepoch 5 --checkpoint 4137 --log_ckpt_name pascal2clipart


echo 'T2S  training'
python train_T2S_contrast.py --dataset clipart --dataset_t clipart2VOC07 --net res101 --cuda --nw 8 --bs 8 --r True --checksession 2 --checkepoch 5 --checkpoint 4137 --lr 0.00002 --s 3 --epochs 5 --lr_decay_step 5 --log_ckpt_name pascal2clipart --t 0.5 --lambd 1.0 --mode image_level 
echo 'test on T2S model'
python test.py --dataset clipart --net res101 --cuda  --checksession 3 --checkepoch 5 --checkpoint 249 --log_ckpt_name pascal2clipart


echo 'Pseudo Label'
python pseudo_label.py --dataset clipart2VOC07 --net res101 --cuda  --checksession 3 --checkepoch 5 --checkpoint 249 --save_dir data/pl_clipart/VOC2007 --log_ckpt_name pascal2clipart --conf_thresh 0.99
# # Remember to remove old pkl file if it's not the first time of PL
# rm data/cache/pl_clipart_2007_trainval_gt_roidb.pkl


echo 'fine tuning on Pseudo Labels'
python train_detector.py --dataset pl_clipart --net res101 --cuda --nw 8 --bs 1 --r True --checksession 3 --checkepoch 5 --checkpoint 249 --lr 0.00001 --s 4 --epochs 3 --lr_decay_step 5 --log_ckpt_name pascal2clipart
echo 'test on Pseudo_Labels_model'
# Remember to change the checkpoint according to your training
python test.py --dataset clipart --net res101 --cuda  --checksession 4 --checkepoch 3 --checkpoint $checkpoint --log_ckpt_name pascal2clipart


echo 'demo on clipart'
# Remember to change the checkpoint according to your training
python demo.py --net res101 --checksession 4 --checkepoch 3 --checkpoint $checkpoint --cuda --dataset clipart --log_ckpt_name pascal2clipart --save_dir clipart_result

# echo 'S2T instance level contrastive training'
# python train_T2S_contrast.py --dataset pascal_voc_0712 --dataset_t dt_clipart --net res101 --cuda --nw 8 --bs 8 --r True --checksession 1 --checkepoch 7 --checkpoint 33101 --lr 0.000005 --s 5 --epochs 5 --lr_decay_step 5 --log_ckpt_name pascal2clipart --t 0.5 --lambd 1.0 --mode instance_level 
# echo 'test on S2T instance_level model'
# python test.py --dataset clipart --net res101 --cuda  --checksession 5 --checkepoch $i --checkpoint 4137 --log_ckpt_name pascal2clipart

# echo 'S2T image level contrastive training'
# python train_T2S_contrast.py --dataset pascal_voc_0712 --dataset_t dt_clipart --net res101 --cuda --nw 8 --bs 8 --r True --checksession 1 --checkepoch 7 --checkpoint 33101 --lr 0.000005 --s 6 --epochs 5 --lr_decay_step 5 --log_ckpt_name pascal2clipart --t 0.5 --lambd 1.0 --mode instance_level 
# echo 'test on S2T image_level model'
# python test.py --dataset clipart --net res101 --cuda  --checksession 6 --checkepoch $i --checkpoint 4137 --log_ckpt_name pascal2clipart