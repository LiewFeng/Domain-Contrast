echo 'sim10k2city'

echo 'base_net training'
python train_detector.py --dataset sim10k --net vgg16 --cuda --bs 1 --nw 8 --lr 1e-3 --lr_decay_step 5 --s 1 --log_ckpt_name sim10k2city --epochs 7 

echo 'S2T contrastive training'
python train_S2T.py --dataset sim10k --dataset_t sim10k2city --net vgg16 --cuda --nw 8 --bs 8 --r True --checksession 1 --checkepoch 7 --checkpoint 19949 --lr 0.000001 --s 2 --epochs 5 --lr_decay_step 5 --log_ckpt_name sim10k2city --t 0.5 --lambd 0.9 --mGPUs 

echo 'T2S contrastive training'
python train_T2S.py --dataset cityscape --dataset_t city2sim10k --net vgg16 --cuda --nw 8 --bs 8 --r True --checksession 2 --checkepoch 5 --checkpoint 19949 --lr 0.0000001 --s 3 --epochs 5 --lr_decay_step 5 --log_ckpt_name sim10k2city --t 0.5 --mode image_level --mGPUs

echo 'Pseudo Label'
python PseudoLabel.py --dataset city2sim10k --net vgg16 --cuda  --checksession 3 --checkepoch 2 --checkpoint 707 --save_dir /userhome/Datasets/pl_cityscape/VOC2007 --log_ckpt_name sim10k2city --conf_thresh 0.95
# Remember to remove old pkl file if it's not the first time of PL
# rm data/cache/pl_cityscape_2007_trainval_gt_roidb.pkl

echo 'fine tuning on Pseudo Labels'
python train_detector.py --dataset pl_cityscape --net vgg16 --cuda --nw 8 --bs 1 --r True --checksession 3 --checkepoch 2 --checkpoint 707 --lr 0.00001 --s 4 --epochs 3 --lr_decay_step 5 --log_ckpt_name sim10k2city
echo 'test on Pseudo_Labels_model'
python test_net.py --dataset cityscape_car --net vgg16 --cuda  --checksession 4 --checkepoch 3 --checkpoint $checkpoint --log_ckpt_name sim10k2city