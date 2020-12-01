import argparse
from model.utils.config import cfg, cfg_from_file, cfg_from_list

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--dataset_t', dest='dataset_t',
                        help='target training dataset',
                        default='dt_clipart', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--mode', dest='mode',
                    help='instance_level, image_level',
                    default='image_level', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=8, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--t', dest='t',
                      help='temperature parameter for contrastive loss',
                      default=1.0, type=float)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch', nargs='*',
                      default=[5], type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)
  parser.add_argument('--lambd', dest='lambd',
                      help='instance level loss weight',
                      default=1.0, type=float)
  parser.add_argument('--gamma', dest='gamma',
                      help='contrastive loss weight',
                      default=0.1, type=float)

  parser.add_argument('--conf_thresh', dest='conf_thresh',
                      help='confidence thresh',
                      default=0.95, type=float)
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="images")
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--log_ckpt_name', dest='log_ckpt_name',
                        help='saved log and ckpt dir name',
                        default='cs2cs_fg', type=str)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  args = parser.parse_args()
  return args

def set_dataset_args(args, test=False):
    if not test: # Training Phase
        # Source Dataset
        if args.dataset == "pascal_voc":
          args.imdb_name = "voc_2007_trainval"
          args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "pascal_voc_0712":
            args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "pascal_voc_water_0712":
            args.imdb_name = "voc_water_2007_trainval+voc_water_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "clipart":
            args.imdb_name = "clipart_2007_trainval"
#             args.imdb_name = "clipart_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "comic":
            args.imdb_name = "comic_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "watercolor":
            args.imdb_name = "watercolor_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "cityscape":
            args.imdb_name = "cityscape_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "sim10k":
            args.imdb_name = "sim10k_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset == "cityscape_car":
            args.imdb_name = "cityscape_car_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "foggy_cityscape":
            args.imdb_name = "foggy_cityscape_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        
        elif args.dataset == "dt_clipart":
            args.imdb_name = "dt_clipart_voc_2007_trainval+dt_clipart_voc_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "dt_comic":
            args.imdb_name = "dt_comic_voc_2007_trainval+dt_comic_voc_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "dt_watercolor":
            args.imdb_name = "dt_watercolor_voc_2007_trainval+dt_watercolor_voc_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
            

        elif args.dataset == "foggy2city":
            args.imdb_name = "foggy2city_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "city2sim10k":
            args.imdb_name = "city2sim10k_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "clipart2VOC07":
            args.imdb_name = "clipart2VOC_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "comic2VOC07":
            args.imdb_name = "comic2VOC_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "watercolor2VOC07":
            args.imdb_name = "watercolor2VOC_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
            
        elif args.dataset == "pl_clipart":
            args.imdb_name = "pl_clipart_voc_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "pl_comic":
            args.imdb_name = "pl_comic_voc_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "pl_watercolor":
            args.imdb_name = "pl_watercolor_voc_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "pl_cityscape":
            args.imdb_name = "pl_cityscape_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "pl_foggy_cityscape":
            args.imdb_name = "pl_foggy_cityscape_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        
        # Target Dataset
        if args.dataset_t == "dt_clipart":
            args.imdb_name_target = "dt_clipart_voc_2007_trainval+dt_clipart_voc_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "dt_comic":
            args.imdb_name_target = "dt_comic_voc_2007_trainval+dt_comic_voc_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "dt_watercolor":
            args.imdb_name_target = "dt_watercolor_voc_2007_trainval+dt_watercolor_voc_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset_t == "clipart2VOC07":
            args.imdb_name_target = "clipart2VOC_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "comic2VOC07":
            args.imdb_name_target = "comic2VOC_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "watercolor2VOC07":
            args.imdb_name_target = "watercolor2VOC_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset_t == "city2foggy":
            args.imdb_name_target = "city2foggy_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset_t == "foggy2city":
            args.imdb_name_target = "foggy2city_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset_t == "sim10k2city":
            args.imdb_name_target = "sim10k2city_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset_t == "city2sim10k":
            args.imdb_name_target = "city2sim10k_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
            
        elif args.dataset_t == "clipart":
            args.imdb_name_target = "clipart_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "comic":
            args.imdb_name_target = "comic_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "watercolor":
            args.imdb_name_target = "watercolor_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset_t == "cityscape":
            args.imdb_name_target = "cityscape_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset_t == "sim10k":
            args.imdb_name_target = "sim10k_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        
        
            
            
    else: # Testing Phase
        if args.dataset == "pascal_voc":
            args.imdb_name = "voc_2007_trainval"
            args.imdbval_name = "voc_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        elif args.dataset == "pascal_voc_0712":
            args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
            args.imdbval_name = "voc_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        elif args.dataset == "pascal_voc_water_07":
            args.imdb_name = "voc_water_2007_trainval"
            args.imdbval_name = "voc_water_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

        elif args.dataset == "clipart":
            args.imdb_name = "clipart_2007_trainval"
            args.imdbval_name = "clipart_2007_trainval"
#             args.imdbval_name = "clipart_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "comic":
            args.imdb_name = "comic_2007_train"
            args.imdbval_name = "comic_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "watercolor":
            args.imdb_name = "watercolor_2007_train"
            args.imdbval_name = "watercolor_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "cityscape":
            args.imdb_name = "cityscape_2007_trainval"
            args.imdbval_name = "cityscape_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "foggy_cityscape":
            args.imdb_name = "foggy_cityscape_2007_trainval"
            args.imdbval_name = "foggy_cityscape_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "cityscape_car":
            args.imdb_name = "cityscape_car_2007_trainval"
            args.imdbval_name = "cityscape_car_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "sim10k":
            args.imdb_name = "sim10k_2012_trainval"
            args.imdbval_name = "sim10k_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']


    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)
    
    return args


