# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc_water import pascal_voc_water

from datasets.clipart import clipart
from datasets.dt_clipart import dt_clipart
from datasets.clipart2VOC import clipart2VOC
from datasets.pl_clipart import pl_clipart

from datasets.comic import comic
from datasets.dt_comic import dt_comic
from datasets.comic2VOC import comic2VOC
from datasets.pl_comic import pl_comic

from datasets.watercolor import watercolor
from datasets.dt_watercolor import dt_watercolor
from datasets.watercolor2VOC import watercolor2VOC
from datasets.pl_watercolor import pl_watercolor


from datasets.cityscape import cityscape
from datasets.foggy_cityscape import foggy_cityscape
from datasets.city2foggy import city2foggy
from datasets.foggy2city import foggy2city
from datasets.pl_foggy_cityscape import pl_foggy_cityscape

from datasets.sim10k import sim10k
from datasets.cityscape_car import cityscape_car
from datasets.sim10k2city import sim10k2city
from datasets.city2sim10k import city2sim10k
from datasets.pl_cityscape import pl_cityscape

from datasets.coco import coco
from datasets.imagenet import imagenet
from datasets.vg import vg

import numpy as np

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
# Set up voc_water_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_water_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc_water(split, year))

    
# Set up clipart_<year>_<split>    
for year in ['2007']:
  devkit_path = '/userhome/Datasets/clipart'
  for split in ['train', 'test', 'trainval']:
    name = 'clipart_{}_{}'.format(year, split)
    __sets[name] = (lambda year=year, split=split, devkit_path=devkit_path : clipart(split,year, devkit_path=devkit_path))
# Set up dt_clipart voc_<year>_<split>    
for year in ['2007', '2012']:
  devkit_path = '/userhome/Datasets/dt_clipart'
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'dt_clipart_voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year, devkit_path=devkit_path: dt_clipart(split, year, devkit_path=devkit_path))
# Set up clipart2VOC_<year>_<split>    
for year in ['2007']:
  devkit_path = '/userhome/Datasets/clipart2VOC'
  for split in ['train', 'test', 'trainval']:
    name = 'clipart2VOC_{}_{}'.format(year, split)
    __sets[name] = (lambda year=year, split=split, devkit_path=devkit_path : clipart2VOC(split,year, devkit_path=devkit_path))
# Set up pl_clipart voc_<year>_<split>    
for year in ['2007', '2012']:
  devkit_path = '/userhome/Datasets/pl_clipart'
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'pl_clipart_voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year, devkit_path=devkit_path: pl_clipart(split, year, devkit_path=devkit_path))

    
# Set up comic_<year>_<split>    
for year in ['2007']:
  devkit_path = '/userhome/Datasets/comic'
  for split in ['train', 'test', 'trainval', 'train_cg']:
    name = 'comic_{}_{}'.format(year, split)
    __sets[name] = (lambda year=year, split=split, devkit_path=devkit_path : comic(split,year, devkit_path=devkit_path))
# Set up dt_comic voc_<year>_<split>    
for year in ['2007', '2012']:
  devkit_path = '/userhome/Datasets/dt_comic'
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'dt_comic_voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year, devkit_path=devkit_path: dt_comic(split, year, devkit_path=devkit_path))
# Set up comic2VOC_<year>_<split>    
for year in ['2007']:
  devkit_path = '/userhome/Datasets/comic2VOC'
  for split in ['train', 'test', 'trainval']:
    name = 'comic2VOC_{}_{}'.format(year, split)
    __sets[name] = (lambda year=year, split=split, devkit_path=devkit_path : comic2VOC(split,year, devkit_path=devkit_path))
# Set up pl_comic_<year>_<split>    
for year in ['2007']:
  devkit_path = '/userhome/Datasets/pl_comic'
  for split in ['train', 'test', 'trainval']:
    name = 'pl_comic_voc_{}_{}'.format(year, split)
    __sets[name] = (lambda year=year, split=split, devkit_path=devkit_path : pl_comic(split,year, devkit_path=devkit_path))

    
# Set up watercolor_<year>_<split>
for year in ['2007']:
  devkit_path = '/userhome/Datasets/watercolor'
  for split in ['train', 'test', 'trainval']:
    name = 'watercolor_{}_{}'.format(year, split)
    __sets[name] = (lambda year=year, split=split, devkit_path=devkit_path : watercolor(split,year, devkit_path=devkit_path))
# Set up dt_watercolor voc_<year>_<split>
for year in ['2007', '2012']:
  devkit_path = '/userhome/Datasets/dt_watercolor'
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'dt_watercolor_voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year, devkit_path=devkit_path: dt_watercolor(split, year, devkit_path=devkit_path))
# Set up watercolor2VOC_<year>_<split>
for year in ['2007']:
  devkit_path = '/userhome/Datasets/watercolor2VOC'
  for split in ['train', 'test', 'trainval']:
    name = 'watercolor2VOC_{}_{}'.format(year, split)
    __sets[name] = (lambda year=year, split=split, devkit_path=devkit_path : watercolor2VOC(split,year, devkit_path=devkit_path))    
# Set up pl_watercolor_<year>_<split>
for year in ['2007']:
  devkit_path = '/userhome/Datasets/pl_watercolor'
  for split in ['train', 'test', 'trainval']:
    name = 'pl_watercolor_voc_{}_{}'.format(year, split)
    __sets[name] = (lambda year=year, split=split, devkit_path=devkit_path : pl_watercolor(split,year, devkit_path=devkit_path))
    
    
# Set up cityscape_<year>_<split>    
for year in ['2007']:
  devkit_path = '/userhome/Datasets/cityscape'
#   devkit_path = '/userhome/Datasets/Cityscape_FroggyCityscape/cityscape'
  for split in ['test', 'trainval']:
    name = 'cityscape_{}_{}'.format(year, split)
    __sets[name] = (lambda year=year, split=split, devkit_path=devkit_path : cityscape(split,year, devkit_path=devkit_path))
# Set up foggy_cityscape_<year>_<split>    
for year in ['2007']:
  devkit_path = '/userhome/Datasets/Cityscape_FroggyCityscape/foggy-cityscape'
  for split in ['test', 'trainval']:
    name = 'foggy_cityscape_{}_{}'.format(year, split)
    __sets[name] = (lambda year=year, split=split, devkit_path=devkit_path : foggy_cityscape(split,year, devkit_path=devkit_path))        
# Set up city2foggy_<year>_<split>
for year in ['2007']:
  devkit_path = '/userhome/Datasets/city2foggy'
  for split in ['test', 'trainval']:
    name = 'city2foggy_{}_{}'.format(year, split)
    __sets[name] = (lambda year=year, split=split, devkit_path=devkit_path : city2foggy(split,year, devkit_path=devkit_path))
# Set up foggy2city_<year>_<split>
for year in ['2007']:
  devkit_path = '/userhome/Datasets/foggy2city'
  for split in ['test', 'trainval']:
    name = 'foggy2city_{}_{}'.format(year, split)
    __sets[name] = (lambda year=year, split=split, devkit_path=devkit_path : foggy2city(split,year, devkit_path=devkit_path))
# Set up pl_foggy_cityscape_<year>_<split> 
for year in ['2007']:
  devkit_path = '/userhome/Datasets/pl_foggy_cityscape'
  for split in ['test', 'trainval']:
    name = 'pl_foggy_cityscape_{}_{}'.format(year, split)
    __sets[name] = (lambda year=year, split=split, devkit_path=devkit_path : pl_foggy_cityscape(split,year, devkit_path=devkit_path))
    
    
# Set up sim10k_<year>_<split> 
for year in ['2012']:
  devkit_path = '/userhome/Datasets/Sim10k'
  for split in ['trainval']:
    name = 'sim10k_{}_{}'.format(year, split)
    __sets[name] = (lambda year=year, split=split, devkit_path=devkit_path : sim10k(split,year, devkit_path=devkit_path))
# Set up cityscape_car_<year>_<split>    
for year in ['2007']:
  devkit_path = '/userhome/Datasets/cityscape'
  for split in ['test', 'trainval']:
    name = 'cityscape_car_{}_{}'.format(year, split)
    __sets[name] = (lambda year=year, split=split, devkit_path=devkit_path : cityscape_car(split,year, devkit_path=devkit_path))
# Set up sim10k2city_<year>_<split> 
for year in ['2012']:
  devkit_path = '/userhome/Datasets/sim10k2city'
  for split in ['trainval']:
    name = 'sim10k2city_{}_{}'.format(year, split)
    __sets[name] = (lambda year=year, split=split, devkit_path=devkit_path : sim10k2city(split,year, devkit_path=devkit_path))
# Set up city2sim10k_<year>_<split>    
for year in ['2007']:
  devkit_path = '/userhome/Datasets/city2sim10k'
  for split in ['test', 'trainval']:
    name = 'city2sim10k_{}_{}'.format(year, split)
    __sets[name] = (lambda year=year, split=split, devkit_path=devkit_path : city2sim10k(split,year, devkit_path=devkit_path))
# Set up pl_cityscape_<year>_<split>    
for year in ['2007']:
  devkit_path = '/userhome/Datasets/pl_cityscape'
  for split in ['test', 'trainval']:
    name = 'pl_cityscape_{}_{}'.format(year, split)
    __sets[name] = (lambda year=year, split=split, devkit_path=devkit_path : pl_cityscape(split,year, devkit_path=devkit_path))

    
    
    
# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2014_cap_<split>
for year in ['2014']:
  for split in ['train', 'val', 'capval', 'valminuscapval', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
for version in ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']:
    for split in ['minitrain', 'smalltrain', 'train', 'minival', 'smallval', 'val', 'test']:
        name = 'vg_{}_{}'.format(version,split)
        __sets[name] = (lambda split=split, version=version: vg(version, split))
        
# set up image net.
for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = 'data/imagenet/ILSVRC/devkit'
    data_path = 'data/imagenet/ILSVRC'
    __sets[name] = (lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split,devkit_path,data_path))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
