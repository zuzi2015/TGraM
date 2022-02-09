from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.jde import JointDataset
from .dataset.tgram import TgramDataset


def get_dataset(dataset, task, dataloader):
  if task == 'mot':
    if dataloader == 'jde':
      return JointDataset
    else:
      return TgramDataset
  else:
    return None
  
