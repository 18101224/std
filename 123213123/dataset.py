from torch.utils.data import Dataset
import numpy as np
import glob
from PIL import Image
from collections import defaultdict
from torch.utils.data import DataLoader

class Affect(Dataset):
    def __init__(self,img_path):
        '''

        :param img_path: the img_path does not have '/'
        '''
        super(Affect).__init__()
        self.img_list=glob.glob(img_path+'/*')
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self,idx):
        img = Image.open(self.img_list[idx])
        img = np.asarray(img)
        img = img.transpose(2,0,1)
        return img


def data_dict(root_path):
    '''

    :param root_path: the root path ends with just name of directory not including the '/'
    :return: return the dictionary that have dataset instances and the keys are label
    '''
    result = defaultdict(None)
    for i in range(8):
        result[i]=Affect(root_path+'/'+str(i))
    return result

def loader_dict(datasets,bs):
    '''
    :param datasets: the dictionary of datasets
    :param bs: batch size
    :return: the dictionary that has dataloders of the datasets
    '''
    result = defaultdict(None)
    for i,dataset in datasets.items():
        result[i] = DataLoader(dataset,batch_size=bs,shuffle=True)
    return result