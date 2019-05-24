from PIL import Image
from torch.utils.data import Dataset
import torch

class Datasetee17(Dataset):
    def __init__(self,root, datatxt, transform=None,target_transform=None):
        super(Datasetee17,self).__init__()
        fh = open(root + datatxt, 'r')
        origin_f = fh.readlines()
        fh.close()
        length_f = len(origin_f)
        imgs = []
        for i in range(length_f):
            line = origin_f[i]
            words = line.split()
            words[0] = words[0]+'.jpg'
            label = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            for k in range(1,len(words)):
                words[k] = int(words[k])
                label[words[k]] = 1
            imgs.append((words[0],label))

        self.imgs = imgs
        self.transform = transform
        self.root = root
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, target = self.imgs[index]
        #print(target)
        img = Image.open(self.root+'/JPEGImages/'+fn).convert('RGB') 
        target = torch.FloatTensor(target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img,target,fn

    def __len__(self):
        return len(self.imgs)

