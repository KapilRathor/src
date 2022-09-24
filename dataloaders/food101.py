import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle

def get(data_path,seed=0,pc_valid=0.10):
    data={}
    taskcla=[]
    size=[3,384,384]

    path = os.path.join(data_path, 'binary_food101')
    if not os.path.isdir(path):
        os.makedirs(path)

        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]

        # Food101
        dat={}
        n=0
        data[n]={}
        data[n]['name']='food101'
        data[n]['ncla']=101
        data[n]['train']={'x': [],'y': []}
        data[n]['test']={'x': [],'y': []}
        dat['train']=datasets.Food101(data_path,split = 'train',download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.Food101(data_path,split = 'test',download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            for image,target in loader:
                data[n][s]['x'].append(image)
                data[n][s]['y'].append(target.numpy()[0])

        # "Unify" and save
        for t in data.keys():
            for s in ['train','test']:
                data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,size[0],size[1],size[2])
                data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
                torch.save(data[t][s]['x'], os.path.join(os.path.expanduser(path),'data'+str(t)+s+'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(os.path.expanduser(path),'data'+str(t)+s+'y.bin'))

    # Load binary files
    data={}
    ids=0
    i=0
    data[i] = dict.fromkeys(['name','ncla','train','test'])
    data[i]['name']='food101'
    data[i]['ncla']=101
    for s in ['train','test']:
        data[i][s]={'x':[],'y':[]}
        data[i][s]['x']=torch.load(os.path.join(os.path.expanduser(path),'data'+str(ids)+s+'x.bin'))
        data[i][s]['y']=torch.load(os.path.join(os.path.expanduser(path),'data'+str(ids)+s+'y.bin'))

    # Validation
    for t in data.keys():
        r=np.arange(data[t]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size
