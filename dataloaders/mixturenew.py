import os,sys
import os.path
import numpy as np
import torch
import torch.utils.data
from torchvision import datasets,transforms
from sklearn.utils import shuffle
import urllib.request
from PIL import Image
import pickle
import utils

########################################################################################################################

def get(data_path,seed=0,pc_valid=0.15, fixed_order=True):
    data={}
    taskcla=[]
    size=[3,32,32]

    idata=np.arange(5)
    if not fixed_order:
        idata=list(shuffle(idata,random_state=seed))
    print('Task order =',idata)

    path = os.path.join(data_path, 'binary_mixture')
    if not os.path.isdir(path):
        os.makedirs(path)
        # Pre-load
        for n,idx in enumerate(idata):
            if idx==0:
                # CIFAR10
                mean=[x/255 for x in [125.3,123.0,113.9]]
                std=[x/255 for x in [63.0,62.1,66.7]]
                dat={}
                dat['train']=datasets.CIFAR10(data_path,train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
                dat['test']=datasets.CIFAR10(data_path,train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
                data[n]={}
                data[n]['name']='cifar10'
                data[n]['ncla']=10
                for s in ['train','test']:
                    loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
                    data[n][s]={'x': [],'y': []}
                    for image,target in loader:
                        data[n][s]['x'].append(image)
                        data[n][s]['y'].append(target.numpy()[0])

            elif idx==1:
                # CIFAR100
                mean=[x/255 for x in [125.3,123.0,113.9]]
                std=[x/255 for x in [63.0,62.1,66.7]]
                dat={}
                dat['train']=datasets.CIFAR100(data_path,train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
                dat['test']=datasets.CIFAR100(data_path,train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
                data[n]={}
                data[n]['name']='cifar100'
                data[n]['ncla']=100
                for s in ['train','test']:
                    loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
                    data[n][s]={'x': [],'y': []}
                    for image,target in loader:
                        data[n][s]['x'].append(image)
                        data[n][s]['y'].append(target.numpy()[0])

            elif idx==2:
                # MNIST
                #mean=(0.1307,) # Mean and std without including the padding
                #std=(0.3081,)
                mean=(0.1,) # Mean and std including the padding
                std=(0.2752,)
                dat={}
                dat['train']=datasets.MNIST(data_path,train=True,download=True,transform=transforms.Compose([
                    transforms.Pad(padding=2,fill=0),transforms.ToTensor(),transforms.Normalize(mean,std)]))
                dat['test']=datasets.MNIST(data_path,train=False,download=True,transform=transforms.Compose([
                    transforms.Pad(padding=2,fill=0),transforms.ToTensor(),transforms.Normalize(mean,std)]))
                data[n]={}
                data[n]['name']='mnist'
                data[n]['ncla']=10
                for s in ['train','test']:
                    loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
                    data[n][s]={'x': [],'y': []}
                    for image,target in loader:
                        image=image.expand(1,3,image.size(2),image.size(3)) # Create 3 equal channels
                        data[n][s]['x'].append(image)
                        data[n][s]['y'].append(target.numpy()[0])

            elif idx == 3:
                # SVHN
                mean=[0.4377,0.4438,0.4728]
                std=[0.198,0.201,0.197]
                dat = {}
                dat['train']=datasets.SVHN(data_path,split='train',download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
                dat['test']=datasets.SVHN(data_path,split='test',download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
                data[n] = {}
                data[n]['name']='svhn'
                data[n]['ncla']=10
                for s in ['train','test']:
                    loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                    data[n][s] = {'x': [], 'y': []}
                    for image, target in loader:
                        data[n][s]['x'].append(image)
                        data[n][s]['y'].append(target.numpy()[0]-1)

            elif idx == 4:
                # usps
                mean=(0.1307,) # Mean and std including the padding
                std=(0.3081,)
                dat={}
                dat['train']=datasets.USPS(data_path,train=True,download=True,
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
                dat['test']=datasets.USPS(data_path,train=False,download=True,
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
                data[n]={}
                data[n]['name']='usps'
                data[n]['ncla']=10
                for s in ['train','test']:
                    loader=torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                    data[n][s]={'x': [], 'y': []}
                    for image,target in loader:
                        data[n][s]['x'].append(image)
                        data[n][s]['y'].append(target.numpy()[0])
            else:
                print('ERROR: Undefined data set',n)
                sys.exit()
            #print(n,data[n]['name'],data[n]['ncla'],len(data[n]['train']['x']))

            # "Unify" and save
            for s in ['train','test']:
                data[n][s]['x']=torch.stack(data[n][s]['x']).view(-1,size[0],size[1],size[2])
                data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)
                torch.save(data[n][s]['x'], os.path.join(os.path.expanduser(path),'data'+str(idx)+s+'x.bin'))
                torch.save(data[n][s]['y'], os.path.join(os.path.expanduser(path),'data'+str(idx)+s+'y.bin'))

    else:

        # Load binary files
        for n,idx in enumerate(idata):
            data[n] = dict.fromkeys(['name','ncla','train','test'])
            if idx==0:
                data[n]['name']='cifar10'
                data[n]['ncla']=10
            elif idx==1:
                data[n]['name']='cifar100'
                data[n]['ncla']=100
            elif idx==2:
                data[n]['name']='mnist'
                data[n]['ncla']=10
            elif idx==3:
                data[n]['name']='svhn'
                data[n]['ncla']=10
            elif idx==4:
                data[n]['name']='usps'
                data[n]['ncla']=10
            else:
                print('ERROR: Undefined data set',n)
                sys.exit()

            # Load
            for s in ['train','test']:
                data[n][s]={'x':[],'y':[]}
                data[n][s]['x'] = torch.load(os.path.join(os.path.expanduser(path),'data'+str(idx)+s+'x.bin'))
                data[n][s]['y'] = torch.load(os.path.join(os.path.expanduser(path),'data'+str(idx)+s+'y.bin'))

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