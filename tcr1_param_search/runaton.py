import glob
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import tcr_cnn
from skimage.transform import resize
import json

def show(im):
    f, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(im)
    plt.show()

class nvesd_cnn_train(Dataset):
    def __init__(self):
        target_files  = glob.glob('/home/bruce/data/ATR/new_samplesets/unscaled_1to2/chips40x80/targets/' + '*.mat')
        clutter_files = glob.glob('/home/bruce/data/ATR/new_samplesets/unscaled_1to2/chips40x80/clutter/' + '*.mat')
        gaussian = loadmat('data/h.mat')['h']

        d1 = 40
        d2 = 80
        count = len(target_files)
        # print(count, 'target chips')
        count = len(clutter_files)
        # print(count, 'clutter chips')
        alltargets = np.zeros((d1, d2, count))
        for idx, chip in enumerate(target_files):
            chiparray = loadmat(chip)['target_chip']
            chiparray = chiparray - chiparray.mean()
            alltargets[:, :, idx] = chiparray

        allclutter = np.zeros((d1, d2, count))
        for idx, chip in enumerate(clutter_files):
            chiparray = loadmat(chip)['clutter_chip']
            chiparray = chiparray - chiparray.mean()
            allclutter[:, :, idx] = chiparray

        # print('clutter',allclutter.shape)


        yt = np.tile(gaussian,(10800,1,1))
        # print('yt',yt.shape)

        yc = np.tile(np.zeros((17,37)),(10800,1,1))
        # print('yc',yc.shape)

        self.x = np.concatenate((alltargets,allclutter),axis=2)
        # print('x',self.x.shape)

        self.y = np.concatenate((yt,yc),axis=0)
        # print('y',self.y.shape)

    def __len__(self):
        return self.x.shape[2]

    def __getitem__(self, idx):
        x = self.x[:,:,idx]
        x = np.expand_dims(x, axis=0)
        y = self.y[idx,:,:]
        y = np.expand_dims(y, axis=0)


        return x,y
def scale(image, factor):
    x, y = image.shape
    x = int(round(factor * x))
    y = int(round(factor * y))
    return resize(image,(x, y),mode='constant',anti_aliasing=True)

def pad(image,nrows,ncols):
    out = np.zeros((nrows,ncols))
    m, n = image.shape
    o1 = nrows/2 + 1
    o2 = ncols/2 + 1
    r1 = int(round(o1 - m/2))
    r2 = int(round(r1 + m - 1))
    c1 = int(round(o2 - n/2))
    c2 = int(round(c1 + n -1))
    out[r1:r2+1,c1:c2+1] = image
    return out

def get_detections(input_image,ndetects):
    image = input_image.copy()
    minval = image.min()
    # print('responses',image.min(),image.max())

    nrows,ncols = image.shape
    confs=[]
    row_dets=[]
    col_dets=[]
    for i in range(ndetects):
        row,col = np.unravel_index(image.argmax(), image.shape)
        val = image[row,col]
        r1 = max(row - 10, 0)
        r2 = min(r1 + 19, nrows)
        r1 = r2 - 19
        c1 = max(col - 20, 1)
        c2 = min(c1 + 39, ncols)
        c1 = c2 - 39
        image[r1: r2+1, c1:c2+1]=np.ones((20, 40)) * minval;
        confs.append(val)
        row_dets.append(row)
        col_dets.append(col)

    confs = np.array(confs)
    Aah = confs.std() * 6** .5 / 3.14158
    cent = confs.mean() - Aah * 0.577216649;
    confs = (confs - cent) / Aah;


    row_dets = np.array(row_dets)
    col_dets = np.array(col_dets)

    return confs, row_dets, col_dets
def train(epochs,lr):
    net = tcr_cnn.tcrNet_lite().cuda()
    # epochs = 30
    trainset = nvesd_cnn_train()

    trainloader = DataLoader(
        dataset=trainset,
        batch_size=100,
        num_workers=5,
        shuffle=True
    )
    results = []

    criterion = tcr_cnn.tcrLoss.apply
    # optimizer = optim.RMSprop(net.parameters(), lr=0.001, momentum=0.9, weight_decay=.01)
    # optimizer = optim.Adam(net.parameters(), lr=0.00001, weight_decay=.01)
    optimizer = optim.Adam(net.parameters(), lr=lr)


    # optimizer = optim.SGD(net.parameters(), lr=0.0001, weight_decay=.01)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=.1)

    losses = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(trainloader):
            x = data[0].float().cuda()
            gt = data[1].float().cuda()
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 10 == 0:    # print every 10 mini-batches
                losses.append(running_loss/10)
                # print('[%d, %5d] loss: %.8f' % (epoch, i, running_loss/10))
                running_loss = 0.0
        scheduler.step()
        # print('Epoch:', epoch, 'LR:', scheduler.get_lr(),'epoch loss',epoch_loss)

    torch.save(net.state_dict(), 'data/trainedweights.pth')
    losses = np.array(losses)
    np.save('data/losses',losses)
    # print('Finished Training')
def validate():
    # samples  = json.load(open('/home/bruce/data/ATR/new_samplesets/unscaled_25to35day/unscaled_25to35day.json'))
    samples = json.load(open('/home/bruce/projects/tcr_lab/data/nvesd1/datasets/far_day_decimated.json'))
    # samples  = json.load(open('../data/nvesd1/datasets/far_day.json'))
    # samples  = json.load(open('/home/bruce/data/ATR/new_samplesets/unscaled_25to35night/unscaled_25to35night.json'))

    imgdir = '/home/bruce/data/ATR/matlab/'
    # samples = samples[3500:]

    net = tcr_cnn.tcrNet_lite().cuda()
    net.load_state_dict(torch.load('data/trainedweights.pth'))

    index = 0
    dets = []
    fas = []
    nframes = 0
    ntgt = 0
    for sample in samples:
        # print(sample['name'] + '_' + sample['frame'])
        imfile = imgdir + sample['name'] + '_' + sample['frame'] + '.mat'
        im = loadmat(imfile)['image']
        target_range = sample['range'] * 1000
        scale_factor = target_range / 2500

        im = scale(im, scale_factor)
        nrows, ncols = im.shape
        # show(im)
        im = torch.tensor(im).unsqueeze(0).unsqueeze(0).float().cuda()
        output = net(im)
        # output = output**2
        output = output.cpu().detach()[0, 0, :, :].numpy()
        # show(output)
        output = pad(output, nrows, ncols)
        confs, row_dets, col_dets = get_detections(output, 20)
        row_dets = row_dets / scale_factor
        col_dets = col_dets / scale_factor

        targets = sample['targets']
        nt = len(targets)
        ndets = confs.shape[0]
        ntgt += nt
        nframes += 1

        foundtgt = np.zeros(ndets);

        for target in targets:
            r = target['center'][1]
            c = target['center'][0]
            tmpdets = []
            for i in range(ndets):
                dist = ((r - row_dets[i]) ** 2 + (c - col_dets[i]) ** 2) ** .5
                # print('dist',dist)
                if dist < 20:
                    foundtgt[i] = 1
                    tmpdets.append(confs[i])
            if len(tmpdets) >= 1:
                dets.append(max(tmpdets))
            # print('dets',dets)
            I = np.where(foundtgt == 0)[0]
            for a in confs[I]:
                fas.append(a)
            # print('fas',fas)

    dets = np.array(dets)
    np.save('data/dets', dets)
    fas = np.array(fas)
    np.save('data/fas', fas)
    ntgt = np.array([ntgt])
    np.save('data/ntgt', ntgt)
    nframes = np.array([nframes])
    np.save('data/nframes', nframes)
    # print(ntgt, nframes, len(dets), len(fas))
def roc():
    dets = np.load('data/dets.npy')
    fas = np.load('data/fas.npy')
    ntgt = np.load('data/ntgt.npy')[0]
    nframes = np.load('data/nframes.npy')[0]
    # print(ntgt, nframes, len(dets), len(fas))

    maxv = max(max(fas), max(dets))
    minv = min(min(fas), min(dets))
    step = (maxv - minv) / 1000;
    # print(maxv, minv)
    pds = []
    fars = []
    t = minv
    while t < maxv:
        x = np.where(dets > t)
        pd = x[0].shape / ntgt
        pds.append(pd)

        y = np.where(fas > t)
        far = y[0].shape / (nframes * 3.4 * 2.6)
        fars.append(far)
        t += step

    print('max pd', pds[0])
    pds = np.array(pds)
    fars = np.array(fars)
    fars_under_one = np.where(fars < 1)
    pds_under_one = pds[fars_under_one]
    aoc = pds_under_one.sum() / pds_under_one.shape
    print('AOC',aoc)
    plt.title(str(aoc))
    plt.plot(fars, pds)
    plt.savefig('data/roc.png')
    plt.show()


if __name__ == '__main__':
    epochs = 2
    lr = .00001
    print('epochs',epochs)
    print('lr',lr)
    for idx in range(10):
        print('*********************** trial',idx)
        train(epochs,lr)
        validate()
        roc()




