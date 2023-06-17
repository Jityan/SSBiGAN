from PIL import Image
import os
import os.path
import pickle
import random
import numpy as np

import torch
import torchvision.transforms as transforms


def processLabel(labels):
    newlabels = []
    temp = {}
    count = 0
    for _, label in enumerate(labels):
        if label not in temp:
            temp[label] = count
            count += 1
        newlabels.append(temp[label])
    return newlabels, count


class OxfordTextDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='./data/oxford', split='train', embedding_type='cnn-rnn', imsize=64, branch=2, transform=None):

        self.branch = branch
        self.imsize = []
        for i in range(self.branch):
            self.imsize.append(imsize)
            imsize = imsize * 2
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(self.imsize[-1] * 76 / 64)),
                transforms.RandomCrop(self.imsize[-1]),
                transforms.RandomHorizontalFlip()])
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.data = []
        self.data_dir = data_dir
        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir, embedding_type)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        # preprocess class label into 0 - max-1
        self.new_class_id, self.num_classes = processLabel(self.class_id)
        self.num_classes += 1

    def get_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        ret = []
        for i in range(self.branch):
            if i < (self.branch - 1):
                re_img = transforms.Resize(self.imsize[i])(img)
            else:
                re_img = img
            ret.append(self.norm(re_img))
        return ret


    def read_captions(self, filenames, class_id):
        name = filenames
        class_name = 'class_{0:05d}/'.format(class_id)
        name = name.replace('jpg/', class_name)
        cap_path = '{}/text_c10/{}.txt'.format(self.data_dir, name)
        
        with open(cap_path, "r") as f:
            captions = f.read().split('\n')
        captions = [cap for cap in captions if len(cap) > 0]
        return captions

    def load_embedding(self, data_dir, embedding_type):
        if embedding_type == 'cnn-rnn':
            embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'cnn-gru':
            embedding_filename = '/char-CNN-GRU-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            embedding_filename = '/skip-thought-embeddings.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f, encoding='latin1')
            embeddings = np.array(embeddings)
            #print('embeddings: ', embeddings.shape)
        return embeddings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding='latin1')
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        #print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames
    
    def load_wrong_images(self, cls_id):
        temp_id = random.randint(0, len(self.filenames)-1)
        w_id = self.class_id[temp_id]
        if cls_id != w_id:
            return self.filenames[temp_id], w_id
        return self.load_wrong_images(cls_id)

    def __getitem__(self, index):
        key = self.filenames[index]
        cls_id = self.class_id[index]
        wkey, wid = self.load_wrong_images(cls_id)
        #
        data_dir = self.data_dir
        captions = self.read_captions(key, self.class_id[index])
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/%s.jpg' % (data_dir, key)
        wimg_name = '%s/%s.jpg' % (data_dir, wkey)
        img = self.get_img(img_name)
        wimg = self.get_img(wimg_name)

        embedding_ix = random.randint(0, embeddings.shape[0]-1)
        embedding = embeddings[embedding_ix, :]
        caption = captions[embedding_ix]
        #
        idata = {
            'right_images': img,
            'wrong_images': wimg,
            'right_embed': embedding,
            'txt': str(caption),
            'cid': cls_id,
            'wcid': wid,
        }
        return idata

    def __len__(self):
        return len(self.filenames)

import sys
from torchvision.utils import save_image

if __name__ == "__main__":
    dataset = OxfordTextDataset('./data/oxford', 'train')
    assert dataset
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, drop_last=True, shuffle=True, num_workers=0)

    for i, sample in enumerate(dataloader, 1):
        rimg = sample['right_images']
        wimg = sample['wrong_images']
        emb = sample['right_embed']
        txt = sample['txt']
        cid = sample['cid']
        wcid = sample['wcid']
        save_image(rimg[0], 'rsmall.jpg', normalize=True)
        save_image(rimg[1], 'rbig.jpg', normalize=True)
        save_image(wimg[0], 'wsmall.jpg', normalize=True)
        save_image(wimg[1], 'wbig.jpg', normalize=True)
        print(i, ":", rimg[0].shape, rimg[1].shape, wimg[0].shape, wimg[1].shape, emb.shape, len(txt), cid, wcid)
        sys.exit()
    print("Complete...")