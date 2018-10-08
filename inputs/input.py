
#import tensorflow as tf
#import cv2
#from skimage import io as imgio
from configs import globals as gb
import scipy
from scipy import ndimage
from scipy import misc
import random
try:
    import queue
except ImportError:
    import Queue as queue
import threading
import numpy as np

from concurrent.futures import ThreadPoolExecutor

W=H=255

class Shape:
    def __init__(self, listfiles, label):
        self.label=label
        self.numb_chans = len(listfiles)
        self.views = self._load_views(listfiles)
        self.done_mean=False

    def _load_views(self, listfiles):
        views=[]
        for i in listfiles:
            im=scipy.ndimage.imread(gb.ROOT_DIR+i.replace("\n", ""))
            im=scipy.misc.imresize(im, (W,H))
            im=im.astype('float32')
            assert im.shape == (W,H,3), 'BGR!'
            views.append(im)
        views=np.asarray(views)
        return views

    def substract_image_mean(self):
        if not self.done_mean:
            for i in range(3):
                self.views[:,:,:,1]-=gb.BGRIMAGE_MEAN[i]
            self.done_mean=True

    def crop_center(self, crop_size=gb.INPUT_IMAGESIZE):
        w, h=self.views.shape[1], self.views.shape[2]
        wn, hn=crop_size
        left = int(w / 2 - wn / 2)
        top = int(h / 2 - hn / 2)
        right = left + wn
        bottom = top + hn
        self.views = self.views[:, left:right, top:bottom, :]


class Dataset:
    def __init__(self, listfiles, labels, flag_substract_mean=gb.FLAG_SUBSTRACT_MEAN, numb_chan=gb.NUMB_CHANNELS, flag_shuffel=gb.FLAG_SHUFFLE):
        self.listfiles=listfiles
        self.labels=labels
        self.flag_substract_mean=flag_substract_mean
        self.numb_chan=numb_chan
        self.flag_shuffel=flag_shuffel

    ## dataset pre-processing: shuffule; substract mean image; crop OR you can define more pre-processing operations here:
    def shuffle(self):
        listfiles_labels = list(zip( self.listfiles, self.labels ))
        random.shuffle(listfiles_labels)
        self.listfiles, self.labels=zip(*listfiles_labels)

    ## load shapes
    def _load_shape(self, listfile_label):
        s=Shape(listfile_label[0], listfile_label[1])
        s.crop_center()
        if self.flag_substract_mean:
            s.substract_image_mean()
        return s

    def batches(self, batch_size=gb.BATCH_SIZE):
        for x,y in self._batches_fast(self.listfiles, self.labels, batch_size):
            yield x,y

    def _batches(self, batch_size=gb.BATCH_SIZE):
        n = len(self.listfiles)
        for i in range(0, n, batch_size):
            lists = self.listfiles[i : i+batch_size]
            labels = self.labels[i : i+batch_size]
            x = np.zeros((batch_size, self.numb_chan, gb.INPUT_IMAGESIZE[0], gb.INPUT_IMAGESIZE[1], 3))
            y = np.zeros(batch_size)

            for j in  range(len(lists)):
                s = Shape(lists[j], labels[j])
                s.crop_center()
                if self.flag_substract_mean:
                    s.substract_image_mean()
                x[j, ...] = s.views
                y[j] = s.label
            yield x, y

    def _batches_fast(self, listfiles, labels, batch_size):
        flag_substract_mean = self.flag_substract_mean
        numb_files = len(listfiles)

        def load(q, listfiles, labels, batch_size):
            n=len(listfiles)
            with ThreadPoolExecutor(max_workers=16) as pool:
                for i in list(range(0, n, batch_size)):
                    sub_listfiles=listfiles[i:i+batch_size] if i<n-1 else [listfiles[-1]]
                    sub_labels=labels[i:i+batch_size] if i<n-1 else [listfiles[-1]]
                    shapes=list(pool.map(self._load_shape, zip(sub_listfiles, sub_labels)))
                    views=np.array([s.views for s in shapes])
                    labels=np.array([s.label for s in shapes])
                    print('haha3 %d-%d'%(i/batch_size, n))
                    q.put((views,labels))
            q.put(None)
        q=queue.Queue(maxsize=gb.INPUT_QUEUE_SIZE)
        p=threading.Thread( target=load, args=(q, listfiles, labels, batch_size) )
        p.daemon=True
        p.start()

        x=np.zeros((batch_size, self.numb_chan, gb.INPUT_IMAGESIZE[0], gb.INPUT_IMAGESIZE[1], 3))
        y=np.zeros(batch_size)

        for j in range(0, numb_files, batch_size):
            item=q.get()
            print('haha1 %d'%(j))
            if item is None:
                print('haha2')
                break

            x,y=item
            yield x,y

    def size(self):
        return len(self.listfiles)



