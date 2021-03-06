import cv2
import numpy as np
import os
import glob2
import tqdm
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

WEIGHT = 0.8


ap = argparse.ArgumentParser(description="Create Image Mosaic")
ap.add_argument("-i", "--image", default="me.jpg",
                help="Path to the content image")
ap.add_argument("-d", "--datasets", default="images",
                help="Path to the images datasets")
ap.add_argument("-r", "--division", default=32, type=int,
                help="Divides the image n division. Default is 32. Higher value leads to better mosaic but it takes more time. ")
ap.add_argument("-s", "--size", nargs='+', default=None, type=int,
                help="Output size of the image")
ap.add_argument('-o', '--output', default="output_multi.jpg",
                help="Path to save the image with filename ")
args = vars(ap.parse_args())


class Mosaic:
    def __init__(self, contentPath, dataPath='images', division=32, contentSize=None):
        '''Create photo mosaic with content image'''
        self.contentPath = contentPath
        self.imagesPath = dataPath
        self.content = None
        self.H = None
        self.W = None
        self.C = None
        self.colors = {}
        self.division = division
        self.contentSize = contentSize
        self.loadContent()
        self.initDatasets()

    def _process_img(self, imgPath):
        img = cv2.imread(imgPath)
        img = cv2.resize(
            img, (self.W//self.division, self.H//self.division))
        mean = np.mean(img, axis=(0, 1))
        mean = np.asarray(mean, dtype=np.uint8)
        name = os.path.basename(imgPath)
        self.colors[name] = mean

    def initDatasets(self):
        '''Load the datasets/images and set their RGB values in colors dictionary '''
        types = ('*.png', '*.jpg', '*.jpeg')  # the tuple of file types
        images = []
        for files in types:
            images.extend(glob2.glob(self.imagesPath+'/'+files))
        print("Loading Images dataset")
        start = time.time()

        if(len(self.colors) == 0):
            with ThreadPoolExecutor() as executer:
                executer.map(self._process_img, tqdm.tqdm(images))
                # executer.shutdown(wait=True)

        print('Loading Images dataset.... Done')
        print("Total ", time.time()-start, 'secs')

    def loadAndResize(self, path, size):

        img = cv2.imread(path)
        img = cv2.resize(img, size)
        return img

    def loadContent(self):
        '''Load and resize content Image'''
        self.content = cv2.imread(self.contentPath)
        if self.contentSize is not None:
            self.content = cv2.resize(self.content, self.contentSize)
        self.H, self.W, self.C = self.content.shape
        return self.content

    def _replaceTile(self, col):
        row = self.row
        roi = self.content[row:row + self.pixelHeight, col:col+self.pixelWidth]

        mean = np.mean(roi, axis=(0, 1))
        mean = np.asarray(mean, dtype=np.float32)/255.0
        dynamicAverage = np.mean(mean*255.0).astype(np.uint8)

        minDistance = 10
        fileName = ''

        if(dynamicAverage in self.dynamicMean.keys()):
            fileName = self.dynamicMean[dynamicAverage]
            minImage = self.loadAndResize(
                os.path.join(self.imagesPath, fileName), (roi.shape[1], roi.shape[0]))
            self.content[row:row + self.pixelHeight,
                         col:col+self.pixelWidth] = cv2.addWeighted(minImage, WEIGHT, roi, (1-WEIGHT), 0)
            return

        for name, value in self.colors.items():
            value = value / 255.0
            dist = np.linalg.norm(
                mean-value)

            if(dist < minDistance):
                fileName = name
                minDistance = dist

        if(fileName != ''):
            self.dynamicMean[dynamicAverage] = fileName

            minImage = self.loadAndResize(
                os.path.join(self.imagesPath, fileName), (roi.shape[1], roi.shape[0]))
            self.content[row:row + self.pixelHeight,
                         col: col+self.pixelWidth] = cv2.addWeighted(minImage, WEIGHT, roi, (1-WEIGHT), 0)

    def mosaicify(self):
        self.pixelHeight, self.pixelWidth = self.H//self.division, self.W//self.division

        print("Creating Mosaics.. This may take a while")
        self.dynamicMean = {}
        start = time.time()
        for row in tqdm.tqdm(range(0, self.H, self.pixelHeight)):
            cols = range(0, self.W, self.pixelWidth)
            with ThreadPoolExecutor() as executer:
                self.row = row
                executer.map(self._replaceTile, cols)

        print("Mosaic Creating Complete... Saving image")
        cv2.imwrite(args['output'], self.content)
        print("Image Saved..." + args['output'])
        print("Total Time taken : ", time.time()-start, 'secs')
        return self.content


mosaic = Mosaic(args['image'], args['datasets'],
                division=args['division'], contentSize=None if args['size'] is None else tuple(args['size']))

mosaic.mosaicify()

cv2.imshow('Output', mosaic.content)
cv2.waitKey(0)
