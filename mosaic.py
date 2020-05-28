import cv2
import numpy as np
import os
import glob2
import tqdm
import argparse
import time

ap = argparse.ArgumentParser(description="Create Image Mosaic")
ap.add_argument("-i", "--image", default="me.jpg",
                help="Path to the content image")
ap.add_argument("-d", "--datasets", default="images",
                help="Path to the images datasets")
ap.add_argument("-r", "--division", default=16, type=int,
                help="Divides the image n division. Default is 32. Higher value leads to better mosaic but it takes more time. ")
ap.add_argument("-s", "--size", nargs='+', default=None, type=int,
                help="Output size of the image")
ap.add_argument('-o', '--output', default="output.jpg",
                help="Path to save the image with filename ")
args = vars(ap.parse_args())
WEIGHT = 0.7


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

    def initDatasets(self):
        '''Load the datasets/images and set their RGB values in colors dictionary '''
        types = ('*.png', '*.jpg', '*.jpeg')  # the tuple of file types
        images = []
        for files in types:
            images.extend(glob2.glob(self.imagesPath+'/'+files))
        print("Loading Images dataset")
        start = time.time()
        if(len(self.colors) == 0):
            for imgPath in tqdm.tqdm(images):
                img = cv2.imread(imgPath)
                img = cv2.resize(
                    img, (self.W//self.division, self.H//self.division))
                mean = np.mean(img, axis=(0, 1))
                mean = np.asarray(mean, dtype=np.uint8)
                name = os.path.basename(imgPath)

                self.colors[name] = mean
        print('Loading Images dataset.... Done')
        print("Total ", time.time()-start)

    def loadAndResize(self, path, size):

        img = cv2.imread(path)
        img = cv2.resize(img, size)
        return img

    def loadContent(self, ):
        '''Load and resize content Image'''
        self.content = cv2.imread(self.contentPath)
        if self.contentSize is not None:
            self.content = cv2.resize(self.content, self.contentSize)
        self.H, self.W, self.C = self.content.shape
        return self.content

    def mosaicify(self):
        pixelHeight, pixelWidth = self.H//self.division, self.W//self.division
        print("Creating Mosaics.. This may take a while")
        dynamicMean = {}
        start = time.time()
        for row in tqdm.tqdm(range(0, self.H, pixelHeight)):
            for col in tqdm.tqdm(range(0, self.W, pixelWidth)):
                roi = self.content[row:row + pixelHeight, col:col+pixelWidth]
                mean = np.mean(roi, axis=(0, 1))
                mean = np.asarray(mean, dtype=np.float32)/255.0
                dynamicAverage = np.mean(mean*255.0).astype(np.uint8)

                minDistance = 10
                fileName = ''

                if(dynamicAverage in dynamicMean.keys()):
                    fileName = dynamicMean[dynamicAverage]
                    minImage = self.loadAndResize(
                        os.path.join(self.imagesPath, fileName), (roi.shape[1], roi.shape[0]))
                    self.content[row:row + pixelHeight,
                                 col:col+pixelWidth] = cv2.addWeighted(minImage, WEIGHT, roi, (1-WEIGHT), 1)
                    continue

                for name, value in self.colors.items():
                    value = value / 255.0
                    dist = np.linalg.norm(
                        mean-value)

                    if(dist < minDistance):
                        fileName = name
                        minDistance = dist

                if(fileName != ''):
                    dynamicMean[dynamicAverage] = fileName

                    minImage = self.loadAndResize(
                        os.path.join(self.imagesPath, fileName), (roi.shape[1], roi.shape[0]))
                    self.content[row:row + pixelHeight,
                                 col:col+pixelWidth] = cv2.addWeighted(minImage, WEIGHT, roi, (1-WEIGHT), 1)
                    # self.content[row:row + pixelHeight,
                    #              col:col+pixelWidth] = minImage
        print("Mosaic Creating Complete... Saving image")
        cv2.imwrite(args['output'], self.content)
        print("Image Saved..." + args['output'])
        print("Total Time taken : ", time.time()-start, 'secs')
        return self.content


mosaic = Mosaic(args['image'], args['datasets'],
                division=args['division'], contentSize=None if args['size'] is None else tuple(args['size']))
cv2.imshow('Output', mosaic.mosaicify())
cv2.waitKey(0)
