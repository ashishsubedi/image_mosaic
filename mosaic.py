import cv2
import numpy as np
import os
import glob2
import tqdm


class Mosaic:
    def __init__(self, contentPath, dataPath='images', division=32, contentSize=None):
        '''Create photo mosaic with content image'''
        self.contentPath = contentPath
        self.imagesPath = dataPath
        self.content = None
        self.H = None
        self.W = None
        self.C = None
        self.ext = os.path.splitext(contentPath)[1]
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
        for imgPath in tqdm.tqdm(images):
            img = cv2.imread(imgPath)
            img = cv2.resize(
                img, (self.W//self.division, self.H//self.division))
            mean = np.mean(img, axis=(0, 1))
            mean = np.asarray(mean, dtype=np.uint8)
            name = os.path.basename(imgPath)

            self.colors[name] = mean
        print('Loading Images dataset.... Done')

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
        for row in tqdm.tqdm(range(0, self.H, pixelHeight)):
            for col in range(0, self.W, pixelWidth):
                roi = self.content[row:row + pixelHeight, col:col+pixelWidth]
                mean = np.mean(roi, axis=(0, 1))
                mean = np.asarray(mean, dtype=np.uint8)
                dynamicAverage = np.mean(mean)
                present = False
                minDistance = np.inf
                fileName = ''
                if(dynamicAverage in dynamicMean.keys()):
                    fileName = dynamicMean[dynamicAverage]
                    minImage = self.loadAndResize(
                        os.path.join(self.imagesPath, fileName), (roi.shape[1], roi.shape[0]))
                    self.content[row:row + pixelHeight,
                                 col:col+pixelWidth] = minImage
                    continue
                for name, value in self.colors.items():
                    dist = np.linalg.norm(mean-value)
                    if(dist < minDistance):
                        fileName = name
                        minDistance = dist

                dynamicMean[dynamicAverage] = fileName
                minImage = self.loadAndResize(
                    os.path.join(self.imagesPath, fileName), (roi.shape[1], roi.shape[0]))
                self.content[row:row + pixelHeight,
                             col:col+pixelWidth] = minImage
        print("Mosaic Creating Complete... Saving image")
        cv2.imwrite('output'+self.ext, self.content)
        print("Image Saved")


mosaic = Mosaic('me.jpg', 'images_test')

mosaic.mosaicify()
cv2.imshow('asd', mosaic.content)

cv2.waitKey(0)
