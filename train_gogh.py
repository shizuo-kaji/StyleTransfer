###
# An implementation of 'A Neural Algorithm of Artistic Style' (http://arxiv.org/abs/1508.06576)
# code based on https://github.com/pfnet-research/chainer-gogh
# and https://github.com/yusuketomoto/chainer-fast-neuralstyle
#
# requirements: chainer, chainercv, Pillow
#
# S. Kaji, 04 July 2018
#

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable, optimizers, serializers


## trancated VGG
class VGG(chainer.Chain):
    def __init__(self):
        super(VGG, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),
            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1)
        )

    def __call__(self, x):
        y1 = F.relu(self.conv1_2(F.relu(self.conv1_1(x))))
        h = F.max_pooling_2d(y1, 2, stride=2)
        y2 = F.relu(self.conv2_2(F.relu(self.conv2_1(h))))
        h = F.max_pooling_2d(y2, 2, stride=2)
        y3 = F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(h))))))
        h = F.max_pooling_2d(y3, 2, stride=2)
        y4 = F.relu(self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(h))))))
        return [y1, y2, y3, y4]

## returns an image in range [0,255]
def postprocess(var):
    if args.gpu >= 0:
        img = var.data.get() + 120
    else:
        img = var.data+120
    img = np.clip(img,0,255)
    img = img.transpose(0, 2, 3, 1)   
    if img.shape[3]==1:
        img=img[:,:,:,0]   
    return np.uint8(img[0,:,:,::-1])    # BGR => RGB

## gram matrix (style matrix)
def gram_matrix(y):
    b, ch, h, w = y.data.shape
    features = F.reshape(y, (b, ch, w*h))
    gram = F.batch_matmul(features, features, transb=True)/np.float32(ch*w*h)
    return gram

## return [0,255]-120 image of shape (3,h,h)
def load_image(path, size=None, removebg=False):
    img = Image.open(path)
    if removebg:
        bkgnd = Image.new('RGBA', img.size, 'black')
        img = Image.alpha_composite(bkgnd, img).convert("RGB")
    if size:
        img = img.resize(size,Image.LANCZOS)
#    Image.fromarray(np.uint8(img).transpose((1,2,0))).save("b.jpg")
    img = np.asarray(img)[:,:,:3].transpose(2, 0, 1)[::-1].astype(np.float32) -120  # BGR
    return(xp.asarray(img[np.newaxis,:]))

## the main image generation routine
def generate_image(image, style, args, img_gen=None):
    ## convolution kernel for total variation
    wh = xp.asarray([[[[1], [-1]], [[0], [0]], [[0], [0]]], [[[0], [0]], [[1], [-1]], [[0], [0]]], [[[0], [0]], [[0], [0]], [[1], [-1]]]], dtype=np.float32)
    ww = xp.asarray([[[[1, -1]], [[0, 0]], [[0, 0]]], [[[0, 0]], [[1, -1]], [[0, 0]]], [[[0, 0]], [[0, 0]], [[1, -1]]]], dtype=np.float32)
    ## compute style matrix
    with chainer.using_config('train', False):
        feature = nn(Variable(image))
        feature_s = nn(Variable(style))
    gram_s = [gram_matrix(y) for y in feature_s]

    ## randamise initial image
    if img_gen is None:
        if args.gpu >= 0:
            img_gen = xp.random.uniform(-20,20,image.shape,dtype=np.float32)
        else:
            img_gen = np.random.uniform(-20,20,image.shape).astype(np.float32)

    ## image optimisation
    img_gen = L.Parameter(img_gen)
    optimizer = optimizers.Adam(alpha=args.lr)
    optimizer.setup(img_gen)
    print("losses: feature, style, variance, total")
    for i in range(args.iter):
        img_gen.zerograds()

        with chainer.using_config('train', False):
            y = nn(img_gen.W)

        # feature consistency evaluated on the output of layer conv3_3
        L_feat = args.lambda_feat * F.mean_squared_error(feature[2], y[2])
        # style resemblance evaluated on the output of four layers
        L_style = Variable(xp.zeros((), dtype=np.float32))
        for f_hat, g_s in zip(y, gram_s):
            L_style += args.lambda_style * F.mean_squared_error(gram_matrix(f_hat), g_s)
        # surpress total variation
        L_tv = args.lambda_tv * (F.sum(F.convolution_2d(img_gen.W, W=wh) ** 2) + F.sum(F.convolution_2d(img_gen.W, W=ww) ** 2))
        loss = L_feat + L_style + L_tv

        loss.backward()
        optimizer.update()

        ## clip to [0,255]-120
        img_gen.W.data = xp.clip(img_gen.W.data, -120.0, 136.0)

        if i%20==0:
            print('iter {}/{}... loss: {}, {}, {}, {}'.format(i, args.iter, L_feat.data, L_style.data, L_tv.data, loss.data))

        ## output image
        if (i % args.interval == 0) and i>0:
            med = postprocess(img_gen.W)
            print("image range {} -- {}".format(np.min(med),np.max(med)))
            med = Image.fromarray(med)
            if args.median_filter > 0:
                med = med.filter(ImageFilter.MedianFilter(args.median_filter))
            med.save(os.path.join(args.out,'count{:0>4}.jpg'.format(i)))


#########################
parser = argparse.ArgumentParser(description='A Neural Algorithm of Artistic Style')
parser.add_argument('input',  default=None,
                    help='Original image')
parser.add_argument('--style', '-s', default=None,
                    help='Style image')
parser.add_argument('--out', '-o', default='result',
                    help='Output directory')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--iter', default=5000, type=int,
                    help='number of iterations')
parser.add_argument('--interval', default=500, type=int,
                    help='image output interval')
parser.add_argument('--lr', default=4.0, type=float,
                    help='learning rate')
parser.add_argument('--lambda_tv', default=1e-6, type=float,
                    help='weight of total variation regularization')
parser.add_argument('--lambda_feat', default=0.01, type=float,
                    help='weight for the original shape; increase to retain it')
parser.add_argument('--lambda_style', default=1.0, type=float,
                    help='weight for the style')
parser.add_argument('--median_filter', '-f', default=0, type=int,
                    help='apply median filter to the output')
parser.add_argument('--random_start', '-rs', action='store_true',
                    help="start optimisation using random image, otherwise use the input image as the initial")
parser.add_argument('--removebg', '-nbg', action='store_true',
                    help="remove background specified by alpha in png")

args = parser.parse_args()
print(args)

try:
    os.mkdir(args.out)
except:
    pass

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    xp = cuda.cupy
else:
    print("runs desperately slowly without a GPU!")
    xp = np

## use pretrained VGG for feature extraction
nn = VGG()
serializers.load_npz('vgg16.model', nn)
if args.gpu>=0:
    nn.to_gpu()

## load images
image = load_image(args.input,removebg=args.removebg)
style = load_image(args.style,size=(image.shape[3],image.shape[2]), removebg=args.removebg)

print("input image:", image.shape, xp.min(image), xp.max(image))
print("style image:", style.shape, xp.min(style), xp.max(style))

## initial image: original or random
if args.random_start:
    generate_image(image, style, args, img_gen=None)
else:
    generate_image(image, style, args, img_gen=image)
