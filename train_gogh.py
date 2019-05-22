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
from chainer import cuda, Variable, optimizers, serializers, training
from chainer.training import extensions

## returns an image in range [0,255]
def postprocess(var, gpu):
    if gpu >= 0:
        img = var.array.get()
    else:
        img = var.array
    img = img.transpose(0, 2, 3, 1)   
    img = np.clip(img[0] + np.array([103.939, 116.779, 123.68]),0,255)
    if img.shape[2]==1:
        return np.uint8(img[:,:,0])
    else:
        return np.uint8(img[:,:,::-1])    # BGR => RGB

## gram matrix (style matrix)
def gram_matrix(y):
    b, ch, h, w = y.data.shape
    features = F.reshape(y, (b, ch, w*h))
    gram = F.batch_matmul(features, features, transb=True)/np.float32(ch*w*h)
    return gram

## return [0,255]-120 image of shape (1,h,h)
def load_dicom(path,base,rng,h,scale):
    import pydicom as dicom
    from chainercv.transforms import center_crop
    from skimage.transform import rescale
    ref_dicom = dicom.read_file(path, force=True)
    ref_dicom.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
    img = ref_dicom.pixel_array.astype(np.float32)+ref_dicom.RescaleIntercept
    if scale != 1.0:
        img = rescale(img,scale,mode="reflect",preserve_range=True)
    img = (np.clip(img,base,base+rng)-base)/rng * 255 - np.array([103.939, 116.779, 123.68])
    img = center_crop(img[np.newaxis,:],(h,h))
    return(img.astype(np.float32))

## return [0,255] image of shape (3,h,w)
def load_image(path, size, removebg=False):
    img = Image.open(path)
    if removebg:
        bkgnd = Image.new('RGBA', img.size, 'black')
        img = Image.alpha_composite(bkgnd, img).convert("RGB")
    if size[0]>0 and size[1]>0:
        img = img.resize(size,Image.LANCZOS)
    img = np.asarray(img)[:,:,:3]
    img = img[:,:,::-1]-np.array([103.939, 116.779, 123.68])  # BGR
    img = img.transpose(2, 0, 1)
    return(img[np.newaxis,:].astype(np.float32))

class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.img_gen, self.perceptual = kwargs.pop('models')
        params = kwargs.pop('params')
        self.args = params['args']
        self.image = params['image']  ## contents image
        self.bkgnd = params['bkgnd']  ## background image
        self.reduced_feature = params['reduced_feature']  ## reduced size contents image
        self.layers = params['layers']  ## layers used to compute image and style losses
        self.gram_s = params['gram_s']   ## gram matrix of style image
        self.feature = params['feature']   ## feature of contents image
        self.layer_id = 2 ## which layer output to use for contents comparison; default = 2
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        optimizer = self.get_optimizer('main')
        self.img_gen.cleargrads()
        xp = self.img_gen.xp
        # get mini-batch
#        batch = self.get_iterator('main').next()
#        img = self.converter(batch, self.device)
#        x = Variable(img)

        with chainer.using_config('train', False):
            y = self.perceptual(self.img_gen.W,layers=self.layers)

        # feature consistency; experiment with different layers to use
        L_feat = F.mean_squared_error(self.feature[self.layers[self.layer_id]], y[self.layers[self.layer_id]])

        # feature consistency for reduced image
        if self.args.lambda_rfeat > 0:
            with chainer.using_config('train', False):
                yr = self.perceptual(F.average_pooling_2d(self.img_gen.W,self.args.ksize),layers=self.layers)
            L_feat_r = F.mean_squared_error(self.reduced_feature[self.layers[self.layer_id]], yr[self.layers[self.layer_id]])
        else:
            L_feat_r = 0
        
        # style resemblance evaluated on the output of all layers
        if self.args.lambda_style > 0:
            L_style = sum([F.mean_squared_error(gram_matrix(y[k]), self.gram_s[k]) for k in self.layers])
        else:
            L_style = 0

        # suppress total variation
        L_tv = F.average(F.absolute(self.img_gen.W[:,:,1:,:]-self.img_gen.W[:,:,:-1,:]))+F.average(F.absolute(self.img_gen.W[:,:,:,1:]-self.img_gen.W[:,:,:,:-1]))
        loss = self.args.lambda_feat * L_feat + self.args.lambda_style * L_style + self.args.lambda_rfeat * L_feat_r + self.args.lambda_tv * L_tv

        loss.backward()
        optimizer.update()

        ## log report of losses
        chainer.report({'loss_tv': L_tv}, self.img_gen)
        chainer.report({'loss_f': L_feat}, self.img_gen)
        chainer.report({'loss_s': L_style}, self.img_gen)
        chainer.report({'loss_r': L_feat_r}, self.img_gen)

        ## clip the current image
        vgg_mean = xp.array([[[103.939]], [[116.779]], [[123.68]]])
        self.img_gen.W.array[0] = xp.clip(self.img_gen.W.array[0], -vgg_mean, 255-vgg_mean)

        # visualise
        if (self.iteration+1) % self.args.vis_freq == 0:
            med = postprocess(self.img_gen.W, self.args.gpu)
#            print("image range {} -- {}".format(np.min(med),np.max(med)))
            med = Image.fromarray(med)
            if self.args.median_filter > 0: ## median filter
                med = med.filter(ImageFilter.MedianFilter(self.args.median_filter))
#            if args.removebg:  ## paste back background
            med = Image.alpha_composite(med.convert("RGBA"),self.bkgnd).convert("RGB")
            med.save(os.path.join(self.args.out,'count{:0>4}.jpg'.format(self.iteration)))


#########################
def main():
    parser = argparse.ArgumentParser(description='A Neural Algorithm of Artistic Style')
    parser.add_argument('input',  default=None,
                        help='Original image')
    parser.add_argument('--style', '-s', default=None,
                        help='Style image')
    parser.add_argument('--reduced_image', '-r', default=None,
                        help='reduced contents image')
    parser.add_argument('--init_image', '-i', default=None,
                        help="start optimisation using this image, otherwise start with a random image")
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--iter', default=5000, type=int,
                        help='number of iterations')
    parser.add_argument('--vis_freq', '-vf', default=500, type=int,
                        help='image output interval')
    parser.add_argument('--ksize', '-k', default=4, type=int,
                        help='kernel size for reduction')
    parser.add_argument('--lr', default=4.0, type=float,
                        help='learning rate')
    parser.add_argument('--lambda_tv', '-ltv', default=1, type=float,
                        help='weight of total variation regularization')
    parser.add_argument('--lambda_feat', '-lf', default=0.1, type=float,
                        help='weight for the original shape; increase to retain it')
    parser.add_argument('--lambda_rfeat', '-lrf', default=0.1, type=float,
                        help='weight for the reduce shape; increase to retain it')
    parser.add_argument('--lambda_style', '-ls', default=1.0, type=float,
                        help='weight for the style')
    parser.add_argument('--median_filter', '-f', default=3, type=int,
                        help='kernel size of the median filter applied to the output')
    parser.add_argument('--removebg', '-nbg', action='store_true',
                        help="remove background specified by alpha in png")
    parser.add_argument('--crop_width', '-cw', type=int, default=0)
    parser.add_argument('--crop_height', '-ch', type=int, default=0)
    ## dicom related
    parser.add_argument('--image_size', default=448, type=int)
    parser.add_argument('--CT_base', '-ctb', type=int, default=-250)
    parser.add_argument('--CT_range', '-ctr', type=int, default=500)
    parser.add_argument('--CT_A_scale', type=float, default=1.0)
    parser.add_argument('--CT_B_scale', type=float, default=2.148/0.7634)

    args = parser.parse_args()
    chainer.config.autotune = True
    chainer.print_runtime_info()

    if not args.style:
        args.lambda_style = 0
    if not args.reduced_image:
        args.lambda_rfeat = 0

    print(args)

    os.makedirs(args.out, exist_ok=True)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        xp = cuda.cupy
    else:
        print("runs desperately slowly without a GPU!")
        xp = np

    ## use pretrained VGG for feature extraction
    nn = L.VGG16Layers()

    ## load images
    if os.path.splitext(args.input)[1] == '.dcm':
        image = load_dicom(args.input, args.CT_base, args.CT_range, args.image_size, args.CT_A_scale)
        image = xp.tile(image,(1,3,1,1))
        style = load_dicom(args.style, args.CT_base, args.CT_range, args.image_size, args.CT_B_scale)
        style = xp.tile(style,(1,3,1,1))
    else:
        image = xp.asarray(load_image(args.input,size=(args.crop_width,args.crop_height),removebg=args.removebg))
        if args.lambda_style > 0:
            style = xp.asarray(load_image(args.style,size=(image.shape[3],image.shape[2]), removebg=args.removebg))
        bg = Image.open(args.input).convert('RGBA')
        if args.crop_height>0 and args.crop_width>0:
            bg = bg.resize((image.shape[3],image.shape[2]),Image.LANCZOS)
        bg = np.array(bg)
        bg[:,:,3] = 255-bg[:,:,3]
        bkgnd = Image.fromarray(bg)

    ## initial image: original or random
    if args.init_image:
        img_gen = load_image(args.init_image,size=(image.shape[3],image.shape[2]), removebg=args.removebg)
    else:
        img_gen = xp.random.uniform(-20,20,image.shape).astype(np.float32)

    print("input image:", image.shape, xp.min(image), xp.max(image))

    ## image to be generated
    img_gen = L.Parameter(img_gen)

    if args.gpu>=0:
        nn.to_gpu()
        img_gen.to_gpu()

    optimizer = optimizers.Adam(alpha=args.lr)
    optimizer.setup(img_gen)

    ## compute style matrix
    layers = ["conv1_2","conv2_2","conv3_3","conv4_3"]
    with chainer.using_config('train', False):
        feature = nn(Variable(image),layers=layers)
        if args.lambda_style > 0:
            feature_s = nn(Variable(style),layers=layers)
            gram_s = { k:gram_matrix(feature_s[k]) for k in layers}
        else:
            gram_s = None
        if args.lambda_rfeat > 0:
            reduced_image = xp.asarray(load_image(args.reduced_image,size=(image.shape[3],image.shape[2]),removebg=args.removebg))
            reduced_feature = nn(F.average_pooling_2d(Variable(reduced_image),args.ksize),layers=layers)
        else:
            reduced_feature = None

    # modify loss weights according to the feature vector size
    args.lambda_rfeat /= args.ksize ** 2
    # setup updater
    dummy_iterator = chainer.iterators.SerialIterator(np.array([1]),1)
    updater = Updater(
        models=(img_gen,nn),
        iterator=dummy_iterator,
        optimizer=optimizer,
    #    converter=convert.ConcatWithAsyncTransfer(),
        device=args.gpu,
        params={'args': args, 'image': image, 'reduced_feature': reduced_feature,
            'bkgnd': bkgnd, 'feature': feature, 'gram_s': gram_s, 'layers': layers}
        )

    trainer = training.Trainer(updater, (args.iter, 'iteration'), out=args.out)

    log_interval = (100, 'iteration')
    log_keys = ['iteration','lr','main/loss_tv','main/loss_f','main/loss_s','main/loss_r']
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.LogReport(keys=log_keys, trigger=log_interval))
    trainer.extend(extensions.PrintReport(log_keys), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
                log_keys[2:], 'iteration',
                trigger=(1000, 'iteration'), file_name='loss.png'))

    trainer.run()

if __name__ == '__main__':
    main()

