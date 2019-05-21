An implementation of 'A Neural Algorithm of Artistic Style' (http://arxiv.org/abs/1508.06576)
=============
This is an implementation of 'A Neural Algorithm of Artistic Style' (http://arxiv.org/abs/1508.06576).
The code is largely based on
- https://github.com/pfnet-research/chainer-gogh
- https://github.com/yusuketomoto/chainer-fast-neuralstyle

The script generates a stylised image from an input image and a style image.

## Demo on Google Colaboratory
You can try the demo on a browser
[Google Colaboratory](https://colab.research.google.com/drive/1ioGT6LE4KKU4Ttx_e0-j1REOAiu32iLo)

## Requirements
- Python 3: [Anaconda](https://www.anaconda.com/download/) is recommended
- Python libraries: chainer >= 5.3.0, cupy:  `pip install cupy chainer`
- CUDA supported GPU is highly recommended. Without one, it takes ages to do anything with CNN.

# How to use
```
python train.py -h
```
gives a brief description of command line arguments.

A typical conversion is performed by
```
python train_gogh.py input.jpg -s style.jpg
```
Generated images will be found under the directory named 'result'.

```
python train_gogh.py input.jpg -s style.jpg -i init.jpg -r input2.jpg
```
generates image blending two images input.jpg and input2.jpg with the style of style.jpg by
altering init.jpg gradually.
