An implementation of 'A Neural Algorithm of Artistic Style' (http://arxiv.org/abs/1508.06576)
=============
This is an implementation of 'A Neural Algorithm of Artistic Style' (http://arxiv.org/abs/1508.06576).
The code is largely based on https://github.com/pfnet-research/chainer-gogh and https://github.com/yusuketomoto/chainer-fast-neuralstyle

The script generates a stylised image from an input image and a style image.

## Requirements
- Python 3: [Anaconda](https://www.anaconda.com/download/) is recommended
- Python libraries: Chainer, cupy:  `pip install cupy chainer`
- CUDA supported GPU is highly recommended. Without one, it takes ages to do anything with CNN.

# How to use
```
python train.py -h
```
gives a brief description of command line arguments.

A typical conversion is performed by
```
python train_gogh.py input.jpg -s style.jpg -rs
```
Generated images will be found under the directory named 'result'.
