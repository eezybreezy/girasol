# girasol

## Running reconstructions using `girasol`
`gs.py` runs the reconstruction experiments presented in the paper.

`python gs.py --img <image loc> --device <device name> --num-chunks <reconstr chunks> --num-tuples <sample tuples> --model <model name> --num-classes <classes> --lr <initial lr> --psnr-thresh <lower lr at threshold> --iterations <ctr value> --img-frq <img dump freq> --rolling --tag <comments> --max-tuples <max tuples to leak> --history <history size>`

For running CIFAR-10 reconstructions on ResNet-18
`python gs.py --img ../data/cifar/cifar_0.jpg --device 'cuda:0' --num-chunks 1 --num-tuples 30 --model ResNet18 --num-classes 10 --lr 1.0 --psnr-thresh 16.5 --iterations 1 --img-frq 20 --rolling --tag 'cifar-expt' --max-tuples 30 --history 20 --cifar`


For running ImageNet reconstructions on ResNet-18
`python gs.py --img ../data/dog.png --device 'cuda:0' --num-chunks 1 --num-tuples 20 --model torchResNet18 --num-classes 512 --lr 1.0 --psnr-thresh 16.5 --iterations 1 --img-frq 20 --rolling --tag 'resnet-reconstr' --max-tuples 200 --history 35 --manipulate`


## Code poisoned layer implementation
The notebook `code-poisoned-layer.ipynb` contains the implementation of the malicious layer discussed in PyTorch.