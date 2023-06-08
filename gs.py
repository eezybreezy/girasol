import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import math
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import trange
import collections
import argparse
from datetime import datetime
import os
import pickle
import pdb
from skimage.metrics import structural_similarity as ssim
from statistics import mean


from models import ResNet18, AlexNet, ResNet152, ResNet50, LeNet, FCN
from models_new import torch_ResNet18, torch_ResNet50, torch_ResNet101

# torch.set_default_tensor_type(torch.floatTensor)

parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default=None)
parser.add_argument('--device', type=str)
parser.add_argument('--num-chunks', type=int, default=1)
parser.add_argument('--num-tuples', type=int, default=50)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--num-classes', type=int, default=None)
parser.add_argument('--cifar', action='store_true')
parser.add_argument('--chest', action='store_true')
parser.add_argument('--lr', type=float, default=1.0)
parser.add_argument('--psnr-thresh', type=float, default=None)
parser.add_argument('--rolling', default=False, action='store_true')
parser.add_argument('--iterations', type=int, default=1)
parser.add_argument('--img-frq', type=int, default=1)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--save', default=None, action='store_true')
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--param-loc', type=str, default=None)
parser.add_argument('--max-tuples', type=int, default=300)
parser.add_argument('--history', type=int, default=20)
parser.add_argument('--normalize', default=False, action='store_true')
parser.add_argument('--manipulate', default=False, action='store_true')


args = parser.parse_args()


def metrics(img_r, img_t):
    if args.chest:
      ssim_n = ssim(img_r.cpu().numpy(), img_t.cpu().numpy(), data_range=int(img_r.max()) - int(img_r.min()))
    else:
      ssim_n = ssim(img_r.cpu().numpy().transpose(1,2,0), img_t.cpu().numpy().transpose(1,2,0), data_range=int(img_r.max()) - int(img_r.min()), channel_axis=2, multichannel=True)
    loss = nn.MSELoss()
    mse = loss(img_r, img_t).cpu()
    psnr = 10 * math.log10(1./mse.item())
    return mse.item(), psnr, ssim_n

def load_image_cifar(path:str, index:int = 0):
    with open(path, 'rb') as fo:
        img_d = pickle.load(fo, encoding='bytes')
    return img_d[b'data'][index].reshape(1, 3,32,32)

def load_image_cifar_2(path:str):
    img = Image.open(path)
    img = np.array(img)
    img = img.transpose(2,0,1).reshape(1,3,32,32)
    print(f'loaded image w/ shape {img.shape}')
    return img

def load_image_chest(path:str):
    img = Image.open(path).convert('L')
    img = np.array(img)
    img = img.reshape(1,1,28,28)
    print(f'loaded image w/ shape {img.shape}')
    return img

def load_image_imgnet(path:str, size:int=224):
    img = Image.open(path)
    # p = transforms.Compose([transforms.Resize((size, size), transforms.InterpolationMode.BICUBIC)])
    p = transforms.Compose([transforms.Resize((size, size), Image.BICUBIC)])
    img = np.array(p(img))
    img = img.transpose(2,0,1).reshape(1,3,224,224)
    print(f'loaded image w/ shape {img.shape}')
    if args.normalize:
      return (img/255. - 0.4)
    else:
      return img

def get_model(model, device, model_args):
  m = model(**model_args).to(device)
  # train the model to have a fixed point
  loss = nn.MSELoss()
  optim = torch.optim.Adam(m.parameters(), lr=0.000001)
  with trange(20) as t:
    for i in t:
      X = torch.randint(0, 255, (1,3,img.shape[-2], img.shape[-1]), dtype=torch.float).to(device)
      for ctr in range(5):
        optim.zero_grad()
        ov = m(X)
        lv = torch.zeros(ov.shape).to(device).float()
        l = -loss(ov, lv)
        l.backward()
        optim.step()
        t.set_description('ctr: %i LOSS: %f' % (i, l.item()))

      X = X.cpu()
      del X

  del optim, loss
  return m

def collect_tuples(N:int=100, model=None, model_args:dict=None, \
        X:torch.tensor=None, device:torch.device=None, logfd=None):
    if logfd:
      fd.write(f"tuples: {N}\nmodel: {model}\nmodel_args: {model_args}\n")
    L = []
    for i in trange(N):

        ## get a model w/ that has been trained to have a fixed point
        # m = get_model(model, device, model_args)
        m = model(**model_args).to(device)
        params = {k : v.clone().detach().cpu() for k,v in m.state_dict().items()}
        # if len(L) > 0:
        #   for k in list(params.keys())[:-2]:
        #     params[k] = L[0][0][k].clone().detach()
        #   m.load_state_dict(params)
        with torch.no_grad():
            s = m(X.float()).to(device)
            L.append((params, s.clone().detach().cpu()))
    return L

def solve_chunk(img, device, num_chunks:int=1, num_tuples:int=50, model=None, 
  model_args:dict=None, lr:int=1.0, psnr_thresh:float=None, iter_per_chunk:int=1, img_frq:int=1, saveloc:str=None, logfd=None, lossfd=None):
  
  if saveloc is None or logfd is None or lossfd is None:
    raise Exception

  if args.checkpoint:
    X = [torch.tensor(load_image_imgnet(checkpoint, 224)).float()]
  else:  
    if args.normalize:
      X = [torch.rand((1,3,img.shape[-2], img.shape[-1]//num_chunks), dtype=torch.float) - 0.4 for _ in range(num_chunks)]
    else:
      X = [torch.randint(0, 255, (1,3,img.shape[-2], img.shape[-1]//num_chunks), dtype=torch.float) for _ in range(num_chunks)]
    
    logfd.write(f'init X w/ shape: ({len(X)},{X[0].shape})\n')
  
  X = [x.to(device) for x in X]
  
  
  if args.rolling is None:
    L = collect_tuples(num_tuples, model, model_args, img, device)

  # setup optimizer and loss
  loss = nn.MSELoss()
  
  ctr = 0
  psnr = 2.
  # psnr_o = 1.
  psnr_o = collections.deque([4.0]*args.history,args.history)
  # psnr_o = psnr_h[0]
  
  logfd.write("time, ctr, chunk_idx, mse, psnr, ssim, grad_mean\n")
  lossfd.write(f"time, chunk_idx, loss\n")
  psnr = .1
  init_time = datetime.now()

  if args.checkpoint and args.param_loc:
    with open(args.param_loc, 'rb') as f:
      L = pickle.load(f)
      # L = np.load(f, allow_pickle=True)
  else:
    L = []

  model_args['manipulate'] = args.manipulate

  while True:
    
    if psnr_thresh and psnr > psnr_thresh and lr > 0.1:
      lr = 0.1
      logfd.write(f"change lr: {lr}\n")
      print((f"change lr: {lr}\n"))
    
    if args.rolling and int(100*mean(psnr_o)) >= int(100*psnr):
      print(f'psnr_o: {psnr_o[0]}\tpsnr:{psnr}')
      #  del L
      if len(L) < args.max_tuples:
        logfd.write(f"sampling: {num_tuples}\n")
        # del L
        print(model_args)
        L += collect_tuples(num_tuples, model, model_args, img, device)

        for _ in range(args.history):
          psnr_o.append(4.0)

        if args.save:
          with open(f'{saveloc}/mapping.pkl', 'wb') as f:
            pickle.dump(L, f)
        #   np.save(f, L)
        # # perturb X
      if not args.normalize:
        for x in X:
          x += torch.empty(x.shape).uniform_(-1.,1.).to(device)
      # psnr_o = 4.
    
    # else:
    #   if ctr % 20 == 0:
    #     psnr_o = psnr

    for chunk_idx, x in enumerate(X):
      x.requires_grad_(True)
      # optim = torch.optim.LBFGS([x], lr = lr, history_size=20, max_iter=30, line_search_fn=None)
      optim = torch.optim.Adam([x], lr = lr)
      
      for _ in range(iter_per_chunk):
        for idx, (params, lv) in enumerate(L):
          loss_v = 1.0
          
          def closure():
            nonlocal loss_v
            optim.zero_grad()
            # m = ResNet18(**{'num_classes': 1024, 'cifar': True}).to(device)
            m = model(**model_args).to(device)
            m.load_state_dict(params)
            ov = m(torch.cat(X, 3))
            l = loss(ov, lv.to(device))
            loss_v = l.item()
            l.backward()
            return l
          optim.step(closure)
          lv.cpu()
          # del lv
          print(f'chunk_idx:{chunk_idx}, loss: {loss_v}', end='\r')
          lossfd.write(f'{datetime.now() - init_time},{chunk_idx},{loss_v}\n')
      
      x.requires_grad_(False)
      del optim

      with torch.no_grad():
        now_time = datetime.now()

        if args.normalize:
          mse_n, psnr_n, ssim_n = metrics((torch.cat(X, 3).squeeze()+0.4).clip(0,1), (img.squeeze() + 0.4).clip(0,1))
        else:
          mse_n, psnr_n, ssim_n = metrics(torch.cat(X, 3).squeeze().clip(0,255)/255.0, img.squeeze().clip(0,255)/255.0)

        print(f"time:{now_time - init_time}\tchunk_idx:{chunk_idx}\tmse: {mse_n}\tpsnr:{psnr_n}\tssim:{ssim_n}\tmean_grad:{torch.mean(x.grad)}\tmean_X:{torch.mean(x)}")
        logfd.write(f"{now_time - init_time},{ctr},{chunk_idx},{mse_n},{psnr_n},{ssim_n},{torch.mean(x.grad)}\n")

        psnr = psnr_n
        psnr_o.append(psnr)

    if ctr % img_frq == 0:
      if not args.chest:
        if args.normalize:
          plt.imsave(f'{saveloc}/dump/{ctr}_{mse_n}_{psnr_n}.jpg', ((torch.cat(X,3)[0].clone().detach().cpu() + 0.4).clip(0,1)).numpy().transpose(1,2,0))
        else:
          plt.imsave(f'{saveloc}/dump/{ctr}_{mse_n}_{psnr_n}.jpg', (torch.cat(X,3)[0].clone().detach().cpu().clip(0,255)/255.0).numpy().transpose(1,2,0))
      else:
        # pdb.set_trace()
        plt.imsave(f'{saveloc}/dump/{ctr}_{mse_n}_{psnr_n}.jpg', (torch.cat(X,3)[0].clone().detach().cpu().clip(0,255)/255.0).numpy()[0], cmap='gray')
    # clip
    if not args.normalize:
      for i in range(len(X)):
        X[i] = X[i].clip(0,255)
    ctr += 1
      
  
if __name__ == '__main__':
  # img = load_image_cifar('../v1/datasets/cifar-10-batches-py/data_batch_1', 12)
    
  test_dir_name = str(datetime.now())
  print(test_dir_name)
  if not os.path.exists(test_dir_name):
    os.makedirs(test_dir_name + "/dump")

  logfd = open(f'{test_dir_name}/metrics.log', 'w', buffering=1)
  lossfd = open(f'{test_dir_name}/loss.log', 'w', buffering=1)

  if args.chest:
    img = load_image_chest(args.img)
  elif args.cifar:
    img = load_image_cifar_2(args.img)
  else:
    img = load_image_imgnet(args.img)
  print(f'img: {args.img}')
  logfd.write(f"img: {args.img}\n")
  # plt.imshow(img[0].transpose(1,2,0))
  # plt.show()
  device = torch.device(args.device)
  print(f'device: {device}')
  logfd.write(f'device: {device}\n')
  if args.tag:
    logfd.write(f'tag: {args.tag}\n')

  # logfd.write(f"num-chunks: {args.num_chunks}\nnum-tuples: {args.num_tuples}\nmodel:{args.model}\nnum-classes: {args.num_classes}\nlr: {args.lr}\npsnr-thresh: {args.psnr_thresh}\niter_per_chunk: {args.iterations}\nimg-frq: {args.img_frq}\n")
  logfd.write(f"{str(args)}\n")

  img = torch.tensor(img).float().to(device)
  # solve_chunk(img, device, 8, 20, ResNet18, {'num_classes': 1024, 'cifar': True}, 5, test_dir_name, logfd, lossfd)
  model_name = {'AlexNet': AlexNet, 'ResNet18': ResNet18, 'ResNet152': ResNet152, 'ResNet50': ResNet50, 'LeNet':LeNet, 'FCN':FCN , 'torchResNet18': torch_ResNet18, 'torchResNet50': torch_ResNet50, 'torchResNet101': torch_ResNet101}
  solve_chunk(img, device, args.num_chunks, args.num_tuples, model_name[args.model], {'num_classes': args.num_classes, 'cifar': args.cifar,}, args.lr, args.psnr_thresh, args.iterations, args.img_frq, test_dir_name, logfd, lossfd)
  