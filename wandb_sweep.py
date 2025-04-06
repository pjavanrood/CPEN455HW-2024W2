import time
import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import wandb
from tqdm import tqdm
from pprint import pprint
import argparse
from pytorch_fid.fid_score import calculate_fid_given_paths
import logging
import pandas as pd
import numpy as np
import utils, dataset, generation_evaluation
import model as model_py

import dotenv
dotenv.load_dotenv()

from dataclasses import dataclass

@dataclass
class Config:
    en_wandb: bool = False
    tag: str = "default"
    sampling_interval: int = 5
    data_dir: str = "data"
    save_dir: str = "models"
    sample_dir: str = "samples"
    dataset: str = "cpen455"
    save_interval: int = 10
    load_params: str = None
    obs: tuple = (3, 32, 32)
    nr_resnet: int = 1
    nr_filters: int = 40
    nr_logistic_mix: int = 5
    lr: float = 0.0002
    lr_decay: float = 0.999995
    batch_size: int = 64
    sample_batch_size: int = 32
    max_epochs: int = 5000
    seed: int = 1


def setup(args: Config):
  pprint(args.__dict__)
  utils.check_dir_and_create(args.save_dir)

  # reproducibility
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  model_name = 'pcnn_' + args.dataset + "_"
  model_path = args.save_dir + '/'
  if args.load_params is not None:
      model_name = model_name + 'load_model'
      model_path = model_path + model_name + '/'
  else:
      model_name = model_name + 'from_scratch'
      model_path = model_path + model_name + '/'

  job_name = "PCNN_Training_" + "dataset:" + args.dataset + "_" + args.tag

  #Reminder: if you have patience to read code line by line, you should notice this comment. here is the reason why we set num_workers to 0:
  #In order to avoid pickling errors with the dataset on different machines, we set num_workers to 0.
  #If you are using ubuntu/linux/colab, and find that loading data is too slow, you can set num_workers to 1 or even bigger.
  kwargs = {'num_workers':2, 'pin_memory':True, 'drop_last':True}

  # set data
  if "mnist" in args.dataset:
      ds_transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), dataset.rescaling, dataset.replicate_color_channel])
      train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True,
                          train=True, transform=ds_transforms), batch_size=args.batch_size,
                              shuffle=True, **kwargs)

      test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False,
                      transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

  elif "cifar" in args.dataset:
      ds_transforms = transforms.Compose([transforms.ToTensor(), dataset.rescaling])
      if args.dataset == "cifar10":
          train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True,
              download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

          test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False,
                      transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
      elif args.dataset == "cifar100":
          train_loader = torch.utils.data.DataLoader(datasets.CIFAR100(args.data_dir, train=True,
              download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

          test_loader  = torch.utils.data.DataLoader(datasets.CIFAR100(args.data_dir, train=False,
                      transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
      else:
          raise Exception('{} dataset not in {cifar10, cifar100}'.format(args.dataset))

  elif "cpen455" in args.dataset:
      ds_transforms = transforms.Compose([transforms.Resize((32, 32)), dataset.rescaling])
      train_loader = torch.utils.data.DataLoader(dataset.CPEN455Dataset(root_dir=args.data_dir,
                                                                mode = 'train',
                                                                transform=ds_transforms),
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  **kwargs)
      test_loader  = torch.utils.data.DataLoader(dataset.CPEN455Dataset(root_dir=args.data_dir,
                                                                mode = 'test',
                                                                transform=ds_transforms),
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  **kwargs)
      val_loader  = torch.utils.data.DataLoader(dataset.CPEN455Dataset(root_dir=args.data_dir,
                                                                mode = 'validation',
                                                                transform=ds_transforms),
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  **kwargs)
  else:
      raise Exception('{} dataset not in {mnist, cifar, cpen455}'.format(args.dataset))
  
  return train_loader, test_loader, val_loader


def train_or_test(model, data_loader, optimizer, loss_op, device, args, epoch, mode = 'training'):
    logging.debug('mode: {}'.format(mode))
    if mode == 'training':
        model.train()
    else:
        model.eval()

    deno =  args.batch_size * np.prod(args.obs) * np.log(2.)
    loss_tracker = utils.mean_tracker()

    for batch_idx, item in enumerate(tqdm(data_loader)):
        logging.debug('batch_idx: {}'.format(batch_idx))
        model_input, label = item
        model_input = model_input.to(device)
        model_output = model(model_input, label)

        loss = loss_op(model_input, model_output)
        loss_tracker.update(loss.item()/deno)
        if mode == 'training':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if args.en_wandb:
        wandb.log({mode + "-Average-BPD" : loss_tracker.get_mean(), 'epoch': epoch})
        wandb.log({mode + "-epoch": epoch})


# 1. Define the training function
def train(args: Config, train_loader, test_loader, val_loader):
    input_channels = args.obs[0]
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_op   = lambda real, fake : utils.discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : utils.sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)

    model = model_py.PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
                input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
    model = model.to(device)

    if args.load_params:
        model.load_state_dict(torch.load(args.load_params))
        print('model parameters loaded')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    for epoch in tqdm(range(args.max_epochs)):
        train_or_test(model = model,
                      data_loader = train_loader,
                      optimizer = optimizer,
                      loss_op = loss_op,
                      device = device,
                      args = args,
                      epoch = epoch,
                      mode = 'training')

        # decrease learning rate
        scheduler.step()
        train_or_test(model = model,
                      data_loader = test_loader,
                      optimizer = optimizer,
                      loss_op = loss_op,
                      device = device,
                      args = args,
                      epoch = epoch,
                      mode = 'test')

        train_or_test(model = model,
                      data_loader = val_loader,
                      optimizer = optimizer,
                      loss_op = loss_op,
                      device = device,
                      args = args,
                      epoch = epoch,
                      mode = 'val')

        if args.sampling_interval != -1 and epoch % args.sampling_interval == 0:
            print('......sampling......')
            sampled_images = generation_evaluation.my_sample(model, args.sample_dir, args.sample_batch_size, args.obs, sample_op)
            sample_result = {label: wandb.Image(img, caption="epoch {}".format(epoch)) for label, img in sampled_images.items()}

            gen_data_dir = args.sample_dir
            ref_data_dir = args.data_dir +'/test'
            paths = [gen_data_dir, ref_data_dir]
            try:
                fid_score = calculate_fid_given_paths(paths, 32, device, dims=192)
                print("Dimension {:d} works! fid score: {}".format(192, fid_score))
            except:
                print("Dimension {:d} fails!".format(192))

            if args.en_wandb:
              for img in sample_result:
                wandb.log({"samples": img,
                            "FID": fid_score})


def wandb_sweep_train():
    # Initialize wandb run
    run = wandb.init()
    wconfig = run.config

    args = Config(
        nr_resnet=wconfig.nr_resnet,
        nr_filters=wconfig.nr_filters,
        nr_logistic_mix=wconfig.nr_logistic_mix,
        batch_size=32,
        sample_batch_size=16,
        sampling_interval=-1,
        max_epochs=20,
        en_wandb=True,
    )

    train_loader, test_loader, val_loader = setup(args)

    train(args, train_loader, test_loader, val_loader)


# 2. Define sweep configuration
sweep_config = {
    'method': 'random',  # Can be 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'test-Average-BPD',
        'goal': 'minimize'
    },
    'parameters': {
        'nr_resnet': {
            'values': [1, 5, 10, 15]
        },
        'nr_filters': {
            'values': [40, 80, 160]
        },
        'nr_logistic_mix': {
            'values': [5, 10, 15]
        }
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3
    }
}

# 3. Create and run the sweep
def main():
    wandb.login()
    # Initialize sweep
    # sweep_id = wandb.sweep(sweep_config, project="CPEN455HW")
    # Run the sweep
    wandb.agent("xf7hbyy9", function=wandb_sweep_train, count=10, project="CPEN455HW")  # count=10 means 10 runs

if __name__ == "__main__":
    main()
