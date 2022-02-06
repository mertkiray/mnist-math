import os
import random

import numpy as np
import torch
import torchvision as torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
from PIL import Image, ImageDraw

from dataset import get_data_loaders
from models.network import Net, Resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--dataset_type", type=str, default='sumnist',
                        help='dataset type: choices{sumnist|diffsumnist}')
    parser.add_argument("--epoch", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int,
                        help='experiment name')
    parser.add_argument("--lr", type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument("--wd", type=float, default=0,
                        help='weight decay')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/MNIST/',
                        help='input data directory')
    parser.add_argument("--ckpt", type=str,
                        help='load ckpt path')

    # debug
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--should_seed", action='store_true',
                        help="Should seed for reproducibility or not")
    parser.add_argument("--seed", type=int, default=3407,
                        help="Seed for reproducibility")
    parser.add_argument("--i_print", type=int, default=250,
                        help='frequency of image logging')
    parser.add_argument("--i_weight", type=int, default=10,
                        help='frequency of ckpt saving')
    parser.add_argument("--patience", type=int, default=3,
                        help='patience for early stopping')
    parser.add_argument("--model", type=str, default='resnet',
                        help='model to use: choices{conv|resnet}')

    return parser


def main():
    parser = config_parser()
    args = parser.parse_args()

    writer = SummaryWriter('runs/' + args.expname)

    config_params = {}
    for config in open(args.config):
        dict_config = config.split('=')
        if len(dict_config) == 2:
            key, value = dict_config[0].strip().replace('\n', ''), dict_config[1].strip().replace('\n', '')
            config_params[key] = value
    writer.add_text('config', str(config_params))

    if args.should_seed:
        seed_everything(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader, val_loader, test_loader = get_data_loaders(data_root=args.datadir, dataset_type=args.dataset_type,
                                                             batch_size=args.batch_size, transforms=transform)

    model = None
    if args.model == 'conv':
        model = Net(args.dataset_type).to(device)
    elif args.model == 'resnet':
        model = Resnet(args.dataset_type).to(device)

    if model is None:
        print('Model can be conv or resnet')
        exit(-1)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    start = 0
    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        print(ckpt['epoch'])
        start = ckpt['epoch'] + 1
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        model.load_state_dict(ckpt['model'])

    if not args.test:

        criterion = nn.MSELoss()
        criterion = criterion.to(device)

        best_accuracy = 0
        patience = 0
        for epoch in tqdm(range(start, args.epoch)):

            # TRAINING
            model.train()
            train_loss = 0
            train_correct = 0
            total = 0
            for i, (images, label) in enumerate(train_loader):

                images, label = images.to(device), label.type(torch.FloatTensor).to(device)

                optimizer.zero_grad()
                out = model(images)

                if (len(train_loader) * epoch + i) % args.i_print == 0:
                    ## channelwise do it TODO:
                    im = torch.stack((images[0], images[0]), dim=0)
                    writer.add_image('Images/SUMNIST', torchvision.utils.make_grid(im), len(train_loader) * epoch + i)
                    writer.add_text('Predictions', f'Label: {label[0]}, Prediction: {out[0]}',
                                    len(train_loader) * epoch + i)

                loss = criterion(out, label)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += torch.round(out).eq(label).sum().item()
                total += len(label)

            train_loss = train_loss / len(train_loader)
            train_accuracy = 100. * (train_correct / len(train_loader.dataset))
            writer.add_scalar('Train/loss', train_loss, epoch)
            writer.add_scalar('Train/accuracy', train_accuracy, epoch)
            tqdm.write(f"[TRAIN] Epoch: {epoch} Loss: {train_loss}")

            # VALIDATION
            model.eval()
            val_loss = 0
            val_correct = 0
            with torch.no_grad():
                for i, (images, label) in enumerate(val_loader):
                    images, label = images.to(device), label.type(torch.FloatTensor).to(device)

                    out = model(images)
                    val_loss += criterion(out, label).item()

                    out = torch.round(out)
                    val_correct += out.eq(label).sum().item()

            val_loss = val_loss / len(val_loader)
            val_accuracy = 100. * val_correct / len(val_loader.dataset)
            writer.add_scalar('Val/loss', val_loss, epoch)
            writer.add_scalar('Val/Accuracy', val_accuracy, epoch)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience = 0
            else:
                patience += 1

            if patience == args.patience:
                print('early stopping')
                os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
                path = os.path.join(args.basedir, args.expname, 'early_{:06d}.tar'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                exit(0)

            # SAVE MODEL AND IMPORTANT STUFF
            if epoch % args.i_weight == 0:
                os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
                path = os.path.join(args.basedir, args.expname, '{:06d}.tar'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)

        writer.close()
    else:
        # TEST
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for i, (images, label) in enumerate(test_loader):
                images, label = images.to(device), label.type(torch.FloatTensor).to(device)
                out = model(images)
                out = torch.round(out)
                test_correct += out.eq(label).sum().item()

        test_accuracy = 100. * test_correct / len(test_loader.dataset)
        print(f'Test Accuracy: {test_accuracy}')


if __name__ == '__main__':
    main()
