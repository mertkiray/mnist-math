import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

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


def log_images(writer, images, prediction, label, step, iteration):
    operation = 'Sum'
    if images.shape[0] > 2:
        if images[2, 0, 0] == 0:
            operation = 'Diff'
        if images[2, 0, 0] == 1:
            operation = 'Sum'

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(images[0].cpu().numpy(), cmap='gray')
    axarr[1].imshow(images[1].cpu().numpy(), cmap='gray')
    axarr[0].axis('off')
    axarr[1].axis('off')
    f.tight_layout()
    f.suptitle(f'Operation:{operation}, Label:{int(label)}, Pred:{int(prediction)}')

    writer.add_figure(f'Images/{step}', f, iteration)


def save_model(epoch, model, optimizer, path):
    torch.save({
        'epoch': epoch,
        'model': model,
        'optimizer_state_dict': optimizer,
    }, path)


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

    if args.test:
        if args.ckpt:
            print(f'Testing {args.dataset_type} with ckpt:{args.ckpt}')
        else:
            print('Cannot test without a checkpoint, please indicate a checkpoint with --ckpt command.')
    else:
        print(f'Training {args.dataset_type} with model: {args.model}')

    if args.should_seed:
        print(f'Seeding with {args.seed} for reproducibility')
        seed_everything(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)

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
        best_epoch = 0
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

                loss = criterion(out, label)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = torch.round(out)
                train_correct += preds.eq(label).sum().item()
                total += len(label)

                if (len(train_loader) * epoch + i) % args.i_print == 0:
                    log_images(writer, images[0], preds[0], label[0], 'Train', len(train_loader) * epoch + i)

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

                    preds = torch.round(out)
                    val_correct += preds.eq(label).sum().item()

                    if (len(val_loader) * epoch + i) % args.i_print == 0:
                        log_images(writer, images[0], preds[0], label[0], 'Val', len(val_loader) * epoch + i)

            val_loss = val_loss / len(val_loader)
            val_accuracy = 100. * val_correct / len(val_loader.dataset)
            writer.add_scalar('Val/loss', val_loss, epoch)
            writer.add_scalar('Val/Accuracy', val_accuracy, epoch)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch
                patience = 0
            else:
                patience += 1
                if patience == args.patience:
                    path = os.path.join(args.basedir, args.expname, 'early_{:06d}.tar'.format(epoch))
                    save_model(epoch, model.state_dict(), optimizer.state_dict(), path)
                    print(f'Early stopping, best epoch is: {best_epoch}')
                    exit(0)

            # SAVE MODEL AND IMPORTANT STUFF
            if epoch % args.i_weight == 0:
                path = os.path.join(args.basedir, args.expname, '{:06d}.tar'.format(epoch))
                save_model(epoch, model.state_dict(), optimizer.state_dict(), path)

        writer.close()
    else:
        # TEST
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for i, (images, label) in enumerate(tqdm(test_loader)):
                images, label = images.to(device), label.type(torch.FloatTensor).to(device)
                out = model(images)
                preds = torch.round(out)
                test_correct += preds.eq(label).sum().item()

                if (len(test_loader) + i) % args.i_print == 0:
                    log_images(writer, images[0], preds[0], label[0], 'Val', len(val_loader) + i)

        test_accuracy = 100. * test_correct / len(test_loader.dataset)
        print(f'Test Accuracy: {test_accuracy}')


if __name__ == '__main__':
    main()
