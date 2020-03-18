import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim.lr_scheduler as lr_scheduler
import shutil
import random
import numpy as np
from tqdm import tqdm
import FGM
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torch.autograd import Variable
def plot_heatmap(scores,epoch):
    ### scores shape (30,8,600) ###
    scores = scores.detach().numpy()[0]
    fig, ax = plt.subplots()
    # print(scores.shape)
    # exit()
    heatmap = ax.pcolor(scores,cmap=cm.gray)

    # put the major ticks at the middle of each cell
    # and the x-ticks on top
    ax.xaxis.tick_top()
    
    ax.set_xticks(np.arange(scores.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(scores.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()
    filename = 'Attention at '+ str(epoch) + '.png'
    plt.colorbar(heatmap)
    plt.savefig(filename)
    print('### Attention map is saved ! ###')
    # plt.show()

def train(train_iter, dev_iter, test_iter, model, args):
    start_epoch                       = 1
    if args.cuda:
        model.cuda()
    print("Adam Training......")
    optimizer                         = torch.optim.Adam(model.parameters(), 
                                                         lr=args.lr, 
                                                         weight_decay=args.init_weight_decay)

    steps                             = 0
    model_count                       = 0
    best_accuracy                     = Best_Result()
    model.train()
    # using one epoch to debug #
    if args.debug:
        args.epochs                   = 1

    # Still having some bugs here # (Just comment)
    if args.resume:
        model, optimizer, start_epoch = loading_model(args.load_epoch, args.load_step, model, optimizer, "./weights/")
        # print('Loading is done !')
        # exit()
    # Training part #
    # Adding Attack #
    print('Ready To initialize FGM')
    fgm                               = FGM.FGM(model)
    print('FGM is Ready !')
    for epoch in range(start_epoch, args.epochs+1):
        print('Current epochs:',epoch)
        steps                         = 0
        print("\n## The {} Epoch, All {} Epochs ! ##".format(epoch, args.epochs))
        for batch in train_iter:
            feature, target           = batch.text, batch.label.data.sub_(1)
            if args.cuda is True:
                feature, target       = feature.cuda(), target.cuda()

            target                    = autograd.Variable(target)  # question 1
            optimizer.zero_grad()
            # print(feature.shape)
            # exit()
            # with torch.no_grad():
                # logit,scores              = model(feature)
            if args.self_att:
                logit,scores              = model(feature)
            else:
                logit                     = model(feature)
            loss                      = F.cross_entropy(logit, target)
            # loss = autograd.Variable(loss,requires_grad=True)
            loss.backward()
            if args.attack:
                fgm.attack()
                logit_adv             = model(feature)
                loss_adv              = F.cross_entropy(logit_adv,target)
                loss_adv.backward()
                fgm.restore()

            if args.init_clip_max_norm is not None:
                utils.clip_grad_norm_(model.parameters(), max_norm=args.init_clip_max_norm)
            optimizer.step()

            steps                    += 1
            if steps % args.log_interval == 0:
                train_size            = len(train_iter.dataset)
                corrects              = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy              = float(corrects)/batch.batch_size * 100.0
                sys.stdout.write(
                    '\rBatch[{}/{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                            train_size,
                                                                             loss.item(),
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                print("\nDev  Accuracy: ", end="")
                eval(dev_iter, model, args, best_accuracy, epoch, test=False)
                print("Test Accuracy: ", end="")
                eval(test_iter, model, args, best_accuracy, epoch, test=True)
            if steps % args.save_interval == 0:
                if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                save_prefix          = os.path.join(args.save_dir, 'Training')
                save_path            = '{}_steps{}.pt'.format(save_prefix, steps)
                if os.path.exists(args.weights_path):
                    saving_path      = args.weights_path + 'steps{}-'.format(steps)  # ./weights/steps200- #
                else:
                    os.mkdir(args.weights_path)
                    saving_path      = args.weights_path + 'steps{}-'.format(steps)
                # print(saving_path)
                # exit()
                save_model(epoch, model, optimizer,saving_path)

                torch.save(model.state_dict(), save_path)
                if os.path.isfile(save_path) and args.rm_model is True:
                    os.remove(save_path)
                model_count         += 1
        if args.self_att:
            plot_heatmap(scores,epoch)
    return model_count


def eval(data_iter, model, args, best_accuracy, epoch, test=False):
    model.eval()
    corrects, avg_loss               = 0, 0
    for batch in data_iter:
        feature, target              = batch.text, batch.label
        target.data.sub_(1)
        if args.cuda:
            feature, target          = feature.cuda(), target.cuda()
        if args.self_att:
            logit,_                        = model(feature)
        else:
            logit                          = model(feature)
        loss                         = F.cross_entropy(logit, target)

        avg_loss                    += loss.item()
        corrects                    += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    size                             = len(data_iter.dataset)
    avg_loss                         = loss.item()/size
    accuracy                         = float(corrects)/size * 100.0
    model.train()
    print(' Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) '.format(avg_loss, accuracy, corrects, size))
    if test is False:
        if accuracy >= best_accuracy.best_dev_accuracy:
            best_accuracy.best_dev_accuracy = accuracy
            best_accuracy.best_epoch        = epoch
            best_accuracy.best_test         = True
    if test is True and best_accuracy.best_test is True:
        best_accuracy.accuracy              = accuracy

    if test is True:
        print("The Current Best Dev Accuracy: {:.4f}, and Test Accuracy is :{:.4f}, locate on {} epoch.\n".format(
            best_accuracy.best_dev_accuracy, best_accuracy.accuracy, best_accuracy.best_epoch))
    if test is True:
        best_accuracy.best_test             = False


def save_model(epoch, model, optimizer,save_path):
    filename                                = save_path + str(epoch) + '.pt'
    state                                   = {
                                                'epoch': epoch,
                                                'state_dict': model.state_dict(),
                                                'optimizer': optimizer.state_dict(),
                                               }
    torch.save(state, filename)

def loading_model(epoch,step, model, optimizer, save_path):
    filename                                = save_path + 'steps{}-'.format(step) + str(epoch) + '.pt'
    if os.path.isfile(filename):
        print("######### loading weights ##########")
        checkpoint                          = torch.load(filename)
        start_epoch                         = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('########## loading weights done ##########')
        return model, optimizer, start_epoch
    else:
        print("no such file: ", filename)


class Best_Result:
    def __init__(self):
        self.best_dev_accuracy              = -1
        self.best_accuracy                  = -1
        self.best_epoch                     = 1
        self.best_test                      = False
        self.accuracy                       = -1






