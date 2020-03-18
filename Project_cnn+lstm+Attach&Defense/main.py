import os
import argparse
import datetime

import torch
import torchtext.data as data
import model
import train_model
import ast
from collections import OrderedDict
import multiprocessing as mu
import shutil
import numpy as np
import random
from DataLoader import mydatasets_self_two
import tqdm
### Attack ###
import FGM
from imp import reload
import sys
import matplotlib.pyplot as plt
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

### visulize attention map ###
def plot_heatmap(scores):

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(scores, cmap='viridis')

    # put the major ticks at the middle of each cell
    # and the x-ticks on top
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(scores.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(scores.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()

    plt.colorbar(heatmap)
    plt.show()
pad = "<pad>"
unk = "<unk>"

def mrs_two(path, train_name, dev_name, test_name, char_data, text_field, label_field, **kargs):
    train_data, dev_data, test_data = mydatasets_self_two.MR.splits(path, 
                                                                    train_name, 
                                                                    dev_name, 
                                                                    test_name, 
                                                                    char_data, 
                                                                    text_field, 
                                                                    label_field)
    print("len(train_data) {} ".format(len(train_data)))
    text_field.build_vocab(train_data.text, min_freq = config.min_freq)
    # text_field.build_vocab(train_data.text, dev_data.text, test_data.text, min_freq=config.min_freq)
    label_field.build_vocab(train_data.label)
    train_iter, dev_iter, test_iter = data.Iterator.splits((train_data, 
                                                            dev_data, 
                                                            test_data),
                                                            batch_sizes=(config.batch_size, len(dev_data), len(test_data)),**kargs)
    return train_iter, dev_iter, test_iter

def load_pretrained_emb_avg(path, text_field_words_dict, pad=None, set_padding=False):
    print("loading pre_train embedding by avg......")
    if not isinstance(text_field_words_dict, dict):
        text_field_words_dict = convert_list2dict(text_field_words_dict)
    assert pad is not None, "pad not allow with None"
    padID                          = text_field_words_dict[pad]
    embedding_dim                  = -1
    with open(path, encoding='utf-8') as f:
        for line in f:
            line_split             = line.strip().split(' ')
            if len(line_split) == 1:
                embedding_dim      = line_split[0]
                break
            elif len(line_split) == 2:
                embedding_dim      = line_split[1]
                break
            else:
                embedding_dim      = len(line_split) - 1
                break
    f.close()
    word_count                     = len(text_field_words_dict)
    print('The number of wordsDict is {} \nThe dim of pretrained embedding is {}\n'.format(str(word_count),
                                                                                           str(embedding_dim)))
    embeddings                     = np.zeros((int(word_count), int(embedding_dim)))

    inword_list                    = {}
    with open(path, encoding = 'utf-8') as f:
        lines                      = f.readlines()
        lines                      = tqdm.tqdm(lines)
        for line in lines:
            lines.set_description("Processing")
            values                 = line.strip().split(" ")
            if len(values) == 1 or len(values) == 2:
                continue
            index                  = text_field_words_dict.get(values[0])  # digit or None
            if index:
                vector             = np.array([float(i) for i in values[1:]], dtype='float32')
                embeddings[index]  = vector
                inword_list[index] = 1
    f.close()
    print("oov words initial by avg embedding, maybe take a while......")
    
    sum_col = np.sum(embeddings, axis=0) / len(inword_list)     # avg
    for i in range(len(text_field_words_dict)):
        if i not in inword_list and i != padID:
            embeddings[i]          = sum_col

    OOVWords                       = word_count - len(inword_list)
    oov_radio                      = np.round(OOVWords / word_count, 6)
    print("All Words = {}, InWords = {}, OOVWords = {}, OOV Radio={}".format(
        word_count, len(inword_list), OOVWords, oov_radio))

    return torch.from_numpy(embeddings).float()

def convert_list2dict(convert_list):
    list_dict                      = OrderedDict()
    for index, word in enumerate(convert_list):
        list_dict[word]            = index
    return list_dict




def load_preEmbedding():
    # load word2vec
    static_pretrain_embed          = None
    pretrain_embed                 = None
    if config.word_Embedding:
        print("word_Embedding_Path {} ".format(config.word_Embedding_Path))
        path                       = config.word_Embedding_Path
        print("loading pretrain embedding......")
        paddingkey                 = pad
        pretrain_embed             = load_pretrained_emb_avg(path = path, 
                                                             text_field_words_dict = config.text_field.vocab.itos,
                                                             pad = paddingkey)
        config.pretrained_weight   = pretrain_embed

        print("pretrain embedding load finished!")


def Load_Data():
    train_iter, dev_iter, test_iter = None, None, None
    print("Executing 2 Classification Task......")
    train_iter, dev_iter, test_iter = mrs_two(config.datafile_path, 
                                              config.name_trainfile, 
                                              config.name_devfile, 
                                              config.name_testfile, 
                                              config.char_data, 
                                              config.text_field,
                                              config.label_field, repeat=False,sort=False)

    return train_iter, dev_iter, test_iter


def define_dict():

    print("use torchtext to define word dict......")
    config.text_field               = data.Field(lower=True)
    config.label_field              = data.Field(sequential=False)
    config.static_text_field        = data.Field(lower=True)
    config.static_label_field       = data.Field(sequential=False)
    print("use torchtext to define word dict finished.")
    # return text_field

def update_arguments():
    config.lr                       = config.learning_rate
    config.init_weight_decay        = config.weight_decay
    config.init_clip_max_norm       = config.clip_max_norm
    config.embed_num                = len(config.text_field.vocab)
    config.class_num                = len(config.label_field.vocab) - 1
    # print(f'config.class_num: {config.class_num}')
    
    config.paddingId                = config.text_field.vocab.stoi[pad]
    config.unkId                    = config.text_field.vocab.stoi[unk]
    # config.kernel_sizes = [int(k) for k in config.kernel_sizes.split(',')]
    mulu                            = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config.mulu                     = mulu
    config.save_dir                 = os.path.join(""+config.save_dir, config.mulu)
    if not os.path.isdir(config.save_dir):
        os.makedirs(config.save_dir)


def load_model():
    cnn_bilstm                      = model.CNN_BiLSTM(config)
    if config.cuda is True:
        cnn_bilstm                  = cnn_bilstm.cuda()
    return cnn_bilstm



def start_train(model, train_iter, dev_iter, test_iter):
    print("\n cpu_count \n", mu.cpu_count())
    torch.set_num_threads(config.num_threads)
    if os.path.exists("./Test_Result.txt"):
        os.remove("./Test_Result.txt")
    model_training,score                  = train_model.train(train_iter, 
                                                        dev_iter, 
                                                        test_iter, 
                                                        model, 
                                                        config)
    resultlist                      = []
    if os.path.exists("./Test_Result.txt"):
        file                        = open("./Test_Result.txt")
        for line in file.readlines():
            if line[:10] == "Evaluation":
                resultlist.append(float(line[34:41]))
        result                      = sorted(resultlist)
        file.close()
        file                        = open("./Test_Result.txt", "a")
        file.write("\nThe Best Result is : " + str(result[len(result) - 1]))
        file.write("\n")
        file.close()



def main():
    """
        main function
    """
    # define word dict
    define_dict()
    # load data
    train_iter, dev_iter, test_iter = Load_Data()
    # load pretrain embedding
    load_preEmbedding()
    # update config and print
    update_arguments()
    model                           = load_model()
    # print('Checking parameters!')
    # exit()
    start_train(model, train_iter, dev_iter, test_iter)


if __name__ == "__main__":
    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))
    parser = argparse.ArgumentParser(description = "2CNN+LSTM")
    # == hyperparameters  == #
    # Debug mode #
    parser.add_argument('--debug',               type = ast.literal_eval, default = False,     
                    dest = 'debug',
                    help = 'True or False flag, input should be either True or False.')

    # Data path & reload word embedding #
    parser.add_argument('--word_Embedding',      type = ast.literal_eval, default = True,     
                    dest = 'word_Embedding',
                    help = 'True or False flag, Reload pretrained embeddings or not.')
    parser.add_argument('--word_Embedding_Path', default = "./word2vec/glove.sentiment.conj.pretrained.txt",
                    help = 'Path for word embeddings')
    parser.add_argument('--datafile_path',       type = str,              default = '',
                    help = 'datafile_path')

    parser.add_argument('--name_trainfile',      default = "./Data/raw.clean.train")
    parser.add_argument('--name_devfile',        default = "./Data/raw.clean.dev")
    parser.add_argument('--name_testfile',       default = "./Data/raw.clean.test")

    parser.add_argument('--min_freq',            type = int,              default = 1)

    parser.add_argument('--char_data',           type = ast.literal_eval, default = False,     
                    dest = 'char_data',
                    help = 'True or False flag, Reload pretrained embeddings or not.')

    parser.add_argument('--shuffle',             type = ast.literal_eval, default = True,     
                    dest = 'shuffle',
                    help = 'True or False flag, shuffle data set')

    parser.add_argument('--test',                type = ast.literal_eval, default = False,     
                    dest = 'test',
                    help = 'True or False flag, Finding the best acc in test or not.') 



    parser.add_argument('--save_dir',            default = 'Training')

    parser.add_argument('--rm_model',            type = ast.literal_eval, default = True,     
                    dest = 'rm_model',
                    help = "True or False flag, Remove model you saved but don't need." )

    parser.add_argument('--attack',              type = ast.literal_eval, default = False,     
                    dest = 'attack',
                    help = "True or False flag, Attack or not." )

    parser.add_argument('--self_att',            type = ast.literal_eval, default = False,     
                    dest = 'self_att',
                    help = "True or False flag, self_attention or not." )
    # model parameters #

    parser.add_argument('--embed_dim',           type = int,   default = 300)
    parser.add_argument('--lstm_hidden_dim',     type = int,   default = 300)
    parser.add_argument('--lstm_num_layers',     type = int,   default = 1)


    parser.add_argument('--dropout',             type = float, default = 0.75)

    parser.add_argument('--max_norm',            default = None)
    parser.add_argument('--clip_max_norm',       type = int,   default = 10)
    parser.add_argument('--kernel_num',          type = int,   default = 300)
    parser.add_argument('--kernel_sizes',        default = [3,4])

    # optimizer #
    parser.add_argument('--learning_rate',       type = float, default = 1e-3)
    parser.add_argument('--optim_momentum_value',type = float, default = 0.9)
    parser.add_argument('--weight_decay',        type = float, default = 1e-8)


    # Training #
    parser.add_argument('--num_threads',         type = int,   default = 1) ## multi-process in cpu ##

    parser.add_argument('--cuda',                type = ast.literal_eval, default = False,     
                    dest = 'cuda',
                    help = 'True or False flag, Cuda or not') ## having no cuda T_T ##


    parser.add_argument('--epochs',              type = int,   default = 15)
    parser.add_argument('--batch_size',          type = int,   default = 16)
    parser.add_argument('--log_interval',        type = int,   default = 10)
    parser.add_argument('--test_interval',       type = int,   default = 200)
    parser.add_argument('--save_interval',       type = int,   default = 200)



    # load epoch #
    parser.add_argument('--load_epoch',   type = str, default = 1,   metavar = "LE",
                        help='number of epoch to be loaded')

    parser.add_argument('-load_step',     type = str, default = 200, metavar = "LS",
                        help='number of step to be loaded')

    parser.add_argument('--resume',       type = ast.literal_eval,   default = False,     
                        dest = 'resume',
                        help = "True or False flag, resume or not" )

    parser.add_argument('--weights_path', type = str, default = "./weights/",
                        help='path to save weights')


    config = parser.parse_args()


    main()






