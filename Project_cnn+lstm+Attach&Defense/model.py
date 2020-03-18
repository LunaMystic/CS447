import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class CNN_BiLSTM(nn.Module):

    def __init__(self, args):
        super(CNN_BiLSTM, self).__init__()
        self.args                = args
        self.hidden_dim          = args.lstm_hidden_dim
        self.num_layers          = args.lstm_num_layers
        embed_num                = args.embed_num
        embed_dim                = args.embed_dim
        num_class                = args.class_num
        self.num_class           = num_class
        input_channel            = 1

        self.embed               = nn.Embedding(embed_num, embed_dim, padding_idx=args.paddingId)
        # pretrained  embedding
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        if args.attack:
            pass

        # 3 layer CNN kernal size (3,300) (4,300)
        self.convs1              = [nn.Conv2d(input_channel, 
                                              args.kernel_num, 
                                              (K_size, embed_dim), 
                                              padding=(K_size//2, 0), stride=1) for K_size in args.kernel_sizes]


        # BiLSTM
        # self.bilstm = nn.LSTM(embed_dim, self.hidden_dim, num_layers=self.num_layers, dropout=args.dropout, bidirectional=True, bias=True)


        ## ready for self-attention ## (need MyRNN class)
        self.bilstm              = nn.LSTM(embed_dim, 
                                           self.hidden_dim, 
                                           num_layers=self.num_layers, 
                                           dropout=args.dropout, 
                                           bidirectional=True, 
                                           bias=True)



        # 2 layers MLP
        L                        = len(args.kernel_sizes) * args.kernel_num + self.hidden_dim * 2
        self.hidden2label1       = nn.Linear(L, L // 2)
        self.hidden2label2       = nn.Linear(L // 2, num_class)

        # dropout
        self.dropout             = nn.Dropout(args.dropout)
        self.q_key               = Self_Attention()


    def forward(self, x):
        embed                    = self.embed(x)

        # CNN
        cnn_x                    = embed
        cnn_x                    = torch.transpose(cnn_x, 0, 1)
        cnn_x                    = cnn_x.unsqueeze(1)
        cnn_x                    = [conv(cnn_x).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        cnn_x                    = [torch.tanh(F.max_pool1d(i, i.size(2)).squeeze(2)) for i in cnn_x]  # [(N,Co), ...]*len(Ks)
        cnn_x                    = torch.cat(cnn_x, 1)
        cnn_x                    = self.dropout(cnn_x)

        # BiLSTM
        bilstm_x                 = embed.view(len(x), embed.size(1), -1)
        # bilstm_out,_  = self.bilstm(bilstm_x)
        # prepare to add self-attention here
        bilstm_out, hidden_state = self.bilstm(bilstm_x)
        # bilstm_out shape 32,16,600 #
        # print(bilstm_out.shape)
        if self.args.self_att:
            bilstm_out,attention = self.q_key(bilstm_out)






        # after adding self - attention
        bilstm_out               = torch.transpose(bilstm_out, 0, 1)
        bilstm_out               = torch.transpose(bilstm_out, 1, 2)
        bilstm_out               = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        bilstm_out               = torch.tanh(bilstm_out)

        # CNN and BiLSTM CAT (missing self-attention cat~)
        cnn_x                    = torch.transpose(cnn_x, 0, 1)
        bilstm_out               = torch.transpose(bilstm_out, 0, 1)
        cnn_bilstm_out           = torch.cat((cnn_x, bilstm_out), 0)
        cnn_bilstm_out           = torch.transpose(cnn_bilstm_out, 0, 1)

        # linear
        cnn_bilstm_out           = self.hidden2label1(torch.tanh(cnn_bilstm_out))
        cnn_bilstm_out           = self.hidden2label2(torch.tanh(cnn_bilstm_out))

        # output
        logit                    = cnn_bilstm_out
        if self.args.self_att:
          return logit,attention
        else:
          return logit


class Self_Attention(nn.Module):
   def __init__(self, dim_out=600, dim_in=600):
       super(Self_Attention, self).__init__()
       self.dim_out              = dim_out
       # full_value: 64, 392, 100    32,16,600
       self.value_fc             = nn.Linear(dim_in, dim_out)
       self.key_fc               = nn.Linear(dim_in, dim_out)
       self.query_fc             = nn.Linear(dim_in, dim_out)

   def forward(self, x):
       # x = x.transpose(0,1)
       # print(f'x.shape{x.shape}')
       value                     = F.leaky_relu(self.value_fc(x),    negative_slope=0.25, inplace=True)
       key                       = F.leaky_relu(self.key_fc(x),      negative_slope=0.25, inplace=True)
       query                     = F.leaky_relu(self.query_fc(x),    negative_slope=0.25, inplace=True)
       # print(f'value_fc: {value.shape}')
       # print(f'query_fc: {query.shape}')
       # print(f'key_fc: {key.shape}')
       # exit()
       # 64, 392, 100
       similarity_score          = F.softmax(torch.bmm(query, key.transpose(1,2)), dim=-1)
       output                    = torch.bmm(similarity_score, value)
       output                    = output / np.sqrt(self.dim_out)
       # output = output.transpose(0,1)
       return output,similarity_score

