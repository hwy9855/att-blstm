import torch
import numpy as np
from torch import nn

class AttBLSTM(nn.Module):
    def __init__(self, input_size, label_size, batch_size, num_layer=1):
        super(AttBLSTM, self).__init__()
        self.blstm = torch.nn.LSTM(input_size, input_size, num_layer, bidirectional=True, batch_first=True)
        self.h0 = torch.randn(2 * num_layer, batch_size, input_size).cuda()
        self.c0 = torch.randn(2 * num_layer, batch_size, input_size).cuda()
        self.softmax = nn.Softmax(dim=2)
        self.tanh = nn.Tanh()
        self.batch_size = batch_size
        self.hidden_size = input_size
        self.loss = nn.BCELoss()
        self.w = torch.randn(input_size).cuda()

        self.embedding_dropout = nn.Dropout(0.3)
        self.lstm_dropout = nn.Dropout(0.3)
        self.attention_dropout = nn.Dropout(0.5)

        self.fc = nn.Sequential(nn.Linear(input_size, label_size))

    def Att_layer(self, H):
        M = self.tanh(H)
        alpha = self.softmax(torch.bmm(M, self.w.repeat(self.batch_size, 1, 1).transpose(1, 2)))
        res = self.tanh(torch.bmm(alpha.transpose(1,2), H))
        return res

    def forward(self, x_input):
        x_input = self.embedding_dropout(x_input)
        h, _ = self.blstm(x_input, (self.h0, self.c0))
        h = h[:,:,self.hidden_size:] + h[:,:,:self.hidden_size]
        h = self.lstm_dropout(h)
        atth = self.Att_layer(h)
        atth = self.attention_dropout(atth)
        out = self.fc(atth)
        out = self.softmax(out)

        return out.view(self.batch_size, -1)

if __name__ == '__main__':
    x = torch.randn(128, 5, 50).cuda()
    model = AttBLSTM(50, 19, 128, 200).cuda()
    print(model.forward(x))