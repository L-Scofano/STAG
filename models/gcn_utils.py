import torch
import torch.nn as nn
import math


class AttModule(nn.Module):
    def __init__(self,
                 time_dim,
                 joints_dim
    ):
        super(AttModule,self).__init__()
        
        self.A=nn.Parameter(torch.FloatTensor(time_dim, joints_dim,joints_dim)) #learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv,stdv)

        self.T=nn.Parameter(torch.FloatTensor(joints_dim , time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv,stdv)

    def forward(self, x):
        x = torch.einsum('nctv,vtq->ncqv', (x, self.T))
        x = torch.einsum('nctv,tvw->nctw', (x, self.A))
        return x.contiguous() 


class STSGCN_layer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 time_dim,
                 joints_dim,
                 dropout=0.,
                 bias=True):
        
        super(STSGCN_layer,self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2,(self.kernel_size[1] - 1) // 2)
        self.att=AttModule(time_dim,joints_dim) 
        
        self.gcn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (self.kernel_size[0], self.kernel_size[1]),
                (stride, stride),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if stride != 1 or in_channels != out_channels: 

            self.residual=nn.Sequential(nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, 1)),
                nn.BatchNorm2d(out_channels),
            )

        else:
            self.residual=nn.Identity()

        self.prelu = nn.PReLU()

    def forward(self, x):
        res=self.residual(x)
        x=self.att(x) 
        x=self.gcn(x)
        x=x+res
        x=self.prelu(x)
        return x



class CNN_layer(nn.Module): 
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout,
                 bias=True):
        
        super(CNN_layer,self).__init__()
        self.kernel_size = kernel_size
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2) # padding so that both dimensions are maintained
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

        self.block= [nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding)
                     ,nn.BatchNorm2d(out_channels),nn.Dropout(dropout, inplace=True)] 

        self.block=nn.Sequential(*self.block)

    def forward(self, x):
        output= self.block(x)
        return output