import torch.nn.functional
from torch import nn

from utils.config import Config
import argparse

from models.gcn_utils import *
class GCN_TRAJ(nn.Module):
    def __init__(self, model_specs):
        super().__init__()
        self.input_dim = input_dim = model_specs['input_dim']
        self.cont_dim = cont_dim = model_specs.get('cont_dim', 84)
        self.out_dim = out_dim = input_dim
        self.aux_dim = aux_dim = model_specs.get('aux_dim', 0)
        self.nh_rnn = nh_rnn = model_specs['nh_rnn']
        self.dct_n = dct_n = model_specs['dct_n']
        self.root_net_is_bn = root_net_is_bn = model_specs.get('root_net_is_bn',False)
        self.root_net_resblock = root_net_resblock = model_specs.get('root_net_resblock',2)
        self.point_feat = point_feat = model_specs['point_feat']
        self.point_extra_feat = point_extra_feat = model_specs.get('point_extra_feat',0)
        self.wscene = wscene = model_specs.get('wscene',True)
        self.wcont = wcont = model_specs.get('wcont',True)
        self.dist_flag = model_specs.get('dist_flag',False)

        # * Number of channels depends on whether we use additional distance or not
        if self.dist_flag:
            cont_channels = 5
            cont_dim = 105
        else:
            cont_channels = 4


        # encode input human poses
        #self.x_enc = nn.Linear(input_dim+aux_dim, nh_rnn)
        self.x_stsgcn_enc = STSGCN_layer(3, nh_rnn, (1,1), 1, time_dim=30, joints_dim=21)

        # encode pose sequences
        #self.x_gru = nn.GRU(nh_rnn,nh_rnn)
        self.x_time_enc = nn.Sequential(nn.Linear(30, 10),
                                      nn.Tanh(),
                                      nn.Linear(10, 1),
                                      nn.Tanh())
        
        self.x_node_enc = nn.Sequential(nn.Linear(21, 10),
                                        nn.Tanh(),
                                        nn.Linear(10, 1),
                                        nn.Tanh())


        # encode contact
        if wcont:
            self.cont_stsgcn_enc = STSGCN_layer(cont_channels, nh_rnn, (1,1), 1, time_dim=60, joints_dim=21)
            self.cont_time_enc = nn.Sequential(nn.Linear(60, 10),
                                      nn.Tanh(),
                                      nn.Linear(10, 1),
                                      nn.Tanh())
            self.cont_node_enc = nn.Sequential(nn.Linear(21, 10),
                                        nn.Tanh(),
                                        nn.Linear(10, 1),
                                        nn.Tanh())

        # predict global trajectory via dct
        if not root_net_is_bn:
            if wcont:
                self.g_w1 = nn.Sequential(nn.Linear(dct_n*3+nh_rnn,nh_rnn),
                                          nn.Tanh())
            else:
                self.g_w1 = nn.Sequential(nn.Linear(dct_n*3+nh_rnn,nh_rnn),
                                          nn.Tanh())
            self.g_w2 = nn.Sequential(nn.Linear(nh_rnn,nh_rnn),
                                      nn.Tanh(),
                                      nn.Linear(nh_rnn,nh_rnn),
                                      nn.Tanh())
            self.g_w3 = nn.Sequential(nn.Linear(nh_rnn,nh_rnn),
                                      nn.Tanh(),
                                      nn.Linear(nh_rnn,nh_rnn),
                                      nn.Tanh())
            self.g_w4 = nn.Linear(nh_rnn,dct_n*3)
        else:
            if wcont:
                self.g_w1 = nn.Sequential(nn.Linear(dct_n*3+nh_rnn,nh_rnn),
                                          nn.BatchNorm1d(nh_rnn),
                                          nn.Tanh())
            else:
                self.g_w1 = nn.Sequential(nn.Linear(dct_n*3+nh_rnn,nh_rnn),
                                          nn.BatchNorm1d(nh_rnn),
                                          nn.Tanh())
            self.g_w2 = nn.Sequential(nn.Linear(nh_rnn,nh_rnn),
                                      nn.BatchNorm1d(nh_rnn),
                                      nn.Tanh(),
                                      nn.Linear(nh_rnn,nh_rnn),
                                      nn.BatchNorm1d(nh_rnn),
                                      nn.Tanh())
            self.g_w3 = nn.Sequential(nn.Linear(nh_rnn,nh_rnn),
                                      nn.BatchNorm1d(nh_rnn),
                                      nn.Tanh(),
                                      nn.Linear(nh_rnn,nh_rnn),
                                      nn.BatchNorm1d(nh_rnn),
                                      nn.Tanh())
            self.g_w4 = nn.Linear(nh_rnn,dct_n*3)


        # encode temporal information
        out_in_dim = aux_dim + out_dim


        self.end_goal = True
        if self.end_goal:
            if wcont:
                out_in_dim += cont_dim
            self.out_enc = nn.Linear(out_in_dim+3*2, nh_rnn)
        else:
            if wcont:
                out_in_dim += cont_dim
            self.out_enc = nn.Linear(out_in_dim+3, nh_rnn)


        # decode pose sequence
        self.out_gru = nn.GRUCell(nh_rnn, nh_rnn)

        # decode pose
        self.out_mlp = nn.Linear(nh_rnn, out_dim)

    def forward(self, x, cont, cont_mask, aux=None, horizon=60,
                dct_m=None, idct_m=None, root_idx=None):
        """
        x: [seq,bs,dim] -> [30, 4, 63]
        scene: [bs, 3, N]
        cont: bs, seq, nj*4 -> [4, 60, 21*4]
        aux: [bs,seq,...] or None # IS NONE
        cont_mask: NEVER USED
        dct_m: -> [60, 90]
        idct_m:  -> [90, 60]
        root_idx: -> [3], exactly idxs of root joints 42,43,44
        """
        
        # * Human Pose
        t_his, bs, nfeat = x.shape
        xi = x.reshape(t_his, bs, -1, 3).permute(1,3,0,2) # BCTV
        b,c,t,v = xi.shape

        # * Contact
        bs_cont, t_cont, nfeat_cont = cont.shape
        bs_cont, t_cont, n_cont, cha_c = cont.reshape(bs_cont, t_cont, 21, -1).shape


        if aux is not None:
            hx = torch.cat([x,aux[:,:x.shape[0]].transpose(0,1)],dim=-1)
        else:
            hx = xi

        hx = self.x_stsgcn_enc(hx).permute(0,1,3,2).reshape(b, -1, t) # [bs, t, c*v]
        hx = self.x_time_enc(hx).squeeze(-1).reshape(b,-1,v) # [b,c,v]
        hx = self.x_node_enc(hx).squeeze(-1) # [b,c]

        # * encode cont
        if self.wcont:
            cont_tmp = cont.clone()
            bc, tc, nc = cont_tmp.shape
            cont_tmp = cont_tmp.reshape(bc, tc, -1, cha_c).permute(0,3,1,2) # BCTV
            bc, cc, tc, vc = cont_tmp.shape
            hcont = self.cont_stsgcn_enc(cont_tmp).permute(0,1,3,2).reshape(bc, -1, tc) # [bs, t, c*v]
            hcont = self.cont_time_enc(hcont).squeeze(-1).reshape(bc,-1,vc) # [b,c,v]
            hcont = self.cont_node_enc(hcont).squeeze(-1) # [b,c]
            
    

        # * padded idx for dct, the first 30 frames are numbered and the rest are the last frame (29) repeated for 60 times 
        pad_idx = list(range(t_his))+[t_his-1]*horizon # [seq+horizon]

        # * dct
        root_his = x[:,:,root_idx][pad_idx].transpose(0,1) # [bs,seq+horizon,3]
        root_his_dct = torch.matmul(dct_m[None],root_his).reshape([bs,-1]) # b,60,3 -> b,60*3

        if self.wcont:
            # root_h = self.g_w1(torch.cat([root_his_dct,hx,hcont],dim=1)) # [bs, 180+128+128] -> [BS, 180]||[BS, t, 128]
            root_h = self.g_w1(torch.cat([root_his_dct,hcont],dim=1)) # [bs, 180+128+128] -> [BS, 180]||[BS, t, 128]
        
        else:
            root_h = self.g_w1(torch.cat([root_his_dct,hx],dim=1))

        # * Root prediction
        root_h = self.g_w2(root_h) + root_h
        root_h = self.g_w3(root_h) + root_h
        root_pred = self.g_w4(root_h) + root_his_dct # [bs, 60*3]
        # * idct
        root_traj = torch.matmul(idct_m[None],root_pred.reshape([bs,-1,len(root_idx)])) #[bs,t_total=90,3]

        return root_traj



# Create main model
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='gta_stage2_GCN_TRAJ')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--is_amp', action='store_true', default=True)

    parser.add_argument('--is_dist', action='store_true', default=False)
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=4, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Data
    x = torch.rand(30, 4, 63).to(device)
    cont = torch.rand(4, 60, 21*4).to(device)
    cont_mask = torch.rand(4, 60, 21).to(device)
    dct_m = torch.rand(60, 90).to(device)
    idct_m = torch.rand(90, 60).to(device)
    root_idx = [42,43,44]

    from models.motion_pred import *

    cfg = Config(f'{args.cfg}', test=args.test)
    dtype = torch.float32
    model = get_model(cfg).to(dtype=dtype, device=device)

    # Number of parameters
    print('Number of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # Forward
    y, root_traj = model(x, cont, None, None, 60, dct_m=dct_m, idct_m=idct_m, root_idx=root_idx)
    print(y.shape, root_traj.shape)



# Create main model
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='gta_stage2_GCN_TRAJ')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--is_amp', action='store_true', default=True)

    parser.add_argument('--is_dist', action='store_true', default=False)
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=4, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Data
    x = torch.rand(30, 4, 63).to(device)
    cont = torch.rand(4, 60, 21*4).to(device)
    cont_mask = torch.rand(4, 60, 21).to(device)
    dct_m = torch.rand(60, 90).to(device)
    idct_m = torch.rand(90, 60).to(device)
    root_idx = [42,43,44]

    from models.motion_pred import *

    cfg = Config(f'{args.cfg}', test=args.test)
    dtype = torch.float32
    model = get_model(cfg).to(dtype=dtype, device=device)

    # Number of parameters
    print('Number of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # Forward
    y, root_traj = model(x, cont, None, None, 60, dct_m=dct_m, idct_m=idct_m, root_idx=root_idx)
    print(y.shape, root_traj.shape)
