from models.gcn_gru import GCN_POSE
from models.rnn import *
from models.pvcnn_dct_gcn import *

from models.gcn_gru import *
from models.gcn import *
from models.gcn_gru_traj import *


named_models = {
                'PVCNN2_DCT_CONT_GCN': PVCNN2_DCT_CONT_GCN,
                'GCN_POSE':GCN_POSE,
                'GCN_TPOSE':GCN_TPOSE,
                'GCN_TRAJ':GCN_TRAJ,
                }

def get_model(cfg):
    model_name = cfg.model_name
    return named_models[model_name](cfg.model_specs)