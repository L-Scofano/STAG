import os
import sys
import math
import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

sys.path.append(os.getcwd())
from utils.config import Config
from datasets.dataset_gta import DatasetGTA
from models.motion_pred import *
from utils import *
from utils.util import *

from datetime import datetime

import wandb


@torch.no_grad()
def train(epoch):
    thres = math.exp(-0.5 * args.cont_thre ** 2 / dataset.sigma ** 2)
    root_joint_idx = 14
    root_idx = [root_joint_idx*3,root_joint_idx*3+1,root_joint_idx*3+2]

    generator = DataLoader(dataset,batch_size=cfg.batch_size,shuffle=True,
                           num_workers=2,pin_memory=True,drop_last=False)

    pose_err = np.zeros(t_pred) # one value for each frame
    path_err = np.zeros(t_pred)
    all_err = np.zeros(t_pred)

    total_num_sample = 1e-20
    pad_idx = list(range(t_his)) + [t_his-1]*t_pred

    y_for_save = {}
    for pose, scene_vert, scene_origin, _, item_key in tqdm(generator):
        
        # * pose: [bs, t_total, 21, 3]
        # * scene_vert: [bs, 10000, 3]
        # * scene_origin: [bs, 3]
        # * item_key: [bs]

        bs = pose.shape[0] # batch size
        nj = pose.shape[2] # 21
        scene_vert = scene_vert.to(device=device)
        npts = scene_vert.shape[1]

        joints = pose.to(device=device)

        # * This is were they calculate the distance between joints. At time t the output is [npts, 21] where there is a distance for each joint
        is_cont = (scene_vert[:, None, :, None, :] - joints[:, :, None, :, :]).norm(dim=-1) # [bs, t_total, npts, 21]

        # * In such a contact map, closer scene points have higher values than far-away ones, whose values will be very close to zero
        is_cont_gauss = torch.exp(-0.5*is_cont**2/dataset.sigma**2)

        joints_orig = joints[:, :, 14:15]
        joints = joints - joints_orig
        joints[:, :, 14:15] = joints_orig


        # * args.w_est_cont: esternal computed contact points
        if args.w_est_cont:
            is_cont_pad = is_cont_gauss[:, pad_idx].reshape([bs, t_his + t_pred, -1])
            is_cont_dct = torch.matmul(dct_m_cont[None], is_cont_pad).reshape([bs, dct_n_cont, npts, nj])
            is_cont_dct = is_cont_dct.permute(0, 1, 3, 2).reshape([bs, dct_n_cont * nj, npts])

            # def forward(self, x, scene, aux=None, cont_dct=None):
            cont_dct_est = model_cont(joints[:, :t_his].reshape([bs, t_his, -1]).transpose(0, 1), scene_vert.transpose(1, 2),
                                      cont_dct=is_cont_dct)  # (x, z, scene, aux_data=None, horizon=30, nk=5)

            cont_dct_est = cont_dct_est.reshape([bs, dct_n_cont, nj, npts]).reshape([bs, dct_n_cont, nj * npts])
            cont_est = torch.matmul(idct_m_cont[None], cont_dct_est)
            cont_est = cont_est.reshape([bs, t_his + t_pred, nj, npts]).transpose(2, 3)[:,t_his:]

            # * Values of the contact map are normalized to [0, 1]
            is_cont_est = 1 - cont_est

            min_dist_value = (is_cont_est.min(dim=2)[0] < (1-thres)).to(dtype=dtype)
            
            min_dist_idx = is_cont_est.min(dim=2)[1].reshape([-1])
            idx_tmp = torch.arange(bs, device=device)[:, None].repeat([1, t_pred * nj]).reshape([-1])

            cont_points = scene_vert[idx_tmp, min_dist_idx, :].reshape([bs, t_pred, nj, 3]) # [bs, t_pred, 21, 3]
            
            cont_points = cont_points * min_dist_value[..., None]

            cont_points = torch.cat([cont_points, min_dist_value[..., None]], dim=-1)


        # * args.w_est_cont: ground truth contact points
        if not args.w_est_cont:
            dist = is_cont

            # * They only take one value for each joint
            min_dist_value = (dist.min(dim=2)[0] < 0.3).to(dtype=dtype) # [bs, t_total, 21] it is a 0/1 tensor
            min_dist_idx = dist.min(dim=2)[1].reshape([-1]) # [bs*t_total*21]
            
            # * They create a temporary tensor to store the index of the batch
            idx_tmp = torch.arange(bs, device=device)[:, None].repeat([1, (t_pred + t_his) * nj]).reshape([-1])

            cont_points = scene_vert[idx_tmp, min_dist_idx, :].reshape([bs, t_his + t_pred, nj, 3]) # [bs, t_total, 21, 3]

            # * They multiply the contact points by a 0/1 value to remove the points that are not in contact
            cont_points = cont_points * min_dist_value[..., None] # [bs, t_total, 21, 3]
            cont_points = torch.cat([cont_points, min_dist_value[..., None]], dim=-1)[:,t_his:]

        y, root_traj = model(joints[:,:t_his].reshape([bs,t_his,-1]).transpose(0,1),
                              cont_points.reshape([bs,t_pred,-1]) if wcont else None,
                              None,
                              None, t_pred,
                              dct_m=dct_m, idct_m=idct_m,
                              root_idx=root_idx)


        y[:,:,root_idx] = root_traj[:,t_his:].transpose(0,1)
        y = y.transpose(0, 1)
        y = y.reshape([bs, t_pred, -1, 3])
        joints = joints[:, t_his:]

        """mpjpe error"""
        path_err += (y[:, :, 14] - joints[:, :, 14]).norm(dim=-1).sum(dim=0).cpu().data.numpy()
        pose_idx = np.setdiff1d(np.arange(21), 14)
        pose_err += (y[:, :, pose_idx] - joints[:, :, pose_idx]).norm(dim=-1).mean(dim=-1).sum(dim=0).cpu().data.numpy()

        y[:, :, pose_idx] = y[:, :, pose_idx] + y[:, :, 14:15]
        joints[:, :, pose_idx] = joints[:, :, pose_idx] + joints[:, :, 14:15]
        all_err += (y - joints).norm(dim=-1).mean(dim=-1).sum(dim=0).cpu().data.numpy()

        if args.save_joint:
            y_tmp = y + scene_origin.to(device=device)[:,None]
            for ii, ik in  enumerate(item_key):
                y_for_save[ik] = y_tmp[ii].cpu().data.numpy()

        total_num_sample += y.shape[0]

    path_err = path_err * 1000 / total_num_sample
    pose_err = pose_err * 1000 / total_num_sample
    all_err = all_err * 1000 / total_num_sample

    log_idxs1 = np.array([14, 29, 44, 59])
    log_idxs = np.arange(60)
    header = ['err'] + list(np.arange(t_pred)[log_idxs]) + ['mean']
    idxs_log = [14, 29, 44, 59]
    header_log = ["err"] + list(np.arange(t_pred)[idxs_log]) + ["mean"]
    csv_dir = f'{cfg.result_dir}/err_{args.mode}_cont_model_{args.cfg_cont if args.w_est_cont else "gt"}.csv'

    # * Log losses to wandb
    # if cfg.wandb_test['mode'] == 'online':
    #     train_platform.report_scalar(name_val = 'path_err',  value=path_err)
    #     train_platform.report_scalar(name_val = 'pose_err',  value=pose_err)
    #     train_platform.report_scalar(name_val = 'all_err',  value=all_err)


    # print all of the losses
    # print('path_err', path_err[log_idxs])
    
    log_idx_frames = np.array([15, 30, 45, 60])
    # print the losses for all log_idx_frames and the mean
    # if cfg.wandb_test['mode'] == 'online':
    #     for i in range(len(log_idx_frames)):
    #         train_platform.report_scalar(name_val = f'path_err_{log_idx_frames[i]}',  value=path_err[log_idxs1[i]])
    #         train_platform.report_scalar(name_val = f'pose_err_{log_idx_frames[i]}',  value=pose_err[log_idxs1[i]])
    #         train_platform.report_scalar(name_val = f'all_err_{log_idx_frames[i]}',  value=all_err[log_idxs1[i]])
    #     train_platform.close()
    for i in range(len(log_idx_frames)):
        print(f'path_err_{log_idx_frames[i]}', path_err[log_idxs1[i]])
        print(f'pose_err_{log_idx_frames[i]}', pose_err[log_idxs1[i]])
        print(f'all_err_{log_idx_frames[i]}', all_err[log_idxs1[i]])
    print('---------------------------------')
    print('path_err', path_err[log_idxs].mean())    
    # print('pose_err', pose_err[log_idxs])
    print('pose_err', pose_err[log_idxs].mean())
    # print('all_err', all_err[log_idxs])
    print('all_err', all_err[log_idxs].mean())



    with open(csv_dir, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        writer.writerow(
            [
                dt_string,
            ]
        )   

        writer.writerow(header_log)
        data_path = ["path_err"] + list(path_err[idxs_log]) + [path_err.mean()]
        writer.writerow(data_path)
        print("path_err")

        data_joint = ["joint_err"] + list(pose_err[idxs_log]) + [pose_err.mean()]
        writer.writerow(data_joint)
        print("joint_err")

        data_all_joint = ["all_joint_err"] + list(all_err[idxs_log]) + [all_err.mean()]
        writer.writerow(data_all_joint)
        print("all_joint_err")

        if cfg.wandb_test['mode'] == 'online':
            for name, value in zip(header_log[1:], data_path[1:]):
                print(f"\t{name}: {value}")
                wandb.log({f"eval/{data_path[0]}/{name}": value})


            for name, value in zip(header_log[1:], data_joint[1:]):
                print(f"\t{name}: {value}")
                wandb.log({f"eval/{data_joint[0]}/{name}": value})


            for name, value in zip(header_log[1:], data_all_joint[1:]):
                print(f"\t{name}: {value}")
                wandb.log({f"eval/{data_all_joint[0]}/{name}": value})

    if args.save_joint:
        np.savez_compressed(f'{cfg.result_dir}/prediction_{args.mode}.npz', y=y_for_save)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_cont', default='gta_stage1_PVCNN2_DCT_CONT')
    parser.add_argument('--cfg', default='gta_stage2_GRU_POSE')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--w_est_cont', action='store_true', default=False)
    parser.add_argument('--iter', type=int, default=50)
    parser.add_argument('--iter_cont', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--cont_thre', type=float, default=0.3)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--save_joint', action='store_true', default=False)
    parser.add_argument('--run_name', type=str, default='test_s2')
    # parser.add_argument('--wandb', action='store_true', default=False)
    

    args = parser.parse_args()

   
    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)

    cfg = Config(f'{args.cfg}', test=args.test)

    if cfg.wandb_test['mode'] == 'online':
        # !TODO: wandb in a more elegant way
        wandb.init(
        mode=cfg.wandb_test['mode'],
        project=cfg.wandb_test['project'],
        entity=cfg.wandb_test['entity'],
        group=cfg.wandb_test['group'],
        job_type=cfg.wandb_test['job_type'],
        name=cfg.wandb_test['name'],
        tags=cfg.wandb_test['tags'],
        notes=cfg.wandb_test['notes'],
        resume=cfg.wandb_test['resume'],
        save_code=cfg.wandb_test['save_code'])


    cfg_cont = Config(f'{args.cfg_cont}', test=args.test)

    logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))
    logger.info(args)

    """parameter"""
    mode = args.mode
    cfg.model_specs['nk'] = nk = 1
    nz = cfg.model_specs['nz']
    rand_rot = cfg.dataset_specs['random_rot']
    t_his = cfg.dataset_specs['t_his']
    t_pred = cfg.dataset_specs['t_pred']
    t_total = t_his + t_pred
    over_all_step = 0
    cfg.dataset_specs['random_rot'] = False
    cfg.dataset_specs['step'] = args.step
    cfg.batch_size = bs = args.bs

    # get contact dct_m, idct_m
    dct_n_cont = cfg_cont.model_specs['dct_n']
    t_total = t_his + t_pred
    dct_m, idct_m = get_dct_matrix(t_total, is_torch=True)
    dct_m_cont = dct_m.to(dtype=dtype, device=device)[:dct_n_cont]
    idct_m_cont = idct_m.to(dtype=dtype, device=device)[:,:dct_n_cont]

    dct_n = cfg.model_specs['dct_n']
    dct_m = dct_m.to(dtype=dtype, device=device)[:dct_n]
    idct_m = idct_m.to(dtype=dtype, device=device)[:,:dct_n]

    """data"""
    wscene = cfg.model_specs.get('wscene', True)
    wcont = cfg.model_specs.get('wcont', True)
    cfg_cont.dataset_specs['wscene'] = True
    cfg_cont.dataset_specs['wcont'] = True
    dataset_cls = DatasetGTA if cfg.dataset == 'GTA' else None
    dataset = dataset_cls(args.mode, cfg_cont.dataset_specs)
    logger.info(f">>> total sub sequences: {dataset.__len__()}")

    """model"""
    model = get_model(cfg)
    model.float()
    logger.info(">>> total params: {:.5f}M".format(sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    model_cont = get_model(cfg_cont)
    model_cont.float()
    logger.info(">>> total params in contact model: {:.5f}M".format(sum(p.numel() for p in list(model_cont.parameters())) / 1000000.0))

    if args.iter > 0:
        cp_path = cfg.model_path % args.iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = torch.load(cp_path, map_location='cpu')
        model.load_state_dict(model_cp['model_dict'])

    if args.iter_cont > 0:
        cp_path = cfg_cont.model_path % args.iter_cont
        print('loading contact model from checkpoint: %s' % cp_path)
        model_cp = torch.load(cp_path, map_location='cpu')
        model_cont.load_state_dict(model_cp['model_dict'])


    model.to(device)
    model.eval()
    model_cont.to(device)
    model_cont.eval()
    train(args.iter)