from typing import Dict
import os
import math
import wandb
from tqdm import tqdm
import csv
from datetime import datetime
import sys

# import hydra
# from hydra.utils import instantiate
# from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.dataset_gta import DatasetGTA
# from src.utils.logger import create_logger
from utils.util import get_dct_matrix

from utils import *
from utils.util import *
from models.motion_pred import *


import argparse
from utils.config import Config

@torch.no_grad()
def eval(
    dataset: DatasetGTA,
    dct: Dict,
    models: Dict,
    device: torch.device,
    dtype: torch.dtype,
    cfg,
) -> None:
    thres = math.exp(-0.5 * cfg.dataset_specs["cont_thre"]**2 / cfg.dataset_specs["sigma"]**2)
    root_joint_idx = 14
    root_idx = [root_joint_idx * 3, root_joint_idx * 3 + 1, root_joint_idx * 3 + 2]

    generator = DataLoader(dataset,batch_size=cfg.batch_size,shuffle=True,
                           num_workers=2,pin_memory=True,drop_last=False)

    # * Variables
    in_frames_n = cfg.dataset_specs["t_his"]
    out_frames_n = cfg.dataset_specs["t_pred"]
    dist_flag = cfg.model_specs['dist_flag']


    pose_err = np.zeros(out_frames_n)
    path_err = np.zeros(out_frames_n)
    all_err = np.zeros(out_frames_n)
    contact_err = np.zeros(out_frames_n)

    total_num_sample = 1e-20
    pad_idx = list(range(in_frames_n)) + [in_frames_n - 1] * out_frames_n

    y_for_save = {}
    for pose, scene_vert, scene_origin, _, item_key in tqdm(generator):
        bs = pose.shape[0]
        nj = pose.shape[2]
        scene_vert = scene_vert.to(device=device)  # [:,:10000]
        npts = scene_vert.shape[1]

        joints = pose.to(device=device)

        # Distance matrix.
        dist_m = (scene_vert[:, None, :, None, :] - joints[:, :, None, :, :]).norm(
            dim=-1
        )
        # Standardize the distance matrix.
        # ! is_cont_gauss
        stand_dist_m = torch.exp(-0.5 * dist_m**2 / cfg.dataset_specs["sigma"]**2)

        joints_orig = joints[:, :, 14:15]
        joints = joints - joints_orig
        joints[:, :, 14:15] = joints_orig

        # * If we want to evaluate the contact points
        if cfg.train_eval["s1"]:
            # Predict contact points.
            cont_dct_n = dct["contact"]["dim"]
            cont_dct_m = dct["contact"]["dct_m"]
            cont_idct_m = dct["contact"]["idct_m"]

            pad_dist_m = stand_dist_m[:, pad_idx].reshape(
                [bs, in_frames_n + out_frames_n, -1]
            )
            dct_dist_m = torch.matmul(cont_dct_m[None], pad_dist_m).reshape(
                [bs, cont_dct_n, npts, nj]
            )
            dct_dist_m = dct_dist_m.permute(0, 1, 3, 2).reshape(
                [bs, cont_dct_n * nj, npts]
            )

            # * Prediction.
            model_cont = models["contact"]
            dct_cont_pred = model_cont(
                x=joints[:, :in_frames_n]
                .reshape([bs, in_frames_n, -1])
                .transpose(0, 1),
                scene=scene_vert.transpose(1, 2),
                aux=None,
                cont_dct=dct_dist_m,
            )  
            dct_cont_pred = dct_cont_pred.reshape([bs, cont_dct_n, nj, npts]).reshape(
                [bs, cont_dct_n, nj * npts]
            )
            cont_pred = torch.matmul(cont_idct_m[None], dct_cont_pred)
            cont_pred = cont_pred.reshape(
                [bs, in_frames_n + out_frames_n, nj, npts]
            ).transpose(2, 3)[:, in_frames_n:]


            # cont_est normalized at some point
            dist_m_pred = 1 - cont_pred

            min_dist_value = (dist_m_pred.min(dim=2)[0] < (1 - thres)).to(dtype=dtype)
            # min_dist_value = (dist_m_pred.min(dim=2)[0]).to(dtype=dtype)
            min_dist_idx = dist_m_pred.min(dim=2)[1].reshape([-1])
            idx_tmp = (
                torch.arange(bs, device=device)[:, None]
                .repeat([1, out_frames_n * nj])
                .reshape([-1])
            )

            cont_points = scene_vert[idx_tmp, min_dist_idx, :].reshape(
                [bs, out_frames_n, nj, 3]
            )
            cont_points = cont_points * min_dist_value[..., None]  # !
            # copy the contact points
            cont_points_copy = cont_points.detach().clone()

            cont_points = torch.cat([cont_points, min_dist_value[..., None]], dim=-1)

            # *GT contact points.
            min_dist_value = (dist_m.min(dim=2)[0] < 0.3).to(dtype=dtype) # ! changed to stand_dist_m
            min_dist_idx = dist_m.min(dim=2)[1].reshape([-1])
            idx_tmp = (
                torch.arange(bs, device=device)[:, None]
                .repeat([1, (out_frames_n + in_frames_n) * nj])
                .reshape([-1])
            )

            cont_points_gt = scene_vert[idx_tmp, min_dist_idx, :].reshape(
                [bs, in_frames_n + out_frames_n, nj, 3]
            )
            cont_points_gt = cont_points_gt * min_dist_value[..., None]

            contact_err += (
            (cont_points_copy - cont_points_gt[:,in_frames_n:,:,:])
            .norm(dim=-1)
            .mean(dim=-1)
            .sum(dim=0)
            .cpu()
            .data.numpy()
                )
            total_num_sample += cont_points_copy.shape[0]
            
        if cfg.train_eval["s2"]:

            # Use ground truth contact points.
            min_dist_value = (dist_m.min(dim=2)[0] < 0.3).to(dtype=dtype)
            min_dist_idx = dist_m.min(dim=2)[1].reshape([-1])
            idx_tmp = (
                torch.arange(bs, device=device)[:, None]
                .repeat([1, (out_frames_n + in_frames_n) * nj])
                .reshape([-1])
            )

            cont_points = scene_vert[idx_tmp, min_dist_idx, :].reshape(
                [bs, in_frames_n + out_frames_n, nj, 3]
            )
            cont_points = cont_points * min_dist_value[..., None]
            cont_points = torch.cat([cont_points, min_dist_value[..., None]], dim=-1)[
                :, in_frames_n:
            ]


            if dist_flag:
                min_dist= dist_m.min(dim=2)[0]
                min_dist = min_dist*min_dist_value
                cont_points = torch.cat([cont_points,min_dist[:,in_frames_n:,:,None]],dim=-1)


            # * Pose prediction.
            pose_dct_n = dct["pose"]["dim"]
            pose_dct_m = dct["pose"]["dct_m"]
            pose_idct_m = dct["pose"]["idct_m"]
            model_pose = models["pose"]
            root_traj = model_pose(
                x=joints[:, :in_frames_n].reshape([bs, in_frames_n, -1]).transpose(0, 1),
                cont=cont_points.reshape([bs, out_frames_n, -1])
                if cfg.model_specs["wcont"]
                else None,
                cont_mask=None,
                aux=None,
                horizon=out_frames_n,
                dct_m=pose_dct_m,
                idct_m=pose_idct_m,
                root_idx=root_idx,
            )
            root_pred = root_traj[:, in_frames_n:]
            joints = joints[:, in_frames_n:]

            # * MPJPE
            path_err += (
                (root_pred - joints[:, :, 14]).norm(dim=-1).sum(dim=0).cpu().data.numpy()
            )

            total_num_sample += root_traj.shape[0]

    # * Contact points.
    if cfg.train_eval["s1"]:

        # * Metrics.
        contact_err = contact_err * 1000 / total_num_sample

        idxs = list(range(60))
        header = ["err"] + list(np.arange(out_frames_n)[idxs]) + ["mean"]

        idxs_log = [14, 29, 44, 59]
        header_log = ["err"] + list(np.arange(out_frames_n)[idxs_log]) + ["mean"]
        
        csv_dir = f'{cfg.result_dir}/err_{cfg.train_eval["mode"]}_cont_model_{"gt"}.csv'


        # * Log.
        with open(csv_dir, "a", encoding="UTF8") as f:
            writer = csv.writer(f)

            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            writer.writerow(
                [
                    dt_string,
                ]
            )   
            writer.writerow(header)

            data = ["contact_err"] + list(contact_err[idxs]) + [path_err.mean()]
            writer.writerow(data)

            writer.writerow(header_log)
            data = ["contact_err"] + list(contact_err[idxs_log]) + [contact_err.mean()]
            writer.writerow(data)
            
            print("contact_err")
            for name, value in zip(header_log[1:], data[1:]):
                print(f"\t{name}: {value}")
                wandb.log({f"eval/{data[0]}/{name}": value})



    else:
        # * Metrics.
        path_err = path_err * 1000 / total_num_sample

        idxs = list(range(60))
        header = ["err"] + list(np.arange(out_frames_n)[idxs]) + ["mean"]

        idxs_log = [14, 29, 44, 59]
        header_log = ["err"] + list(np.arange(out_frames_n)[idxs_log]) + ["mean"]
        csv_dir = f'{cfg.result_dir}/err_{cfg.train_eval["mode"]}_cont_model_{"gt"}.csv'

        # * Log.
        with open(csv_dir, "a", encoding="UTF8") as f:
            writer = csv.writer(f)

            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            writer.writerow(
                [
                    dt_string,
                ]
            )   

            writer.writerow(header)

            data = ["path_err"] + list(path_err[idxs]) + [path_err.mean()]
            writer.writerow(data)
        

            writer.writerow(header_log)
            data = ["path_err"] + list(path_err[idxs_log]) + [path_err.mean()]
            writer.writerow(data)
            print("path_err")
            for name, value in zip(header_log[1:], data[1:]):
                print(f"\t{name}: {value}")
                wandb.log({f"eval/{data[0]}/{name}": value})



def main(cfg, dtype: torch.dtype) -> None:
    """
    Main function
    Args:
        cfg: configuration
        dtype: data type
    """

    # * Setup
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # * Wandb init
    wandb.init(
    mode=cfg.wandb_train['mode'],
    project=cfg.wandb_train['project'],
    entity=cfg.wandb_train['entity'],
    group=cfg.wandb_train['group'],
    job_type=cfg.wandb_train['job_type'],
    name=cfg.wandb_train['name'],
    tags=cfg.wandb_train['tags'],
    notes=cfg.wandb_train['notes'],
    resume=cfg.wandb_train['resume'],
    save_code=cfg.wandb_train['save_code'])

    # * DCT matrices.
    dct_m, idct_m = get_dct_matrix(
        cfg.dataset_specs["t_his"] + cfg.dataset_specs["t_pred"], is_torch=True
    )

    if cfg.train_eval["s1"]:
        cont_dct_n = cfg.model_specs["dct_n"]
        cont_dct_m = dct_m.to(dtype=dtype, device=device)[:cont_dct_n]
        cont_idct_m = idct_m.to(dtype=dtype, device=device)[:, :cont_dct_n]

    ## Pose.
    pose_dct_n = cfg.model_specs["dct_n"]
    pose_dct_m = dct_m.to(dtype=dtype, device=device)[:pose_dct_n]
    pose_idct_m = idct_m.to(dtype=dtype, device=device)[:, :pose_dct_n]

    
    if cfg.dataset == "GTA":
        # change mode to test in hydra to load test data
        cfg.mode = "test"
        dataset = DatasetGTA(cfg.mode, cfg.dataset_specs)

    else:
        raise NotImplementedError(f"Dataset {cfg.dataset} not implemented.")
    

    if cfg.train_eval["s1"]:
        # * Model contact.
        model_cont = get_model(cfg).to(device=device, dtype=dtype)
        model_cont.float()


        # Load checkpoints.
        # * Contact,
        print("Loading contact model from checkpoint: %s" % cfg.train_eval["ckpt_path"])
        model_cont_ckpt = torch.load(
            cfg.train_eval["ckpt_path"], map_location=device
        )  # ! Why map location is cpu?
        model_cont.load_state_dict(model_cont_ckpt["model_dict"])
        model_cont.eval()

    if cfg.train_eval["s2"]:

        # * Model pose.
        model_pose = get_model(cfg).to(device=device, dtype=dtype)
        model_pose.float()

        # * Pose.
        print("Loading pose model from checkpoint: %s" % cfg.train_eval["ckpt_path"])
        model_pose_ckpt = torch.load(
            cfg.train_eval["ckpt_path"], map_location=device
        )  
        model_pose.load_state_dict(model_pose_ckpt["model_dict"])
        model_pose.eval()

    wandb.log({"eval/epoch": cfg.train_eval["epoch"]})

    if cfg.train_eval["s1"]:
        eval(
            dataset=dataset,
            dct={
                "contact": {"dct_m": cont_dct_m, "idct_m": cont_idct_m, "dim": cont_dct_n},
                "pose": {},
            },
            models={"contact": model_cont, "pose": None},
            device=device,
            dtype=dtype,
            cfg=cfg,
        )
        

    elif cfg.train_eval["s2"]:
        eval(
            dataset=dataset,
            dct={
                "contact": {},
                "pose": {"dct_m": pose_dct_m, "idct_m": pose_idct_m, "dim": pose_dct_n},
            },
            models={"contact": None, "pose": model_pose},
            device=device,
            dtype=dtype,
            cfg=cfg,
        )


