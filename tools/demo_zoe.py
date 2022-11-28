import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import open3d as o3d
import re
import os


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*.pcd')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        data_file_list.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.sample_file_list = data_file_list[200:]

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == 'nuscenes':
            ply_point_cloud = o3d.data.PLYPointCloud()
            pcd = o3d.t.io.read_point_cloud(str(self.sample_file_list[index]))
            time = 0.0*np.expand_dims(np.ones(pcd.point["intensity"].shape[0]), axis=1)
            points = np.hstack((pcd.point["positions"].numpy(),pcd.point["intensity"].numpy(),time))
            points[:, [1, 0]] = points[:, [0, 1]]
            # points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5)
        elif self.ext == 'kitti':
            ply_point_cloud = o3d.data.PLYPointCloud()
            pcd = o3d.t.io.read_point_cloud(str(self.sample_file_list[index]))
            # points = pcd.point["positions"].numpy()
            points = np.hstack((pcd.point["positions"].numpy(), pcd.point["intensity"].numpy()/255))
            points[:, [1, 0]] = points[:, [0, 1]]
            # points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5)
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--data_path', type=str,
                        default='/home/khushdeep/Desktop/zaafre/zoe_pointcloud/2022_Apr_12-16_00_59',
                        help='specify the point cloud data file or directory')

    parser.add_argument('--ext', type=str, default='kitti', help='specify the extension of your point cloud data file')


    args = parser.parse_args()


    if args.ext =='nuscenes':
        args.cfg_file = 'cfgs/zoe_models/cbgs_voxel0075_res3d_centerpoint.yaml'
        args.ckpt = 'ckpts/nuscenes/cbgs_voxel0075_centerpoint_nds_6648.pth'

    elif args.ext =='kitti':
        args.cfg_file = 'cfgs/zoe_models/second.yaml'
        args.ckpt = '/home/khushdeep/Desktop/OpenPCDet/tools/ckpts/kitti/second_7862.pth'

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    count = 1
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            index_vehicle = torch.where(pred_dicts[0]['pred_labels'] == 1)[0].cpu().tolist()
            V.draw_scenes(
                points=data_dict['points'][:, 1:],filename = str(count), ref_boxes=pred_dicts[0]['pred_boxes'][index_vehicle, :],
                ref_scores=pred_dicts[0]['pred_scores'][index_vehicle],
                ref_labels=pred_dicts[0]['pred_labels'][index_vehicle]
            )
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )
            count+=1

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
