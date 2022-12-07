#!/usr/bin/env python -V
import sys
sys.path.append("/opt/ros/melodic/lib/python2.7/dist-packages")

import os
import time
import json
import rospy
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from pyquaternion import Quaternion
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, Image
import torch
# import pcl

import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Pose

sys.path.append("/home/khushdeep/Desktop/ROS-tracker/catkin_ws/src:/opt/ros/melodic/share")
print(os.environ['PYTHONPATH'])
import tf
from tf2_ros import Buffer,TransformListener
from tf2_msgs.msg import TFMessage
from message_filters import Subscriber,TimeSynchronizer,ApproximateTimeSynchronizer

from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

from pcdet.datasets import DatasetTemplate
import glob
from pcdet.config import cfg, cfg_from_yaml_file
import argparse
from pathlib import Path
from pcdet.utils import common_utils
from pcdet.models import build_network, load_data_to_gpu

sys.path.append("/home/khushdeep/Desktop/3D-Multi-Object-Tracker")
from tracker.config import cfg, cfg_from_yaml_file
from zoe_3DMOT import Track_seq

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')),
                 (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')),
                 (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}

DUMMY_FIELD_PREFIX = '__'



class SecondROS:
    def __init__(self):
        rospy.init_node('second_ros')
        self.count_frame = 0

        # Subscriber
        self.sub_lidar = rospy.Subscriber("/zoe/velodyne_points", PointCloud2, self.lidar_callback, queue_size=1)
        # self.sub_lidar = Subscriber("/zoe/velodyne_points", PointCloud2)
        self.tf_listener = tf.TransformListener()

        # TODO : synchronize pointcloud and tf messages
        # Issues: pointcloud has time header, however in tf does not have the same
        # tf2_buffer = Buffer(cache_time=rospy.Duration(0.1))
        # self.tf2_listener = tf.TransformListener(cache_time=rospy.Duration(0.1))
        # print('Done tf2')
        # # self.listener = tf.TransformListener()
        # # self.listener = Subscriber("/tf", TFMessage)
        # self.ats = ApproximateTimeSynchronizer([self.sub_lidar, self.tf2_listener], queue_size=10, slop=0.1) # Thats why this shows error
        # self.ats.registerCallback(self.lidar_callback)
        ##########

        # Publisher
        self.pub_bbox = rospy.Publisher("/detections", BoundingBoxArray, queue_size=1)

        self.model = Detection()
        self.model.initialize()

        #Tracking
        parser = argparse.ArgumentParser(description='arg parser')
        parser.add_argument('--cfg_file', type=str, default="/home/khushdeep/Desktop/3D-Multi-Object-Tracker/config/online"
                                                            "/zoe.yaml",
                            help='specify the config for tracking')
        args = parser.parse_args()
        yaml_file = args.cfg_file

        config = cfg_from_yaml_file(yaml_file, cfg)
        self.Tracker = Track_seq(config)

        rospy.spin()
    
    def lidar_callback(self, msg):

        (trans, rot) = self.tf_listener.lookupTransform('zoe/world', 'zoe/velodyne', rospy.Time(0))
        print(trans)
        ego_pose= np.asarray(self.tf_listener.fromTranslationRotation(trans,rot))
        self.count_frame += 1
        # print(f'Matrix:{matrix}')

        intensity_fname = None
        intensity_dtype = None
        for field in msg.fields:
            if field.name == "i" or field.name == "intensity":
                intensity_fname = field.name
                intensity_dtype = field.datatype
            
        dtype_list = self._fields_to_dtype(msg.fields, msg.point_step)
        pc_arr = np.frombuffer(msg.data, dtype_list)
        
        if intensity_fname:
            pc_arr = structured_to_unstructured(pc_arr[["x", "y", "z", intensity_fname]]).copy()
            pc_arr[:, [1, 0]] = pc_arr[:, [0, 1]]
            pc_arr[:, 3] = pc_arr[:, 3] / 255
        else:
            pc_arr = structured_to_unstructured(pc_arr[["x", "y", "z"]]).copy()
            pc_arr = np.hstack((pc_arr, np.zeros((pc_arr.shape[0], 1))))

        lidar_boxes = self.model.predict(pc_arr)
        self.tracker(lidar_boxes, ego_pose)
        self.plot_bbox_lidar(lidar_boxes,ego_pose,msg.header.frame_id)


    def tracker(self,lidar_boxes,ego_pose):
        ####Test code
        lidar_boxes[0]['pred_boxes'] = torch.tensor([[10.0, 0.0, -1.0, 3.7428, 1.6276, 1.5223, 1.57],
                                                     [-10.0, 0.0, -1.0, 3.7428, 1.6276, 1.5223, 1.57],
                                                     [0.0, 10.0, -1.0, 3.7428, 1.6276, 1.5223, 1.57],
                                                     [0.0, -10.0, -1.0, 3.7428, 1.6276, 1.5223, 1.57]])
        lidar_boxes[0]['pred_scores'] = torch.tensor([0.5, 0.5, 0.5, 0.5])
        lidar_boxes[0]['pred_labels'] = torch.tensor([1, 1, 1, 1], dtype=int)

        #####
        if lidar_boxes is not None:
            boxes = lidar_boxes[0]['pred_boxes'].cpu().numpy()
            boxes_coord = np.append(boxes[:, 0:3], np.ones((boxes.shape[0], 1)), axis=1)
            scores = lidar_boxes[0]['pred_scores'].cpu().numpy()
            boxes_world = np.dot(ego_pose, boxes_coord.T).T
            boxes[:, 0:3] = boxes_world[:, 0:3]

        else:
            boxes = None
            scores = None

        tracker = self.Tracker.track_scene(boxes, scores, self.count_frame)

    def plot_bbox_lidar(self,lidar_boxes,ego_pose,msg_frame_id):

        if lidar_boxes is not None:
            num_detects = lidar_boxes[0]['pred_boxes'].shape[0]
            arr_bbox = BoundingBoxArray()
            # TODO no need to do these as everything is already an array
            for i in range(num_detects):
                bbox = BoundingBox()

                bbox.header.frame_id = msg_frame_id
                bbox.header.stamp = rospy.Time.now()

                bbox.pose.position.y = float(lidar_boxes[0]['pred_boxes'][i][0])
                bbox.pose.position.x = float(lidar_boxes[0]['pred_boxes'][i][1])
                bbox.pose.position.z = float(lidar_boxes[0]['pred_boxes'][i][2])
                # bbox.pose.position.z = float(lidar_boxes[0]['pred_boxes'][i][2]) + float(
                #     lidar_boxes[0]['pred_boxes'][i][5]) / 2
                bbox.dimensions.y = float(lidar_boxes[0]['pred_boxes'][i][3])  # width
                bbox.dimensions.x = float(lidar_boxes[0]['pred_boxes'][i][4])  # length
                bbox.dimensions.z = float(lidar_boxes[0]['pred_boxes'][i][5])  # height

                q = Quaternion(axis=(0, 0, 1), radians=float(lidar_boxes[0]['pred_boxes'][i][6]))
                bbox.pose.orientation.x = q.x
                bbox.pose.orientation.y = q.y
                bbox.pose.orientation.z = q.z
                bbox.pose.orientation.w = q.w

                if int(lidar_boxes[0]['pred_labels'][i]) == 1:
                    arr_bbox.boxes.append(bbox)
                    bbox.label = i
                    bbox.value = i

        else:
            boxes = None

    def _fields_to_dtype(self, fields, point_step):
        '''Convert a list of PointFields to a numpy record datatype.
        '''
        offset = 0
        np_dtype_list = []
        for f in fields:
            while offset < f.offset:
                # might be extra padding between fields
                np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
                offset += 1

            dtype = pftype_to_nptype[f.datatype]
            if f.count != 1:
                dtype = np.dtype((dtype, f.count))

            np_dtype_list.append((f.name, dtype))
            offset += pftype_sizes[f.datatype] * f.count

        # might be extra padding between points
        while offset < point_step:
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        return np_dtype_list


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
        self.sample_file_list = data_file_list

    def __len__(self):
        return 1

    def __getitem__(self, pointcloud):

        input_dict = {
            'points': pointcloud,
            'frame_id': 1,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


class Detection:
    def __init__(self):
        self.data_path = ''
    def initialize(self):

        parser = argparse.ArgumentParser(description='arg parser')

        parser.add_argument('--data_path', type=str,
                            default='/home/khushdeep/Desktop/zaafre/zoe_pointcloud/2022_Apr_12-16_00_59',
                            help='specify the point cloud data file or directory')

        args = parser.parse_args()

        # args.cfg_file = 'cfgs/zoe_models/cbgs_voxel0075_res3d_centerpoint.yaml'
        # args.ckpt = 'ckpts/nuscenes/cbgs_voxel0075_centerpoint_nds_6648.pth'

        args.cfg_file = 'cfgs/zoe_models/pointpillar.yaml'
        args.ckpt = '/home/khushdeep/Desktop/OpenPCDet/tools/ckpts/kitti/pointpillar_7728.pth'

        cfg_from_yaml_file(args.cfg_file, cfg)
        logger = common_utils.create_logger()
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(args.data_path), logger=logger
        )

        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        self.model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        self.model.cuda()
        self.model.eval()
        self.count = 1

    def predict(self, pointclouds):
        with torch.no_grad():
            input_data = self.demo_dataset.__getitem__(pointclouds)
            input_data = self.demo_dataset.collate_batch([input_data])
            load_data_to_gpu(input_data)
            pred_dicts, _ = self.model.forward(input_data)
            index_vehicle = torch.where(pred_dicts[0]['pred_labels'] == 1)[0].cpu().tolist()
            # V.draw_scenes(
            #     points=input_data['points'][:, 1:],filename = str(self.count),ref_boxes=pred_dicts[0]['pred_boxes'][index_vehicle, :],
            #     ref_scores=pred_dicts[0]['pred_scores'][index_vehicle],
            #     ref_labels=pred_dicts[0]['pred_labels'][index_vehicle]
            # )
            print(self.count)
            self.count = self.count + 1

        return pred_dicts




if __name__ == "__main__":

    second_ros = SecondROS()
