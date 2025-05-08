#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import argparse
import sys
import threading
import time
import yaml
from collections import deque

import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PImage
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header
import cv2

from scripts.agilex_model import create_model

# sys.path.append("./")

CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']

observation_window = None

lang_embeddings = None

# debug
preload_images = None


# -- coding: UTF-8
import os
import numpy as np
import torch
import torchvision
import argparse
import dm_env
import requests
import collections
from collections import deque
import logging
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from base64 import b64decode
import threading
import cv2
import signal
import subprocess
import multiprocessing
from tqdm import tqdm

start_flag = False
exit_flag = False
num_fails = 0
logger = logging.getLogger(__name__)




# Initialize the model
def make_policy(args):
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    args.config = config
    
    # pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=args.config, 
        dtype=torch.bfloat16,
        pretrained=args.pretrained_model_name_or_path,
        # pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=args.ctrl_freq,
    )

    return model


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# Interpolate the actions to make the robot move smoothly
def interpolate_action(args, prev_action, cur_action):
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


def get_config(args):
    config = {
        'episode_len': args.max_publish_step,
        'state_dim': 14,
        'chunk_size': args.chunk_size,
        'camera_names': CAMERA_NAMES,
    }
    return config


# Get the observation from the ROS topic
def get_ros_observation(args,ros_operator):
    rate = rospy.Rate(args.publish_rate)
    print_flag = True

    while True and not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail when get_ros_observation")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, robot_base) = result
        # print(f"sync success when get_ros_observation")
        return (img_front, img_left, img_right,
         puppet_arm_left, puppet_arm_right)


# Update the observation window buffer
def update_observation_window(args, config, ros_operator):
    # JPEG transformation
    # Align with training
    def jpeg_mapping(img):
        img = cv2.imencode('.jpg', img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img
    
    global observation_window
    if observation_window is None:
        observation_window = deque(maxlen=2)
    
        # Append the first dummy image
        observation_window.append(
            {
                'qpos': None,
                'images':
                    {
                        config["camera_names"][0]: None,
                        config["camera_names"][1]: None,
                        config["camera_names"][2]: None,
                    },
            }
        )
    
    result = ros_operator.get_frame(ignore_master=True)
    img_front, img_left, img_right, img_high, img_side, puppet_arm_left, puppet_arm_right, _, _, robot_base = result
    # img_front, img_left, img_right, puppet_arm_left, puppet_arm_right = get_ros_observation(args,ros_operator)
    img_front = jpeg_mapping(img_front)
    img_left = jpeg_mapping(img_left)
    img_right = jpeg_mapping(img_right)
    
    qpos = np.concatenate(
            (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
    qpos = torch.from_numpy(qpos).float().cuda()
    observation_window.append(
        {
            'qpos': qpos,
            'images':
                {
                    config["camera_names"][0]: img_front,
                    config["camera_names"][1]: img_right,
                    config["camera_names"][2]: img_left,
                },
        }
    )


# RDT inference
def inference_fn(args, config, policy, t):
    global observation_window
    global lang_embeddings
    
    # print(f"Start inference_thread_fn: t={t}")
    while True and not rospy.is_shutdown():
        time1 = time.time()     

        # fetch images in sequence [front, right, left]
        image_arrs = [
            observation_window[-2]['images'][config['camera_names'][0]],
            observation_window[-2]['images'][config['camera_names'][1]],
            observation_window[-2]['images'][config['camera_names'][2]],
            
            observation_window[-1]['images'][config['camera_names'][0]],
            observation_window[-1]['images'][config['camera_names'][1]],
            observation_window[-1]['images'][config['camera_names'][2]]
        ]
        
        # fetch debug images in sequence [front, right, left]
        # image_arrs = [
        #     preload_images[config['camera_names'][0]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][2]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][1]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][0]][t],
        #     preload_images[config['camera_names'][2]][t],
        #     preload_images[config['camera_names'][1]][t]
        # ]
        # # encode the images
        # for i in range(len(image_arrs)):
        #     image_arrs[i] = cv2.imdecode(np.frombuffer(image_arrs[i], np.uint8), cv2.IMREAD_COLOR)
        # proprio = torch.from_numpy(preload_images['qpos'][t]).float().cuda()
        
        images = [PImage.fromarray(arr) if arr is not None else None
                  for arr in image_arrs]
        
        # for i, pos in enumerate(['f', 'r', 'l'] * 2):
        #     images[i].save(f'{t}-{i}-{pos}.png')
        
        # get last qpos in shape [14, ]
        proprio = observation_window[-1]['qpos']
        # unsqueeze to [1, 14]
        proprio = proprio.unsqueeze(0)
        
        # actions shaped as [1, 64, 14] in format [left, right]
        actions = policy.step(
            proprio=proprio,
            images=images,
            text_embeds=lang_embeddings 
        ).squeeze(0).cpu().numpy()
        # print(f"inference_actions: {actions.squeeze()}")
        
        print(f"Model inference time: {time.time() - time1} s")
        
        # print(f"Finish inference_thread_fn: t={t}")
        return actions


# Main loop for the manipulation task
def model_inference(args, config, ros_operator):
    global lang_embeddings
    
    # Load rdt model
    policy = make_policy(args)
    
    lang_dict = torch.load(args.lang_embeddings_path)
    print(f"Running with instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
    lang_embeddings = lang_dict["embeddings"]
    
    max_publish_step = config['episode_len']
    chunk_size = config['chunk_size']

    # Initialize position of the puppet arm
    left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
    right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
    left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
    right1 = [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]
    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input("Press enter to continue")
    ros_operator.puppet_arm_publish_continuous(left1, right1)
    # Initialize the previous action to be the initial robot state
    pre_action = np.zeros(config['state_dim'])
    pre_action[:14] = np.array(
        [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258] + 
        [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]
    )
    action = None
    # Inference loop
    with torch.inference_mode():
        while True and not rospy.is_shutdown():
            # The current time step
            t = 0
            rate = rospy.Rate(args.publish_rate)
    
            action_buffer = np.zeros([chunk_size, config['state_dim']])
            
            while t < max_publish_step and not rospy.is_shutdown():
                # Update observation window
                update_observation_window(args, config, ros_operator)
                
                # When coming to the end of the action chunk
                if t % chunk_size == 0:
                    # Start inference
                    # here inference
                    action_buffer = inference_fn(args, config, policy, t).copy()
                
                raw_action = action_buffer[t % chunk_size]
                action = raw_action
                # Interpolate the original action sequence
                if args.use_actions_interpolation:
                    # print(f"Time {t}, pre {pre_action}, act {action}")
                    interp_actions = interpolate_action(args, pre_action, action)
                else:
                    interp_actions = action[np.newaxis, :]
                # Execute the interpolated actions one by one
                for act in interp_actions:
                    left_action = act[:7]
                    right_action = act[7:14]
                    
                    if not args.disable_puppet_arm:
                        ros_operator.puppet_arm_publish(left_action, right_action)  # puppet_arm_publish_continuous_thread
                
                    if args.use_robot_base:
                        vel_action = act[14:16]
                        ros_operator.robot_base_publish(vel_action)
                    rate.sleep()
                    # print(f"doing action: {act}")
                t += 1
                
                print("Published Step", t)
                pre_action = action.copy()


# ROS operator class
# ros operator
class RosOperator:
    def __init__(self, infer_args, actions):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.master_arm_right_deque = None
        self.master_arm_left_deque = None
        self.img_front_deque = None
        self.bridge = None
        self.infer_args = infer_args
        self.init()
        self.init_ros()
        self.actions = actions
        self.current_action_idx = 0

    def init(self):
        self.bridge = CvBridge()
        self.img_front_deque = deque()
        self.master_arm_left_deque = deque()
        self.master_arm_right_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.master_arm_publish_lock = threading.Lock()
        self.master_arm_publish_lock.acquire()
        self.master_arm_publish_thread = None

    def master_arm_publish_actions_thread(self):
        if self.master_arm_publish_thread is not None:
            self.master_arm_publish_lock.release()
            self.master_arm_publish_thread.join()
            self.master_arm_publish_lock.acquire(False)
            self.master_arm_publish_thread = None
        self.master_arm_publish_thread = threading.Thread(target=self.master_arm_publish_actions)
        self.master_arm_publish_thread.start()

    def master_arm_publish_actions(self):
        global exit_flag
        self.current_action_idx = 0
        while True and not rospy.is_shutdown() and not exit_flag and self.current_action_idx < len(self.actions):
            left_qpos = self.actions[self.current_action_idx][:7]
            right_qpos = self.actions[self.current_action_idx][7:]
            left, right = left_qpos, right_qpos
            self.master_arm_publish_continuous(left, right)
            self.current_action_idx += 1
        exit_flag = True

    def master_arm_publish(self, left, right):
        rate = rospy.Rate(self.infer_args.publish_rate)
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
        joint_state_msg.position = left
        self.master_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.master_arm_right_publisher.publish(joint_state_msg)
        rate.sleep()

    def master_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.infer_args.publish_rate)
        left_arm = None
        right_arm = None
        while True and not rospy.is_shutdown():
            # breakpoint()
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.master_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.infer_args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.infer_args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.infer_args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.infer_args.arm_steps_length[i]
                    flag = True
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = left_arm
            self.master_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.master_arm_right_publisher.publish(joint_state_msg)
            step += 1
            rate.sleep()

    def get_cam_frame(self, camera='high'):
        match camera:
            case "high":
                port = 23000
            case "left":
                port = 23001
            case "right":
                port = 23002
            case "side":
                port = 23003
        frame = requests.get(f'http://localhost:{port}').json()
        frame['image'] = np.frombuffer(b64decode(frame['image']), dtype=np.uint8).reshape(480, 640, 3)
        frame['depth'] = np.frombuffer(b64decode(frame['depth']), dtype=np.uint16).reshape(480, 640)
        return frame['image'], frame['depth'], frame['timestamp']

    def get_frame(self, ignore_master=False):
        if self.infer_args.use_image_front:
            if len(self.img_front_deque) == 0:
                return False, "image front fail"
        frame_times = []
        if self.infer_args.use_image_front:
            frame_times.append(self.img_front_deque[-1].header.stamp.to_sec())

        img_high, img_left, img_right = None, None, None
        img_high_timestamp, img_left_timestamp, img_right_timestamp = None, None, None
        if self.infer_args.use_image_high:
            img_high, _, img_high_timestamp = self.get_cam_frame('high')
            frame_times.append(img_high_timestamp)
        if self.infer_args.use_image_left:
            img_left, _, img_left_timestamp = self.get_cam_frame('left')
            frame_times.append(img_left_timestamp)
        if self.infer_args.use_image_right:
            img_right, _, img_right_timestamp = self.get_cam_frame('right')
            frame_times.append(img_right_timestamp)
        if self.infer_args.use_image_side:
            img_side, _, img_side_timestamp = self.get_cam_frame('side')
            frame_times.append(img_side_timestamp)

        if len(frame_times) == 0:
            frame_time = -1
        else:
            frame_time = min(frame_times)

        if self.infer_args.use_image_front:
            if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
                return False, "image front fail"
        if self.infer_args.use_master_left and not ignore_master:
            if len(self.master_arm_left_deque) == 0 or self.master_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
                return False, "master left fail"
        if self.infer_args.use_master_right and not ignore_master:
            if len(self.master_arm_right_deque) == 0 or self.master_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
                return False, "master right fail"
        if self.infer_args.use_puppet_left:
            if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
                return False, "puppet left fail"
        if self.infer_args.use_puppet_right:
            if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
                return False, "puppet right fail"
        if self.infer_args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False, "robot base fail"

        img_front = None
        if self.infer_args.use_image_front:
            while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_deque.popleft()
            img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')[:, :, [2, 1, 0]]

        if self.infer_args.use_image_high:
            while img_high_timestamp < frame_time:
                img_high, _, img_high_timestamp = self.get_cam_frame('high')

        if self.infer_args.use_image_left:
            while img_left_timestamp < frame_time:
                img_left, _, img_left_timestamp = self.get_cam_frame('left')

        if self.infer_args.use_image_right:
            while img_right_timestamp < frame_time:
                img_right, _, img_right_timestamp = self.get_cam_frame('right')

        if self.infer_args.use_image_side:
            while img_side_timestamp < frame_time:
                img_side, _, img_side_timestamp = self.get_cam_frame('side')

        master_arm_left = None
        if self.infer_args.use_master_left and not ignore_master:
            while self.master_arm_left_deque[0].header.stamp.to_sec() < frame_time:
                self.master_arm_left_deque.popleft()
            master_arm_left = self.master_arm_left_deque.popleft()

        master_arm_right = None
        if self.infer_args.use_master_right and not ignore_master:
            while self.master_arm_right_deque[0].header.stamp.to_sec() < frame_time:
                self.master_arm_right_deque.popleft()
            master_arm_right = self.master_arm_right_deque.popleft()

        puppet_arm_left = None
        if self.infer_args.use_puppet_left:
            while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
                self.puppet_arm_left_deque.popleft()
            puppet_arm_left = self.puppet_arm_left_deque.popleft()

        puppet_arm_right = None
        if self.infer_args.use_puppet_right:
            while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
                self.puppet_arm_right_deque.popleft()
            puppet_arm_right = self.puppet_arm_right_deque.popleft()

        robot_base = None
        if self.infer_args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return img_front, img_left, img_right, img_high, img_side, puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, robot_base

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def master_arm_left_callback(self, msg):
        if len(self.master_arm_left_deque) >= 2000:
            self.master_arm_left_deque.popleft()
        self.master_arm_left_deque.append(msg)

    def master_arm_right_callback(self, msg):
        if len(self.master_arm_right_deque) >= 2000:
            self.master_arm_right_deque.popleft()
        self.master_arm_right_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def init_ros(self):
        rospy.init_node('record_episodes', anonymous=True)
        rospy.Subscriber(self.infer_args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.infer_args.use_master_left:
            rospy.Subscriber(self.infer_args.master_arm_left_topic, JointState, self.master_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        if self.infer_args.use_master_right:
            rospy.Subscriber(self.infer_args.master_arm_right_topic, JointState, self.master_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        if self.infer_args.use_puppet_left:
            rospy.Subscriber(self.infer_args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        if self.infer_args.use_puppet_right:
            rospy.Subscriber(self.infer_args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.infer_args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        self.master_arm_left_publisher = rospy.Publisher(self.infer_args.master_arm_left_cmd_topic, JointState, queue_size=10)
        self.master_arm_right_publisher = rospy.Publisher(self.infer_args.master_arm_right_cmd_topic, JointState, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(self.infer_args.robot_base_cmd_topic, Twist, queue_size=10)

    def process(self):
        self.master_arm_publish_continuous(self.actions[0][:7], self.actions[0][7:])
        timesteps = []
        # 图像数据
        image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        image_dict = dict()
        for cam_name in self.infer_args.camera_names:
            image_dict[cam_name] = image
        count = 0

        print_flag = True
        rate = rospy.Rate(self.infer_args.frame_rate)
        global exit_flag
        global start_flag
        global num_fails
        while not rospy.is_shutdown():
            if exit_flag:
                break
            # 2 收集数据
            result = self.get_frame(ignore_master=True)
            if result[0] is False:
                num_fails += 1
                if print_flag:
                    logger.info(result[1])
                rate.sleep()
                continue
            count += 1
            img_front, img_left, img_right, img_high, img_side, puppet_arm_left, puppet_arm_right, _, _, robot_base = result
            if start_flag is False:
                if np.allclose(np.array(puppet_arm_left.position), self.actions[0][:7], rtol=1, atol=0.12) and np.allclose(np.array(puppet_arm_right.position), self.actions[0][7:], rtol=1, atol=0.12):
                    start_flag = True
                    self.master_arm_publish_actions_thread()
                else:
                    logger.error("Reset failed")
                    exit(0)
                continue
            # 2.1 图像信息
            image_dict = dict()
            image_dict[self.infer_args.camera_names[0]] = img_front
            image_dict[self.infer_args.camera_names[1]] = img_left
            image_dict[self.infer_args.camera_names[2]] = img_right
            image_dict[self.infer_args.camera_names[3]] = img_high
            image_dict[self.infer_args.camera_names[4]] = img_side

            # 2.2 从臂的信息从臂的状态 机械臂示教模式时 会自动订阅
            obs = collections.OrderedDict()  # 有序的字典
            obs['images'] = image_dict
            obs['qpos'] = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
            obs['qvel'] = np.concatenate((np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
            obs['effort'] = np.concatenate((np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
            if self.infer_args.use_robot_base:
                obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            else:
                obs['base_vel'] = [0.0, 0.0]

            # 第一帧 只包含first， fisrt只保存StepType.FIRST
            if count == 1:
                ts = dm_env.TimeStep(
                    step_type=dm_env.StepType.FIRST,
                    reward=None,
                    discount=None,
                    observation=obs)
                timesteps.append(ts)
                continue

            # 时间步
            ts = dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=None,
                discount=None,
                observation=obs)

            timesteps.append(ts)
            if rospy.is_shutdown():
                exit(-1)
            rate.sleep()
        exit_flag = True
        if self.master_arm_publish_thread is not None:
            self.master_arm_publish_lock.release()
            self.master_arm_publish_thread.join()
        self.master_arm_publish_continuous([0] * 6 + [-0.1350], [0] * 6 + [-0.1350])
        logger.info(f'len(timesteps): {len(timesteps)}, num_fails: {num_fails}')
        return timesteps


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_publish_step', action='store', type=int, 
                        help='Maximum number of action publishing steps', default=10000, required=False)
    parser.add_argument('--seed', action='store', type=int, 
                        help='Random seed', default=None, required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store_true', 
                        help='Whether to use the robot base to move around',
                        default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, 
                        help='The rate at which to publish the actions',
                        default=30, required=False)
    parser.add_argument('--ctrl_freq', action='store', type=int, 
                        help='The control frequency of the robot',
                        default=25, required=False)
    
    parser.add_argument('--chunk_size', action='store', type=int, 
                        help='Action chunk size',
                        default=64, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, 
                        help='The maximum change allowed for each joint per timestep',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--use_actions_interpolation', action='store_true',
                        help='Whether to interpolate the actions if the difference is too large',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store_true', 
                        help='Whether to use depth images',
                        default=False, required=False)
    
    parser.add_argument('--disable_puppet_arm', action='store_true',
                        help='Whether to disable the puppet arm. This is useful for safely debugging',default=False)
    
    parser.add_argument('--config_path', type=str, default="configs/base.yaml", 
                        help='Path to the config file')
    # parser.add_argument('--cfg_scale', type=float, default=2.0,
    #                     help='the scaling factor used to modify the magnitude of the control features during denoising')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, help='Name or path to the pretrained model')
    
    parser.add_argument('--lang_embeddings_path', type=str, required=True, 
                        help='Path to the pre-encoded language instruction embeddings')
    
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    if args.seed is not None:
        set_seed(args.seed)
    config = get_config(args)
    model_inference(args, config, ros_operator)


if __name__ == '__main__':
    main()
