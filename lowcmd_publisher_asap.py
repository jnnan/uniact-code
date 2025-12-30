import rclpy
from rclpy.node import Node
from std_msgs.msg import String  # 根据你需要的消息类型更改
import time
from utils.crc import CRC


from gamepad import Gamepad, parse_remote_data

import numpy as np
from unitree_hg.msg import (
    LowState,
    MotorState,
    IMUState,
    LowCmd,
    MotorCmd,
)

import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
crc = CRC()

class MessageRelayNode(Node):
    def __init__(self):
        super().__init__('message_relay_node')
        
        # 创建一个订阅者，监听某个输入话题（如 'input_topic'）
        self.subscription = self.create_subscription(
            LowCmd,  # 替换成你的消息类型
            'lowcmd_buffer',  # 替换成你的输入话题名
            self.listener_callback,
            10
        )
        self.lowlevel_state_sub = self.create_subscription(LowState, "lowstate", self.lowlevel_state_cb, 1)  # "/lowcmd" or  "lf/lowstate" (low frequencies)
        self.lowlevel_state_sub  # prevent unused variable warning

        # 创建一个发布者，向 'lowcmd' 话题发布消息
        self.publisher = self.create_publisher(LowCmd, 'lowcmd', 10)
        
        # 初始化一个接收到的消息变量
        self.last_msg = None
        self.last_last_msg = None
        self.counter = 0

        # init motor command
        self.new_msg = LowCmd()
        self.new_msg.mode_pr = 0
        self.new_msg.mode_machine = 5 # 6-26dof
        self.motor_cmd = []
        for id in range(29):
            cmd=MotorCmd(q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, mode=1, reserve=0)
            self.motor_cmd.append(cmd)
        for id in range(29, 35):
            cmd=MotorCmd(q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, mode=0, reserve=0)
            self.motor_cmd.append(cmd)
        self.new_msg.motor_cmd = self.motor_cmd.copy()

        
        self.gamepad = Gamepad()
        self.Emergency_stop = False

        ChannelFactoryInitialize(0, 'enp6s0')
        self.audio_client = AudioClient() 
        self.audio_client.SetTimeout(10.0)
        self.audio_client.Init()


        time.sleep(1)
        self.audio_client.LedControl(0,255,0)
        time.sleep(1)
        self.audio_client.LedControl(0,0,255)


    def lowlevel_state_cb(self, msg: LowState):
        # wireless_remote btn
        joystick_data = msg.wireless_remote
        parsed_data = parse_remote_data(joystick_data)
        self.gamepad.update(parsed_data)
        
        if self.gamepad.R2.pressed:
            self.Emergency_stop = True
            print(f'Manual emergency stop!!!')
        if self.gamepad.R1.pressed: # R1 is pressed
            self.get_logger().info("Program exiting")
            self.stop = True
            


    def listener_callback(self, msg):
        # 每当接收到消息时，将其保存下来
        if self.last_msg is None:
            self.counter = 0
            self.last_msg = LowCmd()
            self.last_last_msg = LowCmd()
            self.last_msg.motor_cmd = msg.motor_cmd.copy()
            self.last_last_msg.motor_cmd = msg.motor_cmd.copy()
        else:
            self.counter = 0
            self.last_last_msg.motor_cmd = self.last_msg.motor_cmd.copy()
            self.last_msg.motor_cmd = msg.motor_cmd.copy()
    
    def set_motor_position(
        self,
    ):
        for i in range(29):
            count = np.clip(self.counter, 0, 15)
            # self.motor_cmd[i].q = self.last_msg.motor_cmd[i].q * (count/15) + self.last_last_msg.motor_cmd[i].q * (1-count/15)
            self.motor_cmd[i].q = self.last_msg.motor_cmd[i].q
            self.motor_cmd[i].kp = self.last_msg.motor_cmd[i].kp
            self.motor_cmd[i].kd = self.last_msg.motor_cmd[i].kd
        self.new_msg.motor_cmd = self.motor_cmd.copy()
        # self.cmd_msg.crc = get_crc(self.cmd_msg)
        self.new_msg.crc = crc.Crc(self.new_msg)
        self.counter += 1

    def relay_message(self):
        # 如果已经接收到消息，则每次调用此函数时发布消息到lowcmd话题
        if self.last_msg is not None:
            self.set_motor_position()
            self.publisher.publish(self.new_msg)
            # breakpoint()

            # print("########################################")
            # for i in range(29):
            #     print(self.new_msg.motor_cmd[i].q)
            # print(self.new_msg.motor_cmd[18].q)

def main(args=None):
    rclpy.init(args=args)
    
    # 创建节点实例
    node = MessageRelayNode()
    
    
    try:
        while rclpy.ok():
            loop_start_time = time.monotonic()
            if node.Emergency_stop:
                break
            # 处理订阅回调函数
            rclpy.spin_once(node, timeout_sec=0)
            
            # 转发消息到 lowcmd 话题
            node.relay_message()
            
            while 0.000992-time.monotonic()+loop_start_time>0:  #0.012473  0.019963 # 创建1000Hz的发布频率
                pass
    
    except KeyboardInterrupt:
        pass
    
    # 关闭节点
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
