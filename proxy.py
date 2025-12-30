#!/usr/bin/env python3
"""
Proxy端：維護token隊列，處理client請求，與server通信
"""

import socket
import threading
import queue
import time
import json
import sys
import os
import numpy as np
from collections import deque



class MotionProxy:
    def __init__(self, config):
        self.config = config
        
        # 隊列和緩存
        self.token_queue = deque()
        self.dict_queue = deque()
        # 用於生成時回溯的token滑動窗口（保存已被client讀取的最近token）
        keep_for_generate = self.config.get('keep_tokens_for_generate', 0)
        self.generate_token_window = deque(maxlen=keep_for_generate*2 if keep_for_generate > 0 else None)
        self.server_socket = None
        self.connected = False
        
        # 線程控制
        self.running = False
        self.input_thread = None
        self.server_thread = None
        
        # 線程鎖
        self.lock = threading.Lock()
        
        # 狀態標誌
        self.ready_flag = False
        self.generation_started = False
        self.need_interpolation = False  # 標誌是否需要在新token之間插值
        
        # 消息緩衝區 - 處理TCP流式數據
        self.message_buffer = ""
        self.message_delimiter = '\n'  # 使用換行符作為消息分隔符
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.latency_log_path = os.path.join(base_dir, "proxy_time_hot.txt")
        self.read_request_times = deque()

    def _log_latency(self, latency_ms: float):
        """將延遲寫入文件"""
        try:
            with open(self.latency_log_path, 'a', encoding='utf-8') as f:
                f.write(f"{latency_ms:.3f}\n")
        except Exception as exc:
            print(f"Failed to log latency: {exc}")
    
    def _compute_dof_vel(self, current_dof_pos, prev_dof_pos):
        """計算dof_vel，參考server中的方法：dof_vel = (dict_item - prev_dof_pos) * 50"""
        current = np.array(current_dof_pos)
        prev = np.array(prev_dof_pos)
        dof_vel = (current - prev) * 50
        return dof_vel.tolist()
    
    def _interpolate_between_dicts(self, dict_start, dict_end, num_points=20):
        """
        在兩個dict的dof_pos之間進行線性插值
        參考server中的方法計算dof_vel
        
        Args:
            dict_start: 起始dict（原本的最後一個dict），包含'dof_pos'和'dof_vel'
            dict_end: 結束dict（新接收的第一個dict），包含'dof_pos'和'dof_vel'
            num_points: 插值點數量，默認20
        
        Returns:
            (interpolated_dicts, updated_dict_end): 
            - interpolated_dicts: 插值後的dict列表，每個dict的token設為0
            - updated_dict_end: 更新了dof_vel的dict_end（基於最後一個插值點計算）
        """
        
        dof_pos_start = np.array(dict_start['dof_pos'])
        dof_pos_end = np.array(dict_end['dof_pos'])
        
        # 生成插值點
        interpolated_dicts = []
        # 第一個插值點的前一個位置是dict_start的dof_pos
        prev_dof_pos = dof_pos_start
        
        for i in range(1, num_points + 1):  # 從1到num_points，不包括起點和終點
            # 線性插值：在start和end之間均勻分布
            alpha = i / (num_points + 1)  # 0到1之間的插值係數
            interpolated_dof_pos = dof_pos_start * (1 - alpha) + dof_pos_end * alpha
            
            # 計算dof_vel（參考server中的方法：dof_vel = (current - prev) * 50）
            dof_vel = (interpolated_dof_pos - prev_dof_pos) * 50
            
            # 創建插值dict，token設為0（會在調用處設置）
            interp_dict = {
                'dof_pos': interpolated_dof_pos.tolist(),
                'dof_vel': dof_vel.tolist()
            }
            interpolated_dicts.append(interp_dict)
            
            # 更新前一個位置用於下一次速度計算
            prev_dof_pos = interpolated_dof_pos
        
        # 計算最後一個插值點和dict_end之間的dof_vel，更新dict_end
        last_interp_dof_pos = prev_dof_pos  # 最後一個插值點的dof_pos
        dof_vel_end = (dof_pos_end - last_interp_dof_pos) * 50
        
        # 創建更新後的dict_end
        updated_dict_end = dict_end.copy()
        updated_dict_end['dof_vel'] = dof_vel_end.tolist()
        return interpolated_dicts, updated_dict_end
        
    def connect(self):
        """連接到server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.connect((self.config['server_host'], self.config['server_port']))
            self.connected = True
            print(f"Connected to server at {self.config['server_host']}:{self.config['server_port']}")
            
            # 啟動處理線程
            self.running = True
            self.server_thread = threading.Thread(target=self.handle_server_responses)
            self.server_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False
    
    def parse_messages(self, data):
        """解析接收到的數據，處理可能合併的JSON消息"""
        messages = []
        
        # 將新數據添加到緩衝區
        decoded_data = data.decode('utf-8')
        #print(f"Proxy received raw data: {len(decoded_data)} characters")
        self.message_buffer += decoded_data
        
        # 按分隔符分割消息
        while self.message_delimiter in self.message_buffer:
            # 找到第一個分隔符的位置
            delimiter_pos = self.message_buffer.find(self.message_delimiter)
            
            # 提取完整的消息
            message_str = self.message_buffer[:delimiter_pos].strip()
            
            # 從緩衝區中移除已處理的消息
            self.message_buffer = self.message_buffer[delimiter_pos + 1:]
            
            # 解析JSON消息
            if message_str:
                try:
                    message = json.loads(message_str)
                    messages.append(message)
                    #print(f"Successfully parsed message type: {message.get('type')}")
                except json.JSONDecodeError as e:
                    print(f"JSON解析錯誤: {e}")
                    print(f"問題消息 (前100字符): {message_str[:100]}")
                    # 嘗試處理可能的合併消息 {}{}
                    self.handle_merged_messages(message_str, messages)
        
        return messages
    
    def handle_merged_messages(self, message_str, messages):
        """處理可能合併的JSON消息，如{}{}"""
        # 嘗試找到獨立的JSON對象
        brace_count = 0
        current_message = ""
        
        for char in message_str:
            current_message += char
            
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                
                # 當括號平衡時，表示一個完整的JSON對象
                if brace_count == 0:
                    try:
                        message = json.loads(current_message)
                        messages.append(message)
                        current_message = ""
                    except json.JSONDecodeError:
                        # 如果解析失敗，繼續嘗試
                        pass
            
    def ready(self):
        """檢查是否準備就緒"""
        with self.lock:
            # 檢查連接狀態和隊列中的token數量
            ready_threshold = self.config.get('ready_threshold', 5)
            queue_size = len(self.token_queue)
            #print(f"Queue size: {queue_size}")
            is_ready = self.connected and queue_size >= ready_threshold
            return is_ready
        
    def get_motion_state(self):
        """從隊列中獲取一個token"""
        with self.lock:
            if self.dict_queue:
                # 先把即將被client讀取的token放入滑動窗口，用於之後生成時回溯
                token = self.token_queue[0]
                if token == 0:
                    print("插值")
                if self.generate_token_window is not None:
                    self.generate_token_window.append(token)
                self.token_queue.popleft()
                return self.dict_queue.popleft()  # 返回dict
            return None
            
    def start_input_listener(self):
        """啟動輸入監聽線程"""
        self.input_thread = threading.Thread(target=self.handle_input)
        self.input_thread.daemon = True
        self.input_thread.start()
        
    def handle_input(self):
        """處理命令行輸入"""
        print("Input listener started. Type commands:")
        print("- 'start <prompt>' to start generation")
        print("- 'stop' to stop generation")
        print("- 'quit' to exit")
        
        while self.running:
            try:
                user_input = input().strip()
                
                if user_input.lower() == 'quit':
                    self.running = False
                    break
                    
                elif user_input.lower() == 'stop':
                    self.send_stop_command()
                    
                elif user_input.startswith('start '):
                    prompt = user_input[6:].strip()
                    if prompt:
                        self.send_start_command(prompt)
                    else:
                        print("Please provide a prompt")
                        
                else:
                    print("Unknown command")
                    
            except EOFError:
                break
            except Exception as e:
                print(f"Input error: {e}")
                
    def send_start_command(self, prompt):
        print(prompt)
        """發送開始生成命令"""
        with self.lock:
            # 保留前keep_tokens_on_new_instruction個tokens
            keep_count = self.config['keep_tokens_on_new_instruction']
            if len(self.token_queue) > keep_count:
                # 只保留前keep_count個
                new_token_queue = deque(list(self.token_queue)[:keep_count])
                self.token_queue = new_token_queue
                #print(f"self.token_queue: {self.token_queue}")
                #print(f"new_token_queue: {new_token_queue}")
                new_dict_queue = deque(list(self.dict_queue)[:keep_count])
                self.dict_queue = new_dict_queue
            else:
                #print(f"self.token_queue: {self.token_queue}")
                # 如果隊列長度不足，保留全部
                pass
                
            # 準備motion tokens（保留的最後keep_tokens_for_generate個）,注意发的是token而不是dict
            motion_tokens = []
            if self.token_queue:
                keep_for_generate = self.config['keep_tokens_for_generate']
                if keep_for_generate > 0:
                    current_tokens = list(self.token_queue)
                    current_len = len(current_tokens)
                    if keep_for_generate*2 <= current_len:
                        # 隊列裡的token夠用，取最後 keep_for_generate*2 個中序號為偶的
                        last_tokens = current_tokens[-(keep_for_generate*2):]
                        motion_tokens = last_tokens[::2]  # 取偶數索引（0, 2, 4, ...）
                        #print(f"motion_tokens: {motion_tokens}")
                        #print("--------------------------------")
                    else:
                        # 不夠用，從已讀取的token滑動窗口中補前面的部分
                        need_from_history = keep_for_generate*2 - current_len
                        history_part = []
                        if self.generate_token_window is not None and len(self.generate_token_window) > 0:
                            history_tokens = list(self.generate_token_window)
                            need_from_history = min(need_from_history, len(history_tokens))
                            if need_from_history > 0:
                                history_part = history_tokens[-need_from_history:]
                                history_part = history_part[::2]  # 取偶數索引（0, 2, 4, ...）
                        # 先放歷史token，再放當前隊列中的token
                        motion_tokens = history_part + current_tokens[::2]
                        print(f"motion_tokens: {motion_tokens}")
                        print("--------------------------------")
                else:
                    motion_tokens = []
                
            # 發送命令
            command = {
                'type': 'start_generation',
                'prompt': prompt,
                'motion_tokens': motion_tokens
            }
            
            self.send_to_server(command)
            self.generation_started = True
            self.need_interpolation = True  # 標記需要在新token之間插值
            print(f"Started generation with prompt: {prompt[:50]}...")
            
            # 啟動token請求線程
            if not hasattr(self, 'token_requester_started') or not self.token_requester_started:
                self.start_token_requester()
                self.token_requester_started = True
            
    def send_stop_command(self):
        """發送停止命令"""
        command = {'type': 'stop'}
        self.send_to_server(command)
        self.generation_started = False
        print("Stopped generation")
        
    def send_to_server(self, message):
        """發送消息到server"""
        try:
            data = json.dumps(message) + self.message_delimiter
            self.server_socket.send(data.encode('utf-8'))
        except Exception as e:
            print(f"Error sending to server: {e}")
            
    def handle_server_responses(self):
        """處理server響應"""
        while self.running and self.connected:
            try:
                data = self.server_socket.recv(4096)
                if not data:
                    print("Server disconnected")
                    break
                
                # 解析消息（可能包含多個合併的消息）
                messages = self.parse_messages(data)
                #print(f"Proxy received {len(messages)} message(s)")
                # 處理每個消息
                for response in messages:
                    #print(f"Processing response type: {response.get('type')}")
                    if response['type'] == 'tokens_response':
                        # 收到tokens，添加到隊列
                        tokens_list = response['tokens']  # tokens_list 是一個列表，每個元素是 {'token': ..., 'dict': ...}
                        
                        # if tokens_list:
                        #     # 記錄接收到token響應的時間戳並計算延遲（僅在有有效token時）
                        #     receive_time = time.time()
                        #     send_time = None
                        #     with self.lock:
                        #         if self.read_request_times:
                        #             send_time = self.read_request_times.popleft()
                        #     if send_time is not None:
                        #         latency = (receive_time - send_time) * 1000.0
                        #         self._log_latency(latency)
                        # else:
                        #     # 若讀取到空token列表，仍需彈出對應的請求時間以維持隊列同步
                        #     with self.lock:
                        #         if self.read_request_times:
                        #             self.read_request_times.popleft()
                        
                        #print(f"Received {len(tokens_list)} tokens")
                        with self.lock:
                            # 追蹤是否進行了插值，以決定後續是否需要跳過第一個dict
                            interpolation_done = False
                            
                            # 如果需要插值且收到了新token
                            if self.need_interpolation and tokens_list:
                                # 獲取原本隊列中最後一個dict
                                if len(self.dict_queue) > 0:
                                    last_dict = self.dict_queue[-1]
                                    # 獲取新接收的第一個token的dict
                                    first_new_dict = tokens_list[0]['dict']
                                    
                                    # 執行插值，生成20個插值dict和更新後的dict_end
                                    interpolated_dicts, updated_dict_end = self._interpolate_between_dicts(
                                        last_dict, first_new_dict, num_points=20
                                    )
                                    
                                    # 將20個插值結果插入到隊列中（在第一個新token之前）
                                    # 對應的token設為0
                                    for interp_dict in interpolated_dicts:
                                        self.token_queue.append(0)
                                        self.dict_queue.append(interp_dict)
                                    
                                    # 將更新了dof_vel的dict_end也插入到隊列中
                                    # 對應的token使用第一個新token（從tokens_list[0]獲取）
                                    first_token = tokens_list[0]['token']
                                    self.token_queue.append(first_token)
                                    self.dict_queue.append(updated_dict_end)
                                    
                                    print(f"插值完成：在最後一個dict和新第一個dict之間插入了20個dict，並插入了更新後的dict_end（使用第一個新token）")
                                    # 標記插值已完成
                                    interpolation_done = True
                                    self.need_interpolation = False
                                else:
                                    # 隊列為空，無法進行插值
                                    print(f"隊列為空，跳過插值")
                                    # 標記插值已完成（因為無法插值）
                                    self.need_interpolation = False
                            
                            # 遍歷列表，提取每個token和dict
                            for idx, token_dict in enumerate(tokens_list):
                                if interpolation_done and idx == 0:
                                    # 如果進行了插值，跳過第一個token和dict（因為已經用更新後的dict_end和第一個token替代了）
                                    continue
                                # 其他token和dict正常插入
                                self.token_queue.append(token_dict['token'])
                                self.dict_queue.append(token_dict['dict'])
                            # 檢查是否達到ready閾值
                            ready_threshold = self.config.get('ready_threshold', 5)
                            if not self.ready_flag and len(self.token_queue) >= ready_threshold:
                                self.ready_flag = True
                                #print(f"Proxy ready for client requests (queue size: {len(self.token_queue)})")
                            #print(f"Queue sizes - tokens: {len(self.token_queue)}, dicts: {len(self.dict_queue)}")
                            
            except Exception as e:
                print(f"Error handling server response: {e}")
                import traceback
                traceback.print_exc()
                break
                
    def request_tokens_from_server(self):
        """從server請求tokens"""
        if not self.connected or not self.generation_started:
            return
            
        # 檢查是否需要請求更多tokens
        with self.lock:
            queue_size = len(self.token_queue)
            
        if queue_size <= self.config['buffer_threshold']:
            # 請求tokens
            count = self.config['read_batch_size']
            
            # 記錄發送read_tokens指令的時間戳
            send_time = time.time()
            
            command = {
                'type': 'read_tokens',
                'count': count,
                'send_timestamp': send_time  # 添加時間戳
            }
            self.send_to_server(command)
            with self.lock:
                self.read_request_times.append(send_time)
            #read_time = time.time()
            #print(f"[時間測試] Proxy發送read_tokens指令時間: {send_time * 1000:.3f} ms")
            
    def start_token_requester(self):
        """啟動token請求線程"""
        if hasattr(self, 'token_requester_started') and self.token_requester_started:
            return  # 已經啟動過了
            
        def token_requester():
            while self.running and self.connected:
                self.request_tokens_from_server()
                time.sleep(0.1)  # 每100ms檢查一次
                
        requester_thread = threading.Thread(target=token_requester)
        requester_thread.daemon = True
        requester_thread.start()
        self.token_requester_started = True
        
    def cleanup(self):
        """清理資源"""
        self.running = False
        
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join()
            
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join()
            
        if self.server_socket:
            self.server_socket.close()
            
        print("Proxy cleaned up")

def main():
    # 默認配置
    config = {
        'server_host': 'localhost',
        'server_port': 8000,
        'buffer_threshold': 10,
        'ready_threshold': 5,  # 隊列中token數量達到此值時ready為true
        'keep_tokens_on_new_instruction': 5,
        'keep_tokens_for_generate': 3,
        'read_batch_size': 20
    }
    
    # 創建proxy
    proxy = MotionProxy(config)
    
    try:
        # 連接到server
        if not proxy.connect():
            return
            
        # 啟動輸入監聽
        proxy.start_input_listener()
        
        # 啟動token請求線程
        proxy.start_token_requester()
        
        # 保持運行
        while proxy.running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nShutting down proxy...")
    except Exception as e:
        print(f"Proxy error: {e}")
    finally:
        proxy.cleanup()

if __name__ == '__main__':
    main()
