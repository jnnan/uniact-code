#!/usr/bin/env python3
"""
Server: Runs motion generation model and communicates with proxy via socket
"""

import socket
import threading
from collections import deque
import time
import json
import sys
import os
import torch
import numpy as np
import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Qwen2_5_VLForConditionalGeneration
)
from fsq import FSQ


# Add server directory to path to import infer_robot module
# If infer_robot.py is not in the project root, uncomment and set the path:
# sys.path.append('your_infer_robot_directory_path')

from infer_robot import (
    load_finetuned_model,
    prepare_inference_input_t2m,
    create_motion_position_ids,
    parse_generated_ids,
    encode_motion_tokens,
    unified_generation_step,
    MOTION_TOKEN_CONFIG
)

byd_joint_names = [
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 
    'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint', 
    'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint', 
    'left_knee_joint', 'right_knee_joint', 
    'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 
    'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 
    'left_ankle_roll_joint', 'right_ankle_roll_joint', 
    'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
    'left_elbow_joint', 'right_elbow_joint', 
    'left_wrist_roll_joint', 'right_wrist_roll_joint', 
    'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 
    'left_wrist_yaw_joint', 'right_wrist_yaw_joint'
]

# Joint naming order in MuJoCo simulator
mujoco_joint_names = [
    # Left leg (6 joints)
    'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
    # Right leg (6 joints)
    'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
    # Waist (3 joints)
    'waist_yaw', 'waist_roll', 'waist_pitch',
    # Left arm (7 joints)
    'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow', 
    'left_wrist_roll', 'left_wrist_pitch', 'left_wrist_yaw',
    # Right arm (7 joints)
    'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow', 
    'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw',
]

# Create joint index mapping table
# BYD joint order -> MuJoCo joint order
byd_joint_to_mujoco_joint = [byd_joint_names.index(joint_name+'_joint') for joint_name in mujoco_joint_names]
# MuJoCo joint order -> BYD joint order
mujoco_joint_to_byd_joint = [mujoco_joint_names.index(joint_name[:-6]) for joint_name in byd_joint_names]


def get_local_ip():
    """
    Get the local network IP address of this machine
    
    Determines the actual IP address of this machine by connecting to an external address,
    rather than using the loopback address. This is useful for network servers to display
    the correct connection address.
    
    Returns:
        str: The local network IP address of this machine, or "127.0.0.1" if failed
        
    Method:
        1. Create UDP socket
        2. Connect to external address (8.8.8.8:80)
        3. Get the local address of the socket
        4. Close socket and return IP
        
    Note:
        - Uses UDP connection, does not actually send data
        - Returns loopback address if network is unavailable
        - Works in most network environments
    """
    try:
        # Create UDP socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to external address (does not actually send data)
        s.connect(("8.8.8.8", 80))
        # Get the local address of the socket
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        # If failed, return loopback address
        return "127.0.0.1"

class MotionServer:
    def __init__(self, host='localhost', port=8000):
        self.host = host
        self.port = port
        self.socket = None
        self.client_socket = None
        self.running = False
        
        # Motion generation related
        self.model = None
        self.tokenizer = None
        self.token_cache = deque()  # Store generated tokens
        self.dict_cache = deque()  # Store generated dicts
        self.dict_last = {'dof_pos': [0.0] * 29, 'dof_vel': [0.0] * 29}  # Store the last dict in current queue (using list format for JSON serialization)
        self.generation_thread = None
        self.generation_running = False
        
        # Current generation parameters
        self.current_prompt = None
        self.current_motion_tokens = None
        self.past_key_values = None
        self.step_count = 0
        self.timing_lock = threading.Lock()
        self.token_time_records = []
        self.decode_time_records = []
        self.token_total_time = 0.0
        self.decode_total_time = 0.0
        
        # Thread lock
        self.lock = threading.Lock()
        
        # Message buffer - handle TCP streaming data
        self.message_buffer = ""
        self.message_delimiter = '\n'  # Use newline as message delimiter

        self.decoder = torch.jit.load('your_decoder_file_path.pt')
        self.decoder.eval()
        levels = [8, 8, 8, 6, 5]
        self.quantize = FSQ(levels=levels)

        self.min_vals_tensor = torch.tensor([-1.5348, -1.5571, -0.5521, -0.2563, -0.6761, -0.4234, -0.4155, -0.6174,
        -0.3280,  0.0206,  0.0459, -2.5961, -2.7955, -0.7617, -0.7957, -0.4254,
        -2.2515, -0.2618, -0.2551, -1.4000, -1.9968, -0.9473, -1.0472, -1.5193,
        -0.8290, -0.5298, -1.3960, -1.5992, -1.6144], device='cuda:0')

        self.value_range_tensor = torch.tensor([1.7558, 1.7926, 1.1905, 0.9004, 0.9268, 0.8572, 1.0855, 1.0616, 0.8480,
        1.7819, 1.8609, 3.7451, 3.9445, 1.2204, 1.1841, 2.6769, 2.7070, 0.5173,
        0.5169, 3.2215, 3.3968, 2.6473, 2.7151, 2.3118, 2.8012, 1.4724, 3.0105,
        3.2137, 3.2289], device='cuda:0')
        
        # Token output file
        self.token_output_file = 'generated_tokens_5.txt'
        
        # Response time output file
        self.response_time_file = 'read_tokens_response_time_5090.txt'
        
    def denormalize_torch(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize torch data"""
        data = (data+1).cuda(torch.cuda.current_device())
        return data * self.value_range_tensor / 2 + self.min_vals_tensor

    def get_token_dict(self):
        with self.lock:
            token_size = len(self.token_cache)
            dict_size = len(self.dict_cache)
            
            # Calculate difference (number of unconverted tokens)
            diff = token_size - dict_size
            
            # Save the last diff elements (in order), then remove from right side of queue
            tokens_to_process = []
            for _ in range(diff):
                tokens_to_process.append(self.token_cache.pop())
            # Reverse list to maintain original order (because pop is from right to left)
            tokens_to_process.reverse()
            
            # Convert to numpy array
            gen_token_ids = np.array(tokens_to_process, dtype=np.int64)
            # Convert to torch tensor
            gen_token_ids = torch.from_numpy(gen_token_ids)
            # Convert to codes
            gen_token_ids = gen_token_ids.cuda(torch.cuda.current_device())

            self.quantize = self.quantize.to(gen_token_ids.device)
            start_time = time.monotonic()
            gen_codes = self.quantize.indices_to_codes(gen_token_ids)
            with torch.no_grad():
                gen_codes = gen_codes.unsqueeze(0).cuda(torch.cuda.current_device())
                output = self.decoder(gen_codes)
            # Denormalize
            output = self.denormalize_torch(output)
            duration = time.monotonic() - start_time
            self._record_decode_time(duration)

            # Output shape is (1, diff*2, 29), put each (1, 29) item into dict_cache
            for i in range(output.shape[1]):  # Iterate through diff*2 items
                dict_item = output[:, i, :].cuda(torch.cuda.current_device())  # Get (1, 29) item
                dict_item = dict_item[:,byd_joint_to_mujoco_joint]  # Convert to MuJoCo joint order
                # Convert previous dof_pos to tensor for calculation
                prev_dof_pos_tensor = torch.tensor(self.dict_last['dof_pos'], device=dict_item.device)
                dof_vel = (dict_item - prev_dof_pos_tensor) * 50
                # Convert to list format for JSON serialization
                dof_vel = dof_vel.cpu().numpy().tolist()
                dict_item = dict_item.cpu().numpy().tolist()
                self.dict_last = {'dof_pos': dict_item, 'dof_vel': dof_vel}
                self.dict_cache.append(self.dict_last)
            
            # Put removed tokens back into queue in original order
            for token in tokens_to_process:
                self.token_cache.append(token)
                self.token_cache.append(token)  # Ensure synchronization with dict_cache

    def get_token_dict_overlap(self):
        with self.lock:
            token_size = len(self.token_cache)
            dict_size = len(self.dict_cache)
            
            # Calculate difference (number of unconverted tokens)
            diff = token_size - dict_size
            
            # Save the last diff elements (in order), then remove from right side of queue
            tokens_to_process = self.current_motion_tokens
            num = len(tokens_to_process)
            tokens_new = []
            for _ in range(diff):
                tokens_new.append(self.token_cache.pop())
            # Reverse list to maintain original order (because pop is from right to left)
            tokens_new.reverse()
            tokens_to_process = tokens_to_process + tokens_new
            
            # Convert to numpy array
            gen_token_ids = np.array(tokens_to_process, dtype=np.int64)
            # Convert to torch tensor
            gen_token_ids = torch.from_numpy(gen_token_ids)
            # Convert to codes
            gen_token_ids = gen_token_ids.cuda(torch.cuda.current_device())

            self.quantize = self.quantize.to(gen_token_ids.device)
            start_time = time.perf_counter()
            gen_codes = self.quantize.indices_to_codes(gen_token_ids)
            with torch.no_grad():
                gen_codes = gen_codes.unsqueeze(0).cuda(torch.cuda.current_device())
                output = self.decoder(gen_codes)
            # Denormalize
            output = self.denormalize_torch(output)
            duration = time.perf_counter() - start_time
            print(f"get_token_dict_overlap time: {duration * 1000:.3f} ms")
            # Output shape is (1, diff*2, 29), put each (1, 29) item into dict_cache
            for i in range(2*num,output.shape[1]):  # Iterate through diff*2 items
                dict_item = output[:, i, :].cuda(torch.cuda.current_device())  # Get (1, 29) item
                dict_item = dict_item[:,byd_joint_to_mujoco_joint]  # Convert to MuJoCo joint order
                # Convert previous dof_pos to tensor for calculation
                prev_dof_pos_tensor = torch.tensor(self.dict_last['dof_pos'], device=dict_item.device)
                dof_vel = (dict_item - prev_dof_pos_tensor) * 50
                # Convert to list format for JSON serialization
                dof_vel = dof_vel.cpu().numpy().tolist()
                dict_item = dict_item.cpu().numpy().tolist()
                self.dict_last = {'dof_pos': dict_item, 'dof_vel': dof_vel}
                self.dict_cache.append(self.dict_last)
            
            # Put removed tokens back into queue in original order
            for token in tokens_new:
                self.token_cache.append(token)
                self.token_cache.append(token)  # Ensure synchronization with dict_cache

    def load_model(self, model_path):
        """Load motion generation model"""
        print(f"Loading model from {model_path}...")
        self.model, self.tokenizer = load_finetuned_model(model_path)
        print("Model loaded successfully!")
        
    def parse_messages(self, data):
        """Parse received data, handle potentially merged JSON messages"""
        messages = []
        
        # Add new data to buffer
        self.message_buffer += data.decode('utf-8')
        
        # Split messages by delimiter
        while self.message_delimiter in self.message_buffer:
            # Find the position of the first delimiter
            delimiter_pos = self.message_buffer.find(self.message_delimiter)
            
            # Extract complete message
            message_str = self.message_buffer[:delimiter_pos].strip()
            
            # Remove processed message from buffer
            self.message_buffer = self.message_buffer[delimiter_pos + 1:]
            
            # Parse JSON message
            if message_str:
                try:
                    message = json.loads(message_str)
                    messages.append(message)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
                    print(f"Problematic message: {message_str}")
                    # Try to handle potentially merged messages {}{}
                    self.handle_merged_messages(message_str, messages)
        
        return messages
    
    def handle_merged_messages(self, message_str, messages):
        """Handle potentially merged JSON messages, such as {}{}"""
        # Try to find independent JSON objects
        brace_count = 0
        current_message = ""
        
        for char in message_str:
            current_message += char
            
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                
                # When braces are balanced, it indicates a complete JSON object
                if brace_count == 0:
                    try:
                        message = json.loads(current_message)
                        messages.append(message)
                        current_message = ""
                    except json.JSONDecodeError:
                        # If parsing fails, continue trying
                        pass
        
    def start_server(self):
        """Start server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        self.running = True
        
        print(f"Server listening on {self.host}:{self.port}")
        
        # Wait for proxy connection
        self.client_socket, addr = self.socket.accept()
        print(f"Proxy connected from {addr}")
        self.init_decoder()
        self.init_decoder()
        self.init_generate_tokens()
        # Start processing thread
        self.handle_requests()
        
    def handle_requests(self):
        """Handle requests from proxy"""
        while self.running:
            try:
                # Receive data
                data = self.client_socket.recv(4096)
                if not data:
                    print("Proxy disconnected")
                    break
                
                # Parse messages (may contain multiple merged messages)
                messages = self.parse_messages(data)
                
                # Process each message
                for message in messages:
                    message_type = message.get('type')
                    
                    if message_type == 'start_generation':
                        # Start generating new motion tokens
                        self.start_new_generation(
                            message['prompt'], 
                            message['motion_tokens']
                        )
                        
                    elif message_type == 'read_tokens':
                        # Record timestamp when read_tokens message is received
                        receive_time = time.time()
                        count = message.get('count', 1)
                        tokens = self.read_tokens(count)
                        
                        # Record timestamp when token response is sent
                        send_time = time.time()
                        
                        # Only record time difference when tokens are not empty
                        if tokens and len(tokens) > 0:
                            # Calculate time difference (milliseconds)
                            response_time = (send_time - receive_time) * 1000
                            
                            # Write time difference to file
                            try:
                                with open(self.response_time_file, 'a', encoding='utf-8') as f:
                                    f.write(f"{response_time:.3f}\n")
                            except Exception as e:
                                print(f"Error writing time difference file: {e}")
                        
                        response = {
                            'type': 'tokens_response',
                            'tokens': tokens,
                            'count': len(tokens),
                            'send_timestamp': send_time  # Add timestamp
                        }
                        self.send_response(response)
                        #print(f"[Time test] Server sent token response time: {send_time * 1000:.3f} ms")
                        
                    elif message_type == 'stop':
                        # Stop generation
                        self.stop_generation()
                    
            except Exception as e:
                print(f"Error handling request: {e}")
                break
                
        self.cleanup()
        
    def start_new_generation(self, prompt, motion_tokens):
        """Start new generation process"""
        with self.lock:
            # Stop current generation
            if self.generation_thread and self.generation_thread.is_alive():
                self.generation_running = False
                self.generation_thread.join()
            
            # Clear cache
            while len(self.token_cache) > 0:
                self.token_cache.popleft()
            while len(self.dict_cache) > 0:
                self.dict_cache.popleft()
            
            # Set new parameters
            self.current_prompt = prompt
            self.current_motion_tokens = motion_tokens
            # print("received motion_tokens:")
            # print(self.current_motion_tokens)
            self.past_key_values = None
            self.step_count = 0
            #breakpoint()
            # Start new generation thread
            self.generation_running = True
            self.generation_thread = threading.Thread(target=self.generate_tokens)
            self.generation_thread.start()
            
        print(f"Started new generation with prompt: {prompt[:50]}...")
        
    def generate_tokens(self):
        """Continuously generate tokens"""
        try:
            prompt_length = 0
            # First call: process prompt + motion tokens
            start_time = time.perf_counter()
            next_token_id, self.past_key_values, is_first, is_end_token, prompt_length = unified_generation_step(
                self.model, self.tokenizer,
                prompt=self.current_prompt,
                prompt_length=prompt_length,
                #motion_tokens=self.current_motion_tokens,
                motion_tokens=None,
                past_key_values=None,
                step_count=0
            )
            duration = time.perf_counter() - start_time
            self._record_token_time(duration)
            
            # Continuous generation loop
            max_tokens = 100000000  # Maximum number of tokens to generate
            is_end_token = False
            for step in range(max_tokens):
                if not self.generation_running:
                    break
                    
                # Add generated token to cache
                token_item = next_token_id.item()
                # Save token_item to file (one per line)
                try:
                    with open(self.token_output_file, 'a', encoding='utf-8') as f:
                        f.write(f"{token_item-MOTION_TOKEN_CONFIG['code_base_id']}\n")
                except Exception as e:
                    print(f"Error writing token file: {e}")
                self.token_cache.append(token_item-MOTION_TOKEN_CONFIG['code_base_id'])
                #print(f"token_item: {token_item-MOTION_TOKEN_CONFIG['code_base_id']}")
                #print(self.current_motion_tokens)
                if self.current_motion_tokens is not None and len(self.current_motion_tokens) > 0 and len(self.token_cache) - len(self.dict_cache) >= 16:
                    #print("11111")
                    #print(self.current_motion_tokens)
                    self.get_token_dict_overlap()
                    self.current_motion_tokens = None
                elif len(self.token_cache) - len(self.dict_cache) >= 32:
                    #print("22222")
                    self.get_token_dict()
                
                # Generate next token
                start_time = time.perf_counter()
                next_token_id, self.past_key_values, is_first, is_end_token, prompt_length = unified_generation_step(
                    self.model, self.tokenizer,
                    prompt=None,
                    prompt_length=prompt_length,
                    motion_tokens=next_token_id,
                    past_key_values=self.past_key_values,
                    step_count=step + 1
                )
                duration = time.perf_counter() - start_time
                self._record_token_time(duration)
                #if is_end_token:
                    #print(f"is_end_token: {is_end_token}")
                
        except Exception as e:
            print(f"Error in token generation: {e}")
            
    def read_tokens(self, count):
        """Read tokens from cache"""
        tokens = []
        with self.lock:
            for _ in range(min(count, len(self.dict_cache))):
                if len(self.dict_cache) > 0:
                    tokens.append({'token': self.token_cache.popleft(), 'dict': self.dict_cache.popleft()})
        return tokens
        
    def stop_generation(self):
        """Stop generation"""
        with self.lock:
            self.generation_running = False
            if self.generation_thread and self.generation_thread.is_alive():
                self.generation_thread.join()
                
    def send_response(self, response):
        """Send response to proxy"""
        try:
            message = json.dumps(response) + self.message_delimiter
            bytes_sent = self.client_socket.send(message.encode('utf-8'))
            #print(f"Server sent {bytes_sent} bytes, response type: {response.get('type')}, tokens count: {response.get('count', 0)}")
        except Exception as e:
            print(f"Error sending response: {e}")
            import traceback
            traceback.print_exc()
            
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.stop_generation()
        
        if self.client_socket:
            self.client_socket.close()
        if self.socket:
            self.socket.close()
            
        print("Server cleaned up")

    def init_decoder(self):
        token_ids = torch.randint(
            low=1,                # Minimum value of random integer (inclusive)
            high=100,             # Maximum value of random integer (exclusive)
            size=(1, 32),         # Tensor shape: 1 row, 32 columns
            device=torch.cuda.current_device()  # Device consistent with original code (current CUDA device)
        )
        self.quantize = self.quantize.to(torch.cuda.current_device())
        gen_codes = self.quantize.indices_to_codes(token_ids.squeeze(0))
        with torch.no_grad():
            gen_codes = gen_codes.unsqueeze(0).cuda(torch.cuda.current_device())
            output = self.decoder(gen_codes)
        # Denormalize
        output = self.denormalize_torch(output)
        print("init_decoder done")
    
    def init_generate_tokens(self):
        """Continuously generate tokens"""
        try:
            prompt_length = 0
            # First call: process prompt + motion tokens
            next_token_id, self.past_key_values, is_first, is_end_token, prompt_length = unified_generation_step(
                self.model, self.tokenizer,
                prompt="hello qwen",
                prompt_length=prompt_length,
                #motion_tokens=self.current_motion_tokens,
                motion_tokens=None,
                past_key_values=None,
                step_count=0
            )
            
            # Continuous generation loop
            max_tokens = 3  # Maximum number of tokens to generate
            for step in range(max_tokens):
                if not self.generation_running:
                    break
                    
                # Generate next token
                next_token_id, self.past_key_values, is_first, is_end_token, prompt_length = unified_generation_step(
                    self.model, self.tokenizer,
                    prompt=None,
                    prompt_length=prompt_length,
                    motion_tokens=next_token_id,
                    past_key_values=self.past_key_values,
                    step_count=step + 1
                )
                
        except Exception as e:
            print(f"Error in token generation: {e}")
        print("init_generate_tokens done")

    def _record_token_time(self, duration):
        with self.timing_lock:
            self.token_time_records.append(duration)
            self.token_total_time += duration
            count = len(self.token_time_records)
            avg = self.token_total_time / count if count else 0.0
        print(f"[Timing] Token generation #{count} took {duration * 1000:.3f} ms, average {avg * 1000:.3f} ms")

    def _record_decode_time(self, duration):
        with self.timing_lock:
            self.decode_time_records.append(duration)
            self.decode_total_time += duration
            count = len(self.decode_time_records)
            avg = self.decode_total_time / count if count else 0.0
        print(f"[Timing] Decode processing #{count} took {duration * 1000:.3f} ms, average {avg * 1000:.3f} ms")

def main():
    # Configuration
    MODEL_PATH = "your_model_path"
    HOST = '0.0.0.0'  # Listen on all interfaces, allow external access
    PORT = 8000
    
    # Create server
    server = MotionServer(HOST, PORT)

    local_ip = get_local_ip()
    print(f"Server listening on {local_ip}:{PORT}")
    try:
        # Load model
        server.load_model(MODEL_PATH)
        # Start server
        server.start_server()
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        server.cleanup()

if __name__ == '__main__':
    main()
