<div align="center">
<h1 style="border-bottom: none; margin-bottom: 0px ">UniAct: Unified Motion Generation and Action Streaming for Humanoid Robots</h1>

[**Nan Jiang**](https://jnnan.github.io/)<sup>&ast;</sup> · [**Zimo He**](https://mileret.github.io/)<sup>&ast;</sup> · [**Wanhe Yu**](https://pku.ai/author/wanhe-yu/) · [**Lexi Pang**](https://tongclass.ac.cn/author/lexi-pang/) · [**Yunhao Li**](https://pku.ai/author/yunhao-li/) · [**Hongjie Li**](https://awfuact.github.io/) · [**Jieming Cui**](https://jiemingcui.github.io/) · [**Yuhan Li**](https://github.com/jnnan/uniact-code/) · [**Yizhou Wang**](https://cfcs.pku.edu.cn/wangyizhou/) · [**Yixin Zhu**](https://yzhu.io/)<sup>&dagger;</sup> · [**Siyuan Huag**](https://siyuanhuang.com/)<sup>&dagger;</sup> 
<br>
&ast;Equal Contribution&emsp;&dagger;Corresponding Author

<a href=""><img src='https://img.shields.io/badge/arXiv-UniAct-red' alt='Paper PDF'></a>
<a href='https://jnnan.github.io/uniact'><img src='https://img.shields.io/badge/Project_Page-UniAct-green' alt='Project Page'></a>
<a href=''><img src='https://img.shields.io/badge/Video-UniAct-yellow' alt='Video'></a>

</div>

![alt text](https://github.com/jnnan/uniact-code/blob/main/images/framework.png?raw=true)

# Server-Proxy-Client Motion Generation System

This system is a three-tier architecture real-time motion generation system, consisting of three components: Server (server), Proxy (proxy), and Robot Client (robot client).

---

## Table of Contents

- [Server Usage Instructions](#server-usage-instructions)
- [Robot Client Usage Instructions](#robot-client-usage-instructions)

---

## Model Checkpoints

The pretrained checkpoints for the motion generator, decoder, and tracking policy are available for download:

**[Download Checkpoints (Google Drive)](https://drive.google.com/drive/folders/1sh1IQdIjvnxx2s2did0vvZVP9DuCpoGE?usp=sharing)**

The download includes:
- `motion-generator` - Motion generator model weights
- `decoder.pt` - Decoder model weights  
- `policy.pt` - Tracking policy model weights


## Server Usage Instructions

### 1. Environment Setup

#### 1.1 Install Python Dependencies

Install all required dependencies using the `requirements_qwen.txt` file provided by the project:

```bash
pip install -r requirements_qwen.txt
```

**Note**:
- Ensure your Python version is 3.11 or higher
- It is recommended to use a virtual environment (virtualenv or conda) to manage dependencies
- If using GPU, ensure CUDA and cuDNN are properly installed

### 2. Path Configuration

Before running the Server, you need to configure the following paths:

#### 2.1 Configure Model Path

Open the `server.py` file, find the `main()` function (around line 621), and modify `MODEL_PATH` to your model directory path:

```python
def main():
    # Configuration
    MODEL_PATH = "your_model_path"  # Modify to your model path, e.g.: "/path/to/your/model"
    HOST = '0.0.0.0'
    PORT = 8000
    # ...
```

#### 2.2 Configure Decoder File Path

Open the `server.py` file, find the `__init__` method of the `MotionServer` class (around line 147), and modify the decoder file path:

```python
self.decoder = torch.jit.load('your_decoder_file_path.pt')  # Modify to your decoder file path
```

**Note**:
- If the decoder file is in the project root directory, you can directly use the filename (e.g., `'decoder.pt'`)
- If the decoder file is in another location, use a relative or absolute path (e.g., `'/path/to/decoder.pt'` or `'./models/decoder.pt'`)

#### 2.3 Configure infer_robot Module Path (Optional)

Open the `server.py` file, find the `sys.path.append` at the beginning of the file (around line 25):

- **If `infer_robot.py` is in the project root directory**: No modification needed, keep it commented
- **If `infer_robot.py` is in another directory**: Uncomment and set the correct path:

```python
# If infer_robot.py is not in the project root, uncomment and set the path:
sys.path.append('your_infer_robot_directory_path')  # Modify to the directory path containing infer_robot.py
```

### 3. Running the Server

#### 3.1 Direct Run

```bash
python server.py
```

#### 3.2 Running Instructions

- After the Server starts, it will load the model (this may take some time)
- The Server listens on `0.0.0.0:8000` by default, allowing external access
- After successful startup, it will display the local IP address and port number
- The Server will wait for Proxy connection, and after successful connection, it will start processing requests


## Robot Client Usage Instructions

### 1. Environment Setup

Install all required dependencies using the `requirements.txt` file provided by the project:

```bash
pip install -r requirements.txt
```

**Note**:
- Ensure your Python version is 3.11 or higher
- It is recommended to use a virtual environment (virtualenv or conda) to manage dependencies
- If using GPU, ensure CUDA and PyTorch GPU version are properly installed

### 2. Configuration File Path Settings

Before running Robot Client, you need to configure the paths for models and motion files.

#### 2.1 Configuration File Location

- **Using `robot_client.py`**: Requires configuration file `configs/g1_ref_real.yaml`
  - If the `configs/` directory does not exist, you need to create it

#### 2.2 Path Configuration to Modify

Open the configuration file and modify the following **required** path settings:

```yaml
# MJCF robot model file path (must be adapted)
xml_path: "./unitree_description/mjcf/g1.xml"
# Modify to the actual MJCF file path, for example:
# xml_path: "/path/to/your/unitree_description/mjcf/g1.xml"

# Policy model file path (must be modified)
policy_path: "path/to/your_motion_tracking_policy.pt"
# Modify to the actual policy model file path, for example:
# policy_path: "/path/to/your/policy.pt"
# Note: The code will automatically find the corresponding encoder file {policy_path.replace('.pt', '_encoder.pt')}

# Motion data file path (must be modified if ONLINE_MOTION=False)
motion_file: './data/motion_data/raw_walking_g1_deploy.pkl'
# Modify to the actual motion data file path, for example:
# motion_file: '/path/to/your/raw_walking_g1_deploy.pkl'
```

**Path Configuration Description:**

| Configuration Item | Description | Required | Path Type |
|--------|------|---------|---------|
| `xml_path` | MuJoCo robot model file (XML format) | **Required** | Absolute or relative path |
| `policy_path` | Trained motion tracking model (.pt file, TorchScript format) | **Required** | Absolute or relative path |
| `motion_file` | Pre-recorded motion data file (.pkl file) | Required only when `ONLINE_MOTION=False` | Absolute or relative path |

**Path Format Recommendations:**
- It is recommended to use **absolute paths** to avoid path resolution issues
- If using relative paths, ensure they are relative to the current working directory when running `robot_client.py`
- Ensure all file paths exist and are accessible


### 3. Command Line Invocation

#### 3.1 Basic Run

```bash
python robot_client.py
```

### 4. Instruction Reception Mode Switching

Robot Client supports two modes for receiving instructions:

#### 4.1 File Mode (Default)

Automatically reads and sends instructions from the `text.jsonl` file.

**Usage:**
```bash
# Default mode (file mode)
python robot_client.py

# Or explicitly specify
python robot_client.py --use_text_file
```

**File Format Requirements:**

Create a `text.jsonl` file in the project root directory, formatted as JSON Lines (one JSON object per line):

```json
{"frame": 0, "text": "walk forward"}
{"frame": 100, "text": "turn left"}
{"frame": 200, "text": "stop"}
```

- `frame`: Integer, specifies at which frame to send this instruction (frame count starts from 0)
- `text`: String, instruction text content

**Workflow:**
1. The program reads the `text.jsonl` file at startup
2. In the main loop, when the frame count reaches the specified `frame` value, it automatically sends the corresponding instruction
3. Instructions are executed in ascending order of `frame` values

#### 4.2 Command Line Mode

Interactively input instructions through the command line.

**Usage:**
```bash
python robot_client.py --use_commandline
```

**Supported Commands:**
- `start <prompt>`: Start generating motion, e.g., `start walk forward`
- `stop`: Stop generation
- `quit`: Exit the program

### 5. Runtime Configuration Parameters

In the `main()` function of `robot_client.py` (around lines 1309-1319), you can modify the following configuration parameters:

```python
config = {
    'server_host': 'localhost',           # Server host address
    'server_port': 8000,                  # Server port number
    'frequency': 50,                      # Control frequency (Hz)
    'ready_threshold': 30,                # When the number of tokens in the queue reaches this value, ready is true
    'buffer_threshold': 50,               # When the number of tokens in the queue is less than or equal to this value, request more tokens
    'keep_tokens_on_new_instruction': 30, # Number of tokens to keep when a new instruction is sent
    'keep_tokens_for_generate': 48,       # Number of tokens to keep for generating new tokens
    'read_batch_size': 20,                # Number of tokens to read from server each time
    'use_text_file': use_text_file        # Whether to use file mode
}
```

#### 5.1 Parameter Detailed Description

| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| `server_host` | str | `'localhost'` | Proxy/Server host address. If the Server runs on another machine, modify to the corresponding IP address |
| `server_port` | int | `8000` | Proxy/Server port number. Must match the Server configuration |
| `frequency` | int | `50` | Control loop frequency (Hz), i.e., the number of control steps executed per second. 50Hz means executing once every 20ms |
| `ready_threshold` | int | `30` | When the number of tokens in the Proxy queue reaches this threshold, the Client considers the Proxy ready and can start reading motion state. **Increasing this value will increase startup delay but improve stability** |
| `buffer_threshold` | int | `50` | When the number of tokens in the Proxy queue ≤ this threshold, the Proxy will automatically request the Server to generate more tokens. **Increasing this value can increase cache capacity and reduce waiting** |
| `keep_tokens_on_new_instruction` | int | `30` | When sending a new instruction, keep the first N tokens in the queue (for maintaining motion continuity). **Increasing this value allows new instructions to more smoothly continue from previous motions** |
| `keep_tokens_for_generate` | int | `48` | Number of historical tokens sent to the Server when generating a new motion sequence (for context). **Must match the overlap number when decoding on the server** |
| `read_batch_size` | int | `20` | Number of tokens the Proxy requests from the Server each time. **Increasing this value can reduce request frequency but will increase delay** |
| `use_text_file` | bool | `True` | Whether to use file mode. This parameter is usually automatically set through command line arguments `--use_text_file` or `--use_commandline` |

### 6. Distributed Deployment Configuration

When the Server and Robot Client run on different devices, you need to use SSH tunnels to establish connections.

#### 6.1 Deployment Architecture

**Important Principle: Proxy and Robot Client must run on the same device.**

```
Device A (running Server):
  └─ Server (listening on 0.0.0.0:8000)

Device B (running Proxy + Robot Client):
  └─ Proxy (connecting to localhost:8000, forwarded through SSH tunnel)
  └─ Robot Client (connecting to localhost:8000, through Proxy)
```

#### 6.2 Establish SSH Tunnel

On the **device running Proxy and Robot Client**, use SSH port forwarding to establish a tunnel:

```bash
ssh -L 8000:172.17.0.9:8000 -p 40291 root@connect.bjb2.seetacloud.com
```

**Parameter Description:**
- `-L 8000:172.17.0.9:8000`: Forward local port 8000 to remote server's 172.17.0.9:8000
- `-p 40291`: SSH connection port number
- `root@connect.bjb2.seetacloud.com`: SSH server address and username

**Replacements for actual use:**
- `172.17.0.9`: Replace with the actual IP address where the Server runs (usually the internal IP of the container where the Server is located or the IP the Server binds to)
- `40291`: Replace with the actual SSH port number
- `connect.bjb2.seetacloud.com`: Replace with the actual SSH jump server address

#### 6.3 Configuration Steps

1. **Start Server on Remote Server (Device A)**

   First, you need to start the Server on the remote device where the Server runs:

   ```bash
   # On Device A (remote server)
   python server.py
   ```

   The Server listens on `0.0.0.0:8000` by default. Ensure the Server has started and is running normally.

2. **Establish SSH Tunnel**

   On the **device running Proxy and Robot Client (Device B)**, use SSH port forwarding to establish a tunnel:

   ```bash
   ssh -L 8000:<server_ip>:8000 -p <ssh_port> <user>@<ssh_host>
   ```

   For example:
   ```bash
   ssh -L 8000:172.17.0.9:8000 -p 40291 root@connect.bjb2.seetacloud.com
   ```

   **Important**: Keep this SSH terminal window open (disconnecting the SSH connection will interrupt the tunnel, and Proxy and Client will be unable to connect to the Server).

3. **Configure Robot Client**

   In the `config` of `robot_client.py`, set `server_host` to `'localhost'`:

   ```python
   config = {
       'server_host': 'localhost',  # Through SSH tunnel, use localhost
       'server_port': 8000,
       # ... other configurations
   }
   ```

4. **Start Robot Client**

   Start Robot Client in another terminal window on Device B:

   ```bash
   # On Device B (new terminal window)
   python robot_client.py  # Start Robot Client
   ```

#### 6.4 Complete Startup Flow Example

**Device A (Remote Server)**:
```bash
# 1. Start Server, obtain the actual IP address where the Server runs
python server.py
# Server listens on 0.0.0.0:8000
```

Then terminate it

**Device B (Local, running Client)**:
```bash
# 1. Establish SSH tunnel based on the Server's actual IP address obtained in the previous step (keep this terminal window open)
ssh -L 8000:172.17.0.9:8000 -p 40291 root@connect.bjb2.seetacloud.com

# 2. Start Server
python server.py

# 3. Start Robot Client in a new terminal window
python robot_client.py
```
