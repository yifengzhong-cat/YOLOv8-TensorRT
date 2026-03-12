# YOLOv8-TensorRT 云端部署操作文档

本文档详细说明如何使用 Docker 将 YOLOv8 模型部署到云端，并通过 REST API 接口进行图像和视频流的推理分析。

---

## 目录

- [1. 环境准备](#1-环境准备)
- [2. 项目文件说明](#2-项目文件说明)
- [3. 模型准备](#3-模型准备)
- [4. Docker 构建与启动](#4-docker-构建与启动)
- [5. 模型自动转换](#5-模型自动转换)
- [6. API 接口文档](#6-api-接口文档)
  - [6.1 健康检查](#61-健康检查)
  - [6.2 算法能力获取](#62-算法能力获取)
  - [6.3 视频任务管理](#63-视频任务管理)
  - [6.4 分析任务控制](#64-分析任务控制)
  - [6.5 图片分析任务](#65-图片分析任务)
  - [6.6 样本数据上传](#66-样本数据上传)
  - [6.7 更新分析ID](#67-更新分析id)
- [7. 配置说明](#7-配置说明)
- [8. 常见问题与排查](#8-常见问题与排查)

---

## 1. 环境准备

### 1.1 硬件要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA 显卡（推荐 Tesla T4 / V100 / A100 / RTX 30/40 系列） |
| 显存 | >= 4GB（推荐 8GB 以上，多模型加载需要更多显存） |
| 内存 | >= 8GB |
| 磁盘 | >= 20GB（Docker 镜像约 15GB） |

### 1.2 软件要求

| 软件 | 版本要求 | 安装说明 |
|------|---------|---------|
| 操作系统 | Ubuntu 18.04 / 20.04 / 22.04 | - |
| NVIDIA 驱动 | >= 525.60 | `apt install nvidia-driver-525` |
| Docker | >= 20.10 | [安装指南](https://docs.docker.com/engine/install/) |
| Docker Compose | >= 2.0 | [安装指南](https://docs.docker.com/compose/install/) |
| NVIDIA Container Toolkit | >= 1.13 | 见下方安装步骤 |

### 1.3 安装 NVIDIA Container Toolkit

```bash
# 添加 NVIDIA 软件源
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 安装
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 配置 Docker 运行时
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 验证安装
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### 1.4 验证 GPU 可用

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 Docker GPU 支持
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

---

## 2. 项目文件说明

```
YOLOv8-TensorRT/
├── Dockerfile                 # Docker 镜像构建文件
├── docker-compose.yml         # Docker Compose 编排文件
├── models_config.json         # 模型配置文件（算法编码、标签、文件路径）
├── scripts/
│   ├── entrypoint.sh          # Docker 容器入口脚本
│   └── convert_models.py      # 模型转换脚本 (.pt → .onnx → .engine)
├── csrc/api/
│   ├── main.cpp               # C++ HTTP API 服务器源码
│   ├── CMakeLists.txt          # CMake 构建配置
│   └── include/               # YOLOv8 推理引擎头文件
├── models_dir/                # 模型文件存放目录（需自行创建并放入 .pt 文件）
└── output/                    # 推理结果输出目录（自动创建）
```

---

## 3. 模型准备

### 3.1 放置模型文件

在项目根目录下创建 `models_dir` 文件夹，并将 4 个 `.pt` 模型文件放入：

```bash
# 创建模型目录
mkdir -p models_dir

# 复制模型文件
cp /path/to/model_mengdong_raa_adjusted.pt    models_dir/
cp /path/to/model_mengdong_small_SRL.pt       models_dir/
cp /path/to/model_mengdong_tower.pt           models_dir/
cp /path/to/model_mengdong_scene_album.pt     models_dir/
```

### 3.2 模型信息

| 模型文件 | 算法编码 (algCode) | 算法描述 | 检测标签 |
|---------|-------------------|---------|---------|
| `model_mengdong_raa_adjusted.pt` | `010161` | 安全装备检测 | 人、安全带、蓝色安全帽、橘色安全帽、红色安全帽、白色安全帽、黄色安全帽 |
| `model_mengdong_small_SRL.pt` | `010201` | 钩子与差速器检测 | 钩子、差速器 |
| `model_mengdong_tower.pt` | `010301` | 铁塔检测 | 铁塔 |
| `model_mengdong_scene_album.pt` | `010401` | 场景设备检测 | 斗臂车、导线间隔棒、变压器、刀闸、断路器、高空屋顶、脚手架、脚扣、电线杆、线路 |

### 3.3 模型自动转换

容器首次启动时会自动执行：
1. 检测 `models_dir/` 中的 `.pt` 文件
2. 将 `.pt` 导出为 `.onnx`（包含 End2End NMS）
3. 将 `.onnx` 构建为 TensorRT `.engine`（FP16 精度）

> **注意**: 首次转换需要较长时间（每个模型约 5-15 分钟），请耐心等待。后续启动会跳过已转换的模型。

---

## 4. Docker 构建与启动

### 4.1 使用 Docker Compose（推荐）

```bash
# 第一步：构建 Docker 镜像（首次约 15-30 分钟）
docker compose build

# 第二步：启动服务
docker compose up -d

# 查看日志
docker compose logs -f

# 停止服务
docker compose down
```

### 4.2 使用 Docker 命令

```bash
# 构建镜像
docker build -t yolov8-tensorrt-api:latest .

# 运行容器
docker run -d \
    --name yolov8-api \
    --gpus all \
    --runtime nvidia \
    -p 22266:22266 \
    -v $(pwd)/models_dir:/workspace/models_dir \
    -v $(pwd)/output:/workspace/output \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    --restart unless-stopped \
    yolov8-tensorrt-api:latest
```

### 4.3 验证服务启动

```bash
# 等待服务启动完成（首次需等模型转换）
docker compose logs -f

# 当看到以下输出表示启动成功：
# ============================================
#   YOLOv8-TensorRT API Server
# ============================================
# Host:        0.0.0.0
# Port:        22266
# Models:      4 loaded

# 健康检查
curl http://localhost:22266/health
# 返回: {"models_loaded":4,"status":"ok"}
```

### 4.4 自定义环境变量

可以在 `docker-compose.yml` 中修改以下环境变量：

| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| `SERVER_PORT` | `22266` | API 服务监听端口 |
| `SERVER_HOST` | `0.0.0.0` | API 服务监听地址 |
| `MODELS_DIR` | `/workspace/models_dir` | 模型文件目录 |
| `OUTPUT_DIR` | `/workspace/output` | 推理输出目录 |
| `CONFIG_FILE` | `/workspace/models_config.json` | 模型配置文件路径 |

---

## 5. 模型自动转换

### 5.1 自动转换流程

容器启动时，入口脚本 `scripts/entrypoint.sh` 会自动检测并转换模型：

```
启动容器
  ↓
检查 models_dir/ 中是否有 .pt 文件
  ↓ 有 .pt 文件且没有 .engine 文件
执行 convert_models.py 自动转换
  ↓ .pt → .onnx → .engine (FP16)
启动 C++ API 服务器
```

### 5.2 手动转换模型

如果需要自定义转换参数，可以进入容器手动操作：

```bash
# 进入容器
docker compose exec yolov8-api bash

# 手动转换（支持 FP16）
python3 /workspace/scripts/convert_models.py \
    --models-dir /workspace/models_dir \
    --config /workspace/models_config.json \
    --fp16 \
    --conf-thres 0.25 \
    --iou-thres 0.65 \
    --topk 100
```

### 5.3 使用预转换的 Engine 文件

如果你已有 `.engine` 文件，直接放入 `models_dir/` 即可，容器会跳过转换步骤：

```bash
cp /path/to/model_mengdong_raa_adjusted.engine    models_dir/
cp /path/to/model_mengdong_small_SRL.engine       models_dir/
cp /path/to/model_mengdong_tower.engine           models_dir/
cp /path/to/model_mengdong_scene_album.engine     models_dir/
```

> **注意**: `.engine` 文件与 GPU 硬件绑定，不同型号的显卡之间不通用，需重新转换。

---

## 6. API 接口文档

**服务地址**: `http://<服务器IP>:22266`

**所有接口均使用 JSON 格式进行数据交互。**

### 6.1 健康检查

- **接口**: `GET /health`
- **说明**: 检查服务是否正常运行

```bash
curl http://localhost:22266/health
```

**响应示例:**
```json
{
    "status": "ok",
    "models_loaded": 4
}
```

---

### 6.2 算法能力获取

- **接口**: `POST /v1/service/abilities`
- **说明**: 获取当前已加载的所有算法能力列表

```bash
curl -X POST http://localhost:22266/v1/service/abilities \
    -H "Content-Type: application/json"
```

**响应示例:**
```json
{
    "resultCode": "200",
    "resultValue": {
        "abilityInfo": {
            "number": 4,
            "ability": [
                {
                    "algCode": "010161",
                    "algDesc": "安全装备检测",
                    "algParams": [
                        {
                            "key": "--sensitivity",
                            "value": "灵敏度,范围[1,5]"
                        }
                    ]
                },
                {
                    "algCode": "010201",
                    "algDesc": "钩子与差速器检测",
                    "algParams": [
                        {
                            "key": "--sensitivity",
                            "value": "灵敏度,范围[1,5]"
                        }
                    ]
                },
                {
                    "algCode": "010301",
                    "algDesc": "铁塔检测",
                    "algParams": [
                        {
                            "key": "--sensitivity",
                            "value": "灵敏度,范围[1,5]"
                        }
                    ]
                },
                {
                    "algCode": "010401",
                    "algDesc": "场景设备检测",
                    "algParams": [
                        {
                            "key": "--sensitivity",
                            "value": "灵敏度,范围[1,5]"
                        }
                    ]
                }
            ]
        }
    },
    "resultHint": null
}
```

---

### 6.3 视频任务管理

- **接口**: `POST /v1/service/videoTask`
- **说明**: 创建视频流分析任务。服务端会拉取视频流，逐帧推理并保存为标注后的 MP4 文件。

```bash
curl -X POST http://localhost:22266/v1/service/videoTask \
    -H "Content-Type: application/json" \
    -d '{
        "algCode": "010161",
        "startTime": "2024-01-01 00:00:00",
        "endTime": "2024-12-31 23:59:59",
        "interval": 60,
        "command": 1,
        "videoInfo": [
            {
                "analyseId": "task-001",
                "devCode": "camera-001",
                "formatType": 0,
                "videoUrl": "rtsp://192.168.1.100:554/stream1"
            }
        ],
        "rule": {
            "algParams": [
                {
                    "key": "--sensitivity",
                    "value": "3"
                }
            ]
        }
    }'
```

**请求参数:**

| 参数名 | 类型 | 必选 | 说明 |
|--------|------|------|------|
| algCode | String | 是 | 算法能力编码，如 `010161` 代表安全装备检测 |
| startTime | String | 是 | 任务开始时间 |
| endTime | String | 是 | 任务结束时间 |
| interval | Integer | 是 | 结果回传间隔（秒） |
| command | Integer | 是 | 控制指令：0=停止, 1=开始, 2=删除 |
| videoInfo | Array | 是 | 视频信息列表 |
| videoInfo[].analyseId | String | 是 | 分析任务ID |
| videoInfo[].devCode | String | 是 | 摄像头设备编码 |
| videoInfo[].formatType | Integer | 是 | 0: H264, 1: H265 |
| videoInfo[].videoUrl | String | 是 | 视频流地址（支持 RTSP/HTTP） |
| rule | Object | 否 | 规则配置 |

**响应示例:**
```json
{
    "resultCode": "200",
    "resultValue": [
        {
            "analyseId": "task-001",
            "devCode": "camera-001",
            "osdVideoUrl": "http://0.0.0.0:22266/output/task-001.mp4"
        }
    ],
    "resultHint": null
}
```

> **说明**: `osdVideoUrl` 是推理渲染后的视频文件下载地址，格式为 MP4。

---

### 6.4 分析任务控制

- **接口**: `POST /v1/service/controlTask`
- **说明**: 控制已创建的分析任务（启动/停止/删除）

```bash
# 停止任务
curl -X POST http://localhost:22266/v1/service/controlTask \
    -H "Content-Type: application/json" \
    -d '{
        "analyseId": "task-001",
        "command": 0
    }'

# 启动任务
curl -X POST http://localhost:22266/v1/service/controlTask \
    -H "Content-Type: application/json" \
    -d '{
        "analyseId": "task-001",
        "command": 1
    }'

# 删除任务
curl -X POST http://localhost:22266/v1/service/controlTask \
    -H "Content-Type: application/json" \
    -d '{
        "analyseId": "task-001",
        "command": 2
    }'
```

**请求参数:**

| 参数名 | 类型 | 必选 | 说明 |
|--------|------|------|------|
| analyseId | String | 是 | 分析任务ID |
| command | Integer | 是 | 0=停止, 1=启动, 2=删除 |

**响应示例:**
```json
{
    "resultCode": "200",
    "resultValue": null,
    "resultHint": "任务已停止"
}
```

---

### 6.5 图片分析任务

- **接口**: `POST /v1/service/imageTask`
- **说明**: 对单张图片进行推理分析，返回标注后的图片（Base64编码）和检测结果

```bash
# 将图片转为 Base64 并发送请求
IMAGE_BASE64=$(base64 -w 0 test.jpg)
curl -X POST http://localhost:22266/v1/service/imageTask \
    -H "Content-Type: application/json" \
    -d "{
        \"analyseId\": \"img-001\",
        \"algCode\": \"010161\",
        \"imageData\": \"${IMAGE_BASE64}\"
    }"
```

**请求参数:**

| 参数名 | 类型 | 必选 | 说明 |
|--------|------|------|------|
| analyseId | String | 是 | 分析任务ID |
| algCode | String | 是 | 算法编码（如 `010161`） |
| imageData | String | 是 | 图片数据，Base64编码 |
| rule | Object | 否 | 规则框配置 |

**响应示例:**
```json
{
    "resultCode": "200",
    "resultValue": {
        "analyseTime": "2024-01-15 10:30:00",
        "analyseResults": [
            "人",
            "白色安全帽",
            "安全带"
        ],
        "rawImageName": "img-001_raw.jpg",
        "rawImageData": "<Base64编码的原图>",
        "osdImageName": "img-001_osd.jpg",
        "osdImageData": "<Base64编码的结果图>",
        "resultDetail": [
            {
                "algCode": "010161",
                "resultDesc": "安全装备检测",
                "num": 3,
                "resultItems": [
                    {
                        "score": 95.7,
                        "leftTopX": 100,
                        "leftTopY": 50,
                        "rightBottomX": 300,
                        "rightBottomY": 400
                    },
                    {
                        "score": 89.3,
                        "leftTopX": 120,
                        "leftTopY": 10,
                        "rightBottomX": 280,
                        "rightBottomY": 60
                    },
                    {
                        "score": 87.1,
                        "leftTopX": 150,
                        "leftTopY": 200,
                        "rightBottomX": 250,
                        "rightBottomY": 350
                    }
                ]
            }
        ]
    },
    "resultHint": null
}
```

**结果图解码保存示例（Python）:**
```python
import base64
import json
import requests

# 发送请求
with open("test.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:22266/v1/service/imageTask", json={
    "analyseId": "img-001",
    "algCode": "010161",
    "imageData": image_b64
})

result = resp.json()

# 保存结果图
osd_data = base64.b64decode(result["resultValue"]["osdImageData"])
with open("result.jpg", "wb") as f:
    f.write(osd_data)

# 打印检测结果
for item in result["resultValue"]["resultDetail"][0]["resultItems"]:
    print(f"置信度: {item['score']:.1f}%, "
          f"位置: ({item['leftTopX']},{item['leftTopY']}) - "
          f"({item['rightBottomX']},{item['rightBottomY']})")
```

---

### 6.6 样本数据上传

- **接口**: `POST /v1/service/uploadSamples`
- **说明**: 上传样本数据（zip压缩包）的文件ID和MD5

```bash
curl -X POST http://localhost:22266/v1/service/uploadSamples \
    -H "Content-Type: application/json" \
    -d '{
        "fileId": "sample-file-001",
        "md5": "d41d8cd98f00b204e9800998ecf8427e"
    }'
```

**请求参数:**

| 参数名 | 类型 | 必选 | 说明 |
|--------|------|------|------|
| fileId | String | 是 | 样本文件ID |
| md5 | String | 是 | 文件MD5校验值 |

**响应示例:**
```json
{
    "resultCode": "200",
    "resultValue": {
        "fileId": "sample-file-001"
    },
    "resultHint": "接收样本成功"
}
```

---

### 6.7 更新分析ID

- **接口**: `POST /analysis/api/v1/updateAnalyseID`
- **说明**: 更新正在运行的分析任务的ID，服务会自动用新ID重新拉流

```bash
curl -X POST http://localhost:22266/analysis/api/v1/updateAnalyseID \
    -H "Content-Type: application/json" \
    -d '{
        "oldAnalyseId": "task-001",
        "newAnalyseId": "task-001-updated"
    }'
```

**请求参数:**

| 参数名 | 类型 | 必选 | 说明 |
|--------|------|------|------|
| oldAnalyseId | String | 是 | 原分析ID |
| newAnalyseId | String | 是 | 新分析ID |

**响应示例:**
```json
{
    "resultCode": "200",
    "resultValue": {
        "newAnalyseId": "task-001-updated"
    },
    "resultHint": "更新成功"
}
```

---

### 6.8 视频分析结果回调

视频任务运行时，服务端会按照设定的 `interval`（秒）定期将分析结果推送到配置的回调地址。

**回调接口**: `POST /analysis/api/v1/analyseResult`

回调数据格式：
```json
{
    "algCode": "010161",
    "analyseId": "task-001",
    "analyseTime": "2024-01-15 10:30:00",
    "analyseResults": ["未带安全帽", "人"],
    "osdImageName": "task-001.jpg",
    "osdImageData": "<Base64编码的检测结果图>",
    "resultDetail": [
        {
            "algCode": "010161",
            "resultDesc": "detection",
            "num": 2,
            "resultItems": [
                {
                    "score": 95.7,
                    "leftTopX": 200,
                    "leftTopY": 200,
                    "rightBottomX": 600,
                    "rightBottomY": 600
                }
            ]
        }
    ]
}
```

### 6.9 服务保活

如果配置了 `--callback-url`，服务端会每 30 秒自动发送心跳到平台。

**心跳接口**: `POST /analysis/api/v1/keepAlive`

```json
{
    "devIP": "0.0.0.0",
    "devPort": 22266
}
```

---

## 7. 配置说明

### 7.1 模型配置 (models_config.json)

```json
{
    "models": [
        {
            "name": "model_mengdong_raa_adjusted",
            "algCode": "010161",
            "algDesc": "安全装备检测",
            "ptFile": "model_mengdong_raa_adjusted.pt",
            "engineFile": "model_mengdong_raa_adjusted.engine",
            "labels": {
                "0": "person",
                "1": "safetybelt",
                "2": "mapblu"
            },
            "labelDescriptions": {
                "0": "人",
                "1": "安全带",
                "2": "蓝色安全帽"
            }
        }
    ],
    "server": {
        "host": "0.0.0.0",
        "port": 22266,
        "modelsDir": "/workspace/models_dir",
        "outputDir": "/workspace/output"
    },
    "inference": {
        "scoreThreshold": 0.25,
        "iouThreshold": 0.65,
        "topk": 100,
        "inputWidth": 640,
        "inputHeight": 640
    }
}
```

**配置字段说明:**

| 字段 | 说明 |
|------|------|
| `models[].name` | 模型名称 |
| `models[].algCode` | 算法编码（6位，用于API调用标识） |
| `models[].algDesc` | 算法描述 |
| `models[].ptFile` | PyTorch 模型文件名 |
| `models[].engineFile` | TensorRT Engine 文件名 |
| `models[].labels` | 检测类别英文名（key 为类别ID） |
| `models[].labelDescriptions` | 检测类别中文描述（key 为类别ID） |
| `inference.scoreThreshold` | 置信度阈值（0-1） |
| `inference.iouThreshold` | NMS IoU 阈值（0-1） |
| `inference.topk` | 最大检测框数量 |

### 7.2 算法编码规则

算法编码为6字节编码：

| 位置 | 含义 | 示例 |
|------|------|------|
| 前2位 | 专业编码 | `01` = 变电专业 |
| 中间2位 | 算法类型 | `01` = 人员穿戴, `02` = 工具检测 |
| 后2位 | 具体算法 | `61` = 安全帽检测, `01` = 具体算法 |

当前模型算法编码：

| algCode | 说明 |
|---------|------|
| `010161` | 安全装备检测（人员穿戴 - 安全帽、安全带） |
| `010201` | 钩子与差速器检测 |
| `010301` | 铁塔检测 |
| `010401` | 场景设备检测 |

### 7.3 docker-compose.yml 配置

```yaml
version: "3.8"

services:
  yolov8-api:
    build:
      context: .
      dockerfile: Dockerfile
    image: yolov8-tensorrt-api:latest
    runtime: nvidia
    ports:
      - "22266:22266"         # 映射API端口
    volumes:
      - ./models_dir:/workspace/models_dir   # 挂载模型目录
      - ./output:/workspace/output           # 挂载输出目录
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
      - MODELS_DIR=/workspace/models_dir
      - OUTPUT_DIR=/workspace/output
      - CONFIG_FILE=/workspace/models_config.json
      - SERVER_PORT=22266
      - SERVER_HOST=0.0.0.0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
```

---

## 8. 常见问题与排查

### 8.1 容器无法启动

**现象**: `docker compose up` 失败

```bash
# 检查 GPU 是否可用
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# 查看详细日志
docker compose logs -f

# 检查 nvidia-container-toolkit
dpkg -l | grep nvidia-container-toolkit
```

### 8.2 模型转换失败

**现象**: 日志显示 `WARNING: xxx.pt not found, skipping`

```bash
# 检查模型文件是否正确挂载
docker compose exec yolov8-api ls -la /workspace/models_dir/

# 手动重新转换
docker compose exec yolov8-api python3 /workspace/scripts/convert_models.py \
    --models-dir /workspace/models_dir \
    --config /workspace/models_config.json \
    --fp16
```

### 8.3 API 返回 404 算法编码不存在

**原因**: 请求的 `algCode` 未匹配到已加载的模型

```bash
# 查看已加载的算法
curl -X POST http://localhost:22266/v1/service/abilities

# 确认 algCode 是否正确（如 010161, 010201, 010301, 010401）
```

### 8.4 显存不足 (OOM)

**现象**: 容器启动后立即退出，日志显示 CUDA OOM

```bash
# 检查 GPU 显存使用
nvidia-smi

# 方案1: 减少同时加载的模型数量
# 编辑 models_config.json，只保留需要的模型

# 方案2: 使用显存更大的 GPU
```

### 8.5 视频流拉取失败

**现象**: 视频任务创建成功但无推理结果

```bash
# 检查视频流是否可访问
ffmpeg -i rtsp://192.168.1.100:554/stream1 -frames:v 1 test_frame.jpg

# 检查容器网络
docker compose exec yolov8-api curl -v http://your-video-server/stream
```

### 8.6 Engine 文件不兼容

**现象**: 日志显示 `Engine deserialization failed`

> TensorRT Engine 文件与 GPU 型号绑定。如果更换了 GPU，需要删除旧 `.engine` 文件重新生成：

```bash
# 删除旧 engine 文件
rm models_dir/*.engine models_dir/*.onnx

# 重启容器触发重新转换
docker compose restart
```

### 8.7 如何查看推理结果视频

视频任务推理结果保存为 MP4 文件，可通过以下方式访问：

```bash
# 方式1: 通过 HTTP 下载
curl -O http://localhost:22266/output/task-001.mp4

# 方式2: 直接访问挂载目录
ls -la output/

# 方式3: 浏览器打开
# http://localhost:22266/output/task-001.mp4
```

---

## 快速开始（完整流程）

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd YOLOv8-TensorRT

# 2. 准备模型文件
mkdir -p models_dir
cp /path/to/your/models/*.pt models_dir/

# 3. 构建并启动
docker compose build
docker compose up -d

# 4. 等待模型转换完成（查看日志）
docker compose logs -f
# 等待出现 "YOLOv8-TensorRT API Server" 启动信息

# 5. 验证服务
curl http://localhost:22266/health

# 6. 查看支持的算法
curl -X POST http://localhost:22266/v1/service/abilities

# 7. 测试图片推理
IMAGE_BASE64=$(base64 -w 0 test.jpg)
curl -X POST http://localhost:22266/v1/service/imageTask \
    -H "Content-Type: application/json" \
    -d "{
        \"analyseId\": \"test-001\",
        \"algCode\": \"010161\",
        \"imageData\": \"${IMAGE_BASE64}\"
    }" | python3 -m json.tool

# 8. 创建视频流任务
curl -X POST http://localhost:22266/v1/service/videoTask \
    -H "Content-Type: application/json" \
    -d '{
        "algCode": "010161",
        "command": 1,
        "interval": 60,
        "startTime": "2024-01-01 00:00:00",
        "endTime": "2025-12-31 23:59:59",
        "videoInfo": [{
            "analyseId": "stream-001",
            "devCode": "cam-001",
            "formatType": 0,
            "videoUrl": "rtsp://your-camera-ip/stream"
        }]
    }'
```

---

## 响应码说明

| 响应码 | 说明 |
|-------|------|
| 200 | 响应成功 |
| 400 | 消息格式错误 |
| 403 | 请求被禁止，无权限 |
| 404 | 请求的对象不存在 |
| 500 | 服务异常 |
| 503 | 当前负荷满，稍后再尝试 |
