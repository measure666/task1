# 第三次挑战

### 硬件：

通过AutoDL租了一块GPU,**单卡切 4 个 SGLang 实例**

|     GPU      |            **NVIDIA GeForce RTX 3090(24GB)**            |
| :----------: | :-----------------------------------------------------: |
|   **CPU**    | **12vCPU Intel(R) Xeon(R) Platinum 8255C CPU @2.50GHz** |
|   **内存**   |                        **43GB**                         |
| **镜像环境** | **Minconda conda3 Python 3.10(ubuntu22.04) CUDA 11.8**  |
|   **硬盘**   |               **系统盘:30GB 数据盘:50GB**               |

### 软件：

|        **软件**        |     **安装环境**     |
| :--------------------: | :------------------: |
|   **python 3.12.15**   | **sglang/ray-serve** |
|   **sglang 0.5.14**    |      **sglang**      |
|    **torch 2.11.0**    |      **sglang**      |
| **transformers 5.8.1** |      **sglang**      |
| **modelscope 1.40.1**  |      **sglang**      |
|    **numpy 2.3.5**     |      **sglang**      |
| **ray[serve] 2.56.0**  |    **ray-serve**     |
|      **aiohttp**       |    **ray-serve**     |
|  **protobuf 6.33.5**   |    **ray-serve**     |

模型：Qwen/Qwen3-0.6B

### 安装方法:

打开AutoDL的容器实例，利用JupyterLab，打开终端，建立两个conda环境：

```bash
# ---- 环境 1：SGLang 服务端 ----
conda create --prefix /root/autodl-tmp/envs/sglang python=3.12 -y
conda activate /root/autodl-tmp/envs/sglang
pip install "sglang==0.5.14" modelscope

# ---- 环境 2：客户端（Ray Serve + 流量发生器）----
conda create --prefix /root/autodl-tmp/envs/ray-serve python=3.12 -y
conda activate /root/autodl-tmp/envs/ray-serve
pip install "ray[serve]==2.56.0" "aiohttp==3.14.3" "protobuf==6.33.5"
```

**更改CUDA版本**（这部分由claude修改）**：**

由于系统自带的nvcc是11.8，而 FlashInfer 做 JIT 编译时需要 ≥12，否则报`CUDA versions below 12 are not supported`。`cuda-keyring` 的 deb 包在本机装不上（`/etc/apt/sources.list.d/` 一直是空的），改用手写源：

```bash
echo "deb [trusted=yes] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
  > /etc/apt/sources.list.d/cuda.list
apt update && apt install -y cuda-toolkit-12-4
ln -sfn /usr/local/cuda-12.4 /usr/local/cuda     
```

下载Qwen模型：

```bash
modelscope download --model Qwen/Qwen3-0.6B --local_dir /root/autodl-tmp/models/Qwen3-0.6B
```

克隆仓库：

```bash
git clone https://github.com/Zirkland/26fall-HW-data.git /root/autodl-tmp/26fall-HW-data
```

### 启动和回放：

运行程序时，请将文件夹以下放在同一个目录下：

|       **/envs**        |    **放两个conda环境的文件**     |
| :--------------------: | :------------------------------: |
| **/models/Qwen3-0.6B** |         **放模型的文件**         |
|  **/26fall-HW-data**   |       **要求clone的仓库**        |
|        **/hw3**        | **放入/src/target3中的三个脚本** |

例如我放在AutoDL容器固定的数据盘自带目录/root/autodl-tmp，以下以我这个目录举例

如果你也在AutoDL容器中复现，只要按上面的安装方法后，建立/hw3文件夹，将脚本放入/hw3后即可。

#### 脚本用途：

| 脚本          | 作用                                           |
| ------------- | ---------------------------------------------- |
| `app.py`      | 搭 Ray 集群、部署 4 个 Replica、按轮次切换路由 |
| `router_d.py` | D 组的自定义路由类，只被 `app.py` 引用         |

**建立/hw3文件夹，将脚本放入/hw3，并且保证与/26fall-HW-data，/models/Qwen3-0.6B，/envs放在同一目录下**

#### 启动四个SGLang后端：

```bash
#进入sglang环境
conda activate /root/autodl-tmp/envs/sglang

for p in 30000 30001 30002 30003; do
  nohup /root/autodl-tmp/envs/sglang/bin/python -m sglang.launch_server \
    --model-path /root/autodl-tmp/models/Qwen3-0.6B \
    --host 0.0.0.0 --port $p \
    --mem-fraction-static 0.40 \
    --max-total-tokens 20000 \
    --context-length 8192 \
    --disable-cuda-graph \
    > /root/sglang_$p.log 2>&1 &
  # 逐个启动：等前一个能响应再起下一个
  until curl -sf http://127.0.0.1:$p/get_server_info >/dev/null; do sleep 3; done
done
```

待程序出现例如以下字样即代表成功：

```bash
[1] 371837
启动 :30000 pid=371837
  :30000 就绪
[2] 372076
启动 :30001 pid=372076
  :30001 就绪
[3] 372320
启动 :30002 pid=372320
  :30002 就绪
[4] 372563
启动 :30003 pid=372563
  :30003 就绪
```

#### 启动Ray Serve运行A-D:

每一轮都按以下顺序执行：

1. 停止上一轮 Ray Serve 应用，使用本轮配置重新部署。
2. 分别调用四个 SGLang 后端的 `POST /flush_cache`，确认全部成功。
3. 启动流量发生器。它会先顺序完成 64 条预热请求，再发送 2048 条测量请求。
4. 检查进程退出状态和 `summary.json`，确认本轮有效。

```bash
#进入放脚本的文件夹，具体路径以自己为准，以下指令均在本路径下输入
cd /root/autodl-tmp/hw3
```

**A:**

在终端1：

```bash
#A:P2C,max_ongoing=5下
python app.py --round A_default
```

待程序出现例如以下字样即代表成功：

```bash
[部署] 本机生效的写法：serve.run(app, name=, route_prefix=)
[放置检查] 4 个副本 -> 后端端口 [30000, 30001, 30002, 30003]

[就绪] 部署完成。HTTP 口在 127.0.0.1:8000，现在到**另一个终端**跑 run_workload.py。
       本终端 Ctrl-C 结束本轮并拆掉集群。
```

另开一个终端2，并保持终端1不关闭

在终端2：

```bash
for p in 30000 30001 30002 30003; do
    echo -n "$p: "
    curl -s -X POST localhost:$p/flush_cache
    echo
  done
```

调用四个 SGLang 后端的 `POST /flush_cache`

接着：

```bash
cd /root/autodl-tmp/26fall-HW-data/workloads/hw2/target3-routing-policies
python run_workload.py \
    --policy serve \
    --run-name A_default_run1 \
    --router-name p2c \
    --max-ongoing-requests 5 \
    --base-url http://127.0.0.1:8000 \
    --workload mooncake_prefix_workload_v2_seed2026.jsonl \
    --max-in-flight 2048 \
    --timeout 600 \
    --output-dir results/target3/A_default
```

产物会生成在26fall-HW-data/workloads/hw2/target3-routing-policies/result中

成功执行后生成四个文件：

| 文件           | 内容                                                         |
| -------------- | ------------------------------------------------------------ |
| `config.json`  | 命令、模型、路由器、并发参数、负载 SHA-256、校验状态和自定义备注 |
| `warmups.csv`  | 64 条预热请求的结果                                          |
| `requests.csv` | 2048 条测量请求的逐请求结果                                  |
| `summary.json` | 全程、各流量阶段和各热度层级的汇总指标                       |

**B:**

**max-ongoing-requests=8 ：**

在终端1：

按ctrl+c终止Ray serve

并输入：

```bash
python app.py --round B_cand8
```

在终端2：

```bash
for p in 30000 30001 30002 30003; do
    echo -n "$p: "
    curl -s -X POST localhost:$p/flush_cache
    echo
  done
```

调用四个 SGLang 后端的 `POST /flush_cache`

接着：

```bash
python run_workload.py \
    --policy serve \
    --run-name B_cand8_run1 \
    --router-name p2c \
    --max-ongoing-requests 8 \
    --base-url http://127.0.0.1:8000 \
    --workload mooncake_prefix_workload_v2_seed2026.jsonl \
    --max-in-flight 2048 \
    --timeout 600 \
    --output-dir results/target3/B_candidates/candidate-1
```

产物会生成在26fall-HW-data/workloads/hw2/target3-routing-policies/result中

成功执行后生成四个文件：

| 文件           | 内容                                                         |
| -------------- | ------------------------------------------------------------ |
| `config.json`  | 命令、模型、路由器、并发参数、负载 SHA-256、校验状态和自定义备注 |
| `warmups.csv`  | 64 条预热请求的结果                                          |
| `requests.csv` | 2048 条测量请求的逐请求结果                                  |
| `summary.json` | 全程、各流量阶段和各热度层级的汇总指标                       |

**max-ongoing-requests=32：** 

在终端1：

按ctrl+c终止Ray serve

并输入：

```bash
python app.py --round B_cand32
```

在终端2：

```bash
for p in 30000 30001 30002 30003; do
    echo -n "$p: "
    curl -s -X POST localhost:$p/flush_cache
    echo
  done
```

调用四个 SGLang 后端的 `POST /flush_cache`

接着：

```bash
python run_workload.py \
    --policy serve \
    --run-name B_cand32_run1 \
    --router-name p2c \
    --max-ongoing-requests 32 \
    --base-url http://127.0.0.1:8000 \
    --workload mooncake_prefix_workload_v2_seed2026.jsonl \
    --max-in-flight 2048 \
    --timeout 600 \
    --output-dir results/target3/B_candidates/candidate-2
```

产物会生成在26fall-HW-data/workloads/hw2/target3-routing-policies/result中

成功执行后生成四个文件：

| 文件           | 内容                                                         |
| -------------- | ------------------------------------------------------------ |
| `config.json`  | 命令、模型、路由器、并发参数、负载 SHA-256、校验状态和自定义备注 |
| `warmups.csv`  | 64 条预热请求的结果                                          |
| `requests.csv` | 2048 条测量请求的逐请求结果                                  |
| `summary.json` | 全程、各流量阶段和各热度层级的汇总指标                       |

**C:**

根据B的两次候选值，得出max_on_going=32时更优，三项主要指标都有数量级优势，而缓存命中率与后端分布几乎不动

详细数据见result中B的两次summary.csv

所以C的max_on_going设置成32

在终端1：

按ctrl+c终止Ray serve

并输入：

```bash
python app.py --round C_affinity --max-ongoing 32
```

在终端2：

```bash
for p in 30000 30001 30002 30003; do
    echo -n "$p: "
    curl -s -X POST localhost:$p/flush_cache
    echo
  done
```

调用四个 SGLang 后端的 `POST /flush_cache`

接着：

```bash
python run_workload.py \
    --policy serve \
    --run-name C_affinity_run1 \
    --router-name consistent_hash \
    --max-ongoing-requests 32 \
    --base-url http://127.0.0.1:8000 \
    --workload mooncake_prefix_workload_v2_seed2026.jsonl \
    --max-in-flight 2048 \
    --timeout 600 \
    --output-dir results/target3/C_affinity
```

产物会生成在26fall-HW-data/workloads/hw2/target3-routing-policies/result中

成功执行后生成四个文件：

| 文件           | 内容                                                         |
| -------------- | ------------------------------------------------------------ |
| `config.json`  | 命令、模型、路由器、并发参数、负载 SHA-256、校验状态和自定义备注 |
| `warmups.csv`  | 64 条预热请求的结果                                          |
| `requests.csv` | 2048 条测量请求的逐请求结果                                  |
| `summary.json` | 全程、各流量阶段和各热度层级的汇总指标                       |

**D:**

根据B的两次候选值，得出max_on_going=32时更优，三项主要指标都有数量级优势，而缓存命中率与后端分布几乎不动

详细数据见result中B的两次summary.csv

所以D的max_on_going设置成32

改进路由脚本：`router_d.py`（详细改进见 report.pdf）

在终端1：

按ctrl+c终止Ray serve

并输入：

```bash
python app.py --round C_affinity --max-ongoing 32
```

在终端2：

```bash
for p in 30000 30001 30002 30003; do
    echo -n "$p: "
    curl -s -X POST localhost:$p/flush_cache
    echo
  done
```

调用四个 SGLang 后端的 `POST /flush_cache`

接着：

```bash
python run_workload.py \
    --policy serve \
    --run-name D_improved_run1 \
    --router-name prefix_affinity_load_aware \
    --max-ongoing-requests 32 \
    --base-url http://127.0.0.1:8000 \
    --workload mooncake_prefix_workload_v2_seed2026.jsonl \
    --max-in-flight 2048 \
    --timeout 600 \
    --metadata preflush=cold \
    --output-dir results/target3/D_improved
```

产物会生成在26fall-HW-data/workloads/hw2/target3-routing-policies/result中

成功执行后生成四个文件：

| 文件           | 内容                                                         |
| -------------- | ------------------------------------------------------------ |
| `config.json`  | 命令、模型、路由器、并发参数、负载 SHA-256、校验状态和自定义备注 |
| `warmups.csv`  | 64 条预热请求的结果                                          |
| `requests.csv` | 2048 条测量请求的逐请求结果                                  |
| `summary.json` | 全程、各流量阶段和各热度层级的汇总指标                       |

### 汇总实验结果

```bash
python compare_runs.py \
  --baseline A \
  --run A=results/A_default/summary.json \
  --run B1=results/B_candidates/candidate-1/summary.json \
  --run B2=results/B_candidates/candidate-2/summary.json \
  --run C=results/C_affinity/summary.json \
  --run D=results/D_improved/summary.json \
  --output results/comparison.json
```

## 结果目录与报告表格的对应关系

```text
HW3-陈铭杰/
├── README.md                                  ← 本文档
├── report.pdf
├── AI 使用说明情况（第三次挑战）.pdf
├── src/
│   ├── target1/                              
│   └── target3/                               ← 本关代码，全部在这
│       ├── app.py
│       └── router_d.py
└── results/
    ├── target1/                              
    ├── comparison.json                        compare_runs.py 的输出
    └── target3/
        ├── A_default/                         {config,warmups,requests}.csv + summary.json
        ├── B_candidates/
        │   ├── candidate-1/                   max_ongoing_requests = 8
        │   └── candidate-2/                   max_ongoing_requests = 32
        ├── C_affinity/                        ConsistentHashRouter, num_fallback_replicas=0
        └── D_improved/                        D 自定义路由
```

每轮目录里都是 `run_workload.py` 的四件套：`config.json` / `warmups.csv` / `requests.csv` /`summary.json`

### 主表见report.pdf

各个表格字段含义：

#### `config.json`

| 字段                                                         | 含义                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `schema_version`                                             | 注意 `summary.json` 的 `schema_version` 是 `2`，两者不同源，别混用 |
| `created_at_utc`                                             | 本轮开始时刻（UTC，带时区）                                  |
| `run_name`                                                   | `--run-name` 给的名字。交付的五轮依次是 `A_default_run1` / `B_cand8_run1` / `B_cand32_run1` / `C_affinity_run1` / `D_improved_run1`。 |
| `client_policy`                                              | 恒为 `serve`，指流量发生器从 Ray Serve 的 HTTP 口打进去      |
| `model`                                                      | `Qwen/Qwen3-0.6B`，与四个 SGLang 后端的 `--model-path` 同源  |
| `router_name`                                                | `--router-name` 给的名字，**只被记录、不生效**。五轮分别是 `p2c` / `p2c` / `p2c` / `consistent_hash` / `prefix_affinity_load_aware` —— 它们描述的是意图，真正生效的路由在 `app.py` 的 `ROUNDS` 里 |
| `max_ongoing_requests`                                       | 同上，**只被记录**；真正生效的是 `app.py --round` 选中的那份 `serve.deployment(max_ongoing_requests=…)`。本交付五轮的记录值是 5 / 8 / 32 / 32 / 32，**恰好**都等于对应轮次的部署值 —— 这是每次刻意传成一样的，字段本身不保证二者相等 |
| `base_url`                                                   | `http://127.0.0.1:8000`，即 Ray Serve HTTP Proxy             |
| `session_header`                                             | `X-Session-Id`；它的值就是 `prefix_family` —— C/D 的前缀亲和路由全靠它 |
| `max_in_flight`                                              | 压测端自己的并发闸（2048 = 等于不设限）。**和 `max_ongoing_requests` 是两回事**：一个是客户端闸，一个是服务端每副本的排队上限 |
| `timeout_s`                                                  | 单请求超时 600 s（脚本默认 300，会把尾部的长排队误判成失败） |
| `token_base` / `token_span` / `token_salt`                   | 生成 token ID 的三个常量；同种子下逐字节可复现（`run_workload.py` 的 `TokenBuilder`） |
| `sampling_seed`                                              | 恒 `2026`，写进 `sampling_params.sampling_seed`              |
| `expected_replicas` / `expected_backends` / `expected_nodes` | 都是 `4`，用于校验                                           |
| `require_routing_headers`                                    | `true`：响应里缺 `X-Ray-Replica-ID` / `X-Ray-Node-ID` / `X-SGLang-Backend` 就记进 `error`，该请求计入 `failed` |
| `workload`                                                   | 本轮所用负载的**绝对路径**                                   |
| `workload_sha256` / `expected_workload_sha256` / `workload_verified` | 实际哈希 / 期望哈希 / 是否相符。不相符时脚本**直接抛错拒绝开跑**，所以这份 `config.json` 只要存在，就说明负载是原版 |
| `warmup_requests` / `measured_requests`                      | `64` / `2048`                                                |
| `metadata`                                                   | `--metadata k=v` 原样存。D 轮填了 `preflush=cold`，把「开跑前四个后端已 flush」这个前提写进产物（它不在 `compare_runs.py` 的比对字段里，不影响跨轮可比性） |
| `command`                                                    | `[sys.executable, *sys.argv]` —— **只记录 `run_workload.py` 自己那一次调用的完整 argv** |

#### `warmups.csv` / `requests.csv` 

| 列                                       | 含义                                                         |
| ---------------------------------------- | ------------------------------------------------------------ |
| `request_id`                             | `requests.csv` 为 `0…2047`；`warmups.csv` 为 `warmup-000…warmup-063` |
| `phase`                                  | `measured` / `warmup`                                        |
| `prefix_family`                          | 前缀族 ID（12 位十六进制），同时是请求头 `X-Session-Id` 的值 |
| `popularity_tier`                        | 该族的热度档：`superhot`(1 族) / `hot`(7 族) / `warm`(24 族) / `cold`(32 族) |
| `traffic_phase`                          | 该请求落在到达过程的哪一段：`steady`(512 条) / `burst`(1024 条) / `recovery`(512 条) |
| `family_request_count`                   | 该族在本轮被请求的总次数                                     |
| `affinity_backend`                       | 负载为该族**指定**的后端（`0…3`）—— C/D 要逼近的目标映射；A/B 不看这个字段 |
| `source_line`                            | 该族在**原始 trace 文件**里的行号（1-based，`sample_prefix_families.py:40` 打的） |
| `planned_send_s`                         | 计划发送时刻，相对本轮 t0（秒），来自负载的 `arrival_offset_s` |
| `actual_send_s`                          | 实际发出请求的时刻，相对本轮 t0（秒）                        |
| `dispatch_lag_s`                         | `actual_send_s − planned_send_s`，客户端自身的调度误差       |
| `client_queue_s`                         | 抢压测端并发闸的等待时间（`started − ready_at`），即 `max_in_flight` 造成的排队 |
| `input_tokens`                           | 请求的输入长度 = 负载给的 token ID 个数                      |
| `requested_output_tokens`                | 负载计划的输出长度；它被**同时**写进 `max_new_tokens` 与 `min_new_tokens` |
| `prompt_tokens`                          | 服务端在**首个带 `output_ids` 的 chunk** 上报告的 `prompt_tokens` |
| `cached_tokens`                          | 同一个 chunk 上报告的 `cached_tokens`，即 radix cache 实际复用掉的输入长度 |
| `output_tokens`                          | 客户端数到的实际生成数 = `len(final_output_ids)`             |
| `finish_reason`                          | 服务端 `meta_info.finish_reason` 的 JSON 串                  |
| `status_code`                            | HTTP 状态码                                                  |
| `error`                                  | 出错信息。**非空即失败** —— `successful` 只统计 `status_code == 200 且 error == ""` 的行 |
| `ttft_s`                                 | 首个**带 `output_ids` 的 chunk** 到达时刻 − `started`。注意不是「收到第一个字节」 |
| `tpot_s`                                 | `(last_token_at − first_token_at) / (output_tokens − 1)`，**用逐 chunk 时间戳算的** |
| `latency_s`                              | `finished − started`                                         |
| `finished_s`                             | 请求读完的时刻，相对本轮 t0                                  |
| `replica_id` / `ray_node_id` / `backend` | 取自响应头 `X-Ray-Replica-ID` / `X-Ray-Node-ID` / `X-SGLang-Backend` |

#### `summary.json`

| 字段                                                         | 含义                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `schema_version`                                             | `2`                                                          |
| `run_name` / `policy` / `router_name` / `max_ongoing_requests` / `model` / `workload_sha256` / `max_in_flight` / `token_base` / `token_span` / `token_salt` / `expected_*` | 从 `config.json` 抄一份。`compare_runs.py` 就是靠这些判定「跨轮的受控量是否一致」 |
| `requests` / `successful` / `failed` / `incomplete`          | 有效轮次要求 `2048 / 2048 / 0 / 0`。`failed = requests − successful`；`incomplete` 单指 `output_tokens != requested_output_tokens` 的行（`ignore_eos=true` 下应为 0） |
| `warmup_successful` / `warmup_failed` / `warmup_incomplete`  | 要求 `64 / 0 / 0`                                            |
| `validation_errors`                                          | `expected_replicas` / `expected_backends` / `expected_nodes` 与实际分布项数不符时的报错列表；必须为 `[]` |
| `elapsed_s`                                                  | 从第一个请求发出，到最后一个请求读完                         |
| `planned_duration_s`                                         | 负载里最后一个测量请求的计划到达时刻 = `29.538768` s         |
| `offered_rate_rps`                                           | `(requests − 1) / planned_duration_s` = `69.2988`。**分子是 N−1**（脚本原样如此），不是 N |
| `throughput_rps`                                             | `successful / elapsed_s`                                     |
| `prompt_tokens_total`                                        | 五轮恒为 `3,233,280`（2048 个请求的输入之和）                |
| `cached_tokens_total` / `computed_prefill_tokens_total`      | 命中复用掉的 / 真正还要算的输入 token 数；两者之和恒等于 `prompt_tokens_total` |
| `cache_hit_rate`                                             | `cached_tokens_total / prompt_tokens_total`，**token 级聚合**，部分命中按比例计入（不是按请求数算的命中率） |
| `ttft_s` / `tpot_s` / `latency_s` / `dispatch_lag_s` / `client_queue_s` | 五个 dict，键恒为 `mean` / `p50` / `p95` / `p99`；**单位为秒**；只对 `successful` 行统计，并自动跳过 `None`（即预热的 `tpot_s`） |
| `replica_distribution` / `node_distribution` / `backend_distribution` | 按 `successful` 行计数，各 4 项                              |
| `backend_token_distribution`                                 | 同一个分组的 token 加权版；每项含 `requests` / `input_tokens` / `cached_tokens` / `requested_output_tokens` / `output_tokens` |
| `warmup_replica_distribution` / `warmup_node_distribution` / `warmup_backend_distribution` | 预热行的同名分布。预热**故意不打散**，本来就该不均，**不要**拿它判断负载均衡 |
| `traffic_phases`                                             | `steady` / `burst` / `recovery` 三段各自的 `requests` / `successful` / `cache_hit_rate` / `ttft_s` / `latency_s` / `backend_distribution`。**没有** `tpot_s` / `dispatch_lag_s` / `client_queue_s` 三项。报告表 6 的来源 |
| `popularity_tiers`                                           | `cold` / `warm` / `hot` / `superhot` 四档的同一组字段。报告表 5 的原始数据 |