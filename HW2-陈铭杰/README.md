# 第二次挑战

### 硬件：

|    **GPU**     | **NVIDIA GeForce RTX 5070 Laptop GPU（8 GB 显存）** |
| :------------: | :-------------------------------------------------: |
| **NVIDIA驱动** |                     **582.05**                      |
|    **CPU**     |                      **32核**                       |
|    **内存**    |                      **16GB**                       |
|  **操作系统**  |                 **Ubuntu on WSL2**                  |

### 软件：

| 软件          | 版本    | 安装环境      |
| ------------- | ------- | ------------- |
| python        | 3.12.14 | sglang/client |
| SGLang        | 0.5.14  | sglang        |
| sglang-kernel | 0.4.4   | sglang        |
| PyTorch       | 2.11.0  | sglang        |
| transformers  | 5.8.1   | sglang        |
| modelscope    | 1.40.1  | sglang        |
| Ray           | 2.56.0  | client        |
| aiohttp       | 3.14.3  | client        |

### 安装方法：

**conda自带python，创建环境时令版本为3.12即可**

```bash
#创建sglang和client环境并使python版本为3.12，详情可见第一次挑战的README
conda create -n sglang python=3.12 -y
conda create -n client python=3.12 -y 
```

##### 通过pip安装各种软件包：

**在sglang环境中：**

```bash
pip install "sglang==0.5.14"
pip install modelscope
```

**在client环境中：**

```bash
pip install "ray==2.56.0" "httpx==0.28.1" "aiohttp"
```

其中aiohttp主要用于实现bench.py脚本

### 启动：

开启终端，打开wsl的Ubuntu启动环境sglang:

```
conda activate sglang
```

启动服务：

```
python -m sglang.launch_server \
  --model-path ~/models/Qwen3-0.6B \
  --host 0.0.0.0 --port 30000 \
  --mem-fraction-static 0.70 \
  --context-length 8192 \
  --reasoning-parser qwen3
```

出现以下内容即代表服务启动成功：

![sglang-start-success](/image/sglang-start-success.png)

保持这个终端不要关闭!

### 回放命令：

**1.生成负载**

开启另一个终端，打开wsl的Ubuntu启动环境并运行**build_loads.py(**由claude code生成)生成随机的token:

```
conda activate sglang
cd "/mnt/../HW2-陈铭杰" #下载到windows系统，以你下载到的位置为主
python src/target1/build_loads.py --out src/target1/loads.json
```

生成结果loads.json

![loads_json](/image/loads_json.png)

由于令SEED=2026(用于random函数，作为伪随机)任何机器上都会生成完全相同的 64 条负载

我已经生成好了，可以直接使用，放在src/target1中：[loads.json](./src/target1/loads.json)

也可以更改SEED的值来生成其它结果进行验证

(确保loads.json放在src\target1中，否则代码无法复现)

2.**运行压测**

**运行脚本：bench.py** (由claude code生成)

```
conda activate client
cd "/mnt/../HW2-陈铭杰" #下载到windows系统，以你下载到的位置为主，与第一步一样
```

第一次运行：

```
python src/target1/bench.py --run run-1
```

第二次运行：

```
python src/target1/bench.py --run run-2
```

......

第五次运行：

```
python src/target1/bench.py --run run-5
```

**一共运行五次，生成五次结果，重复实验以验证稳定性**

**共享前缀结果**放在results/target1/shared_prefix/

**分散前缀结果**放在results/target1/dispersed_prefix/

**两组对照结果**放在results/target1/         (后缀有summary_all字样的 )

如若要增加实验次数，请将run-N中的N改为你要测试的次数，

在我数据的基础上增加(run-6,run-7...)，以免我的数据被覆盖

### 脚本用途说明：

#### build_loads.py：

**输入：无**

**输出：**

​	**loads.json**含三个数组(按照题目要求):

​	**shared_prefix：**2048 个共享 token（供预热请求使用）

​	**shared_requests：**32 条 × 2112 token，前 2048 个逐元素相同

​	**dispersed_requests：**32 条 × 2112 token，首 token 两两不同(不进行前缀预热)

其中，token id 取自[0, 151643),可天然避开全部特殊 token；写文件前内置自检,不合格直接报错不产出

实验参数：

![build_loads_1](/image/build_loads_1.png)

详细代码可见，配有注释[build_loads.py](.\src\target1\build_loads.py)

### `bench_py`:

**输入：loads.json** (确保loads.json放在src\target1中，否则代码无法复现)

**输出：**两组负载的 32 条测量请求(放在目录/results/target1/)

​	**1) shared_prefix/run-N/requests.csv   (共享前缀组的逐一请求结果)**

​	**2) shared_prefix/run-N/summary.csv   (共享前缀组的汇总)**

​	**3) shared_prefix/run-N/requests.csv  （分散前缀组的逐一请求结果)**

​	**4) dispersed_prefix/run-N/summary.csv   (分散前缀组的汇总)**

​	**5) run-N-summary_all.csv   (两组并排,即主表)**

**配置：**根据题目要求设置参数：

![bench_setting](/image/bench_setting.png)

**接口：**

​	**1)通过原生 `POST /generate` 接口发送 input_ids(由loads.json提供)，并使用流式响应**

![generate](/image/generate.png)

​	**2)先用短请求完成服务预热,每组测量前等待已有请求结束，调用 `POST /flush_cache` 并确认成功**

![flush_cache](/image/flush_cache.png)

**实验流程**

```text
服务预热（5 条短请求，不计入结果）
── 共享前缀组 ──
  POST /flush_cache 并确认返回 200
  前缀预热 1 条（含 2048 共享前缀，不计入结果）
  32 条测量请求，并发上限 8，流式响应
── 分散前缀组 ──
  POST /flush_cache 并确认返回 200
  不做前缀预热
  32 条测量请求，并发上限 8，流式响应
```

**说明：**脚本会自动完成：服务预热 → 清空缓存 → 前缀预热 → 发送 32 条测量请求，
两组依次执行

详细代码可见，配有注释[bench.py](.\src\target1\bench.py)

### 结果目录与报告表格的对应关系

#### 结果目录：

```text
results/target1/
├── shared_prefix/
│   ├── run-1/
│   │   ├── requests.csv      32 条测量请求的逐请求明细
│   │   └── summary.csv       该组汇总（1 行）
│   ├── run-2/
│   ├── ...                   （run-1 ~ run-5，共重复 5 次，结构相同）
│   └── run-5/
├── dispersed_prefix/
│   ├── run-1/
│   │   ├── requests.csv
│   │   └── summary.csv
│   ├── ...
│   └── run-5/
├── run-1-summary_all.csv     两组对照表（每组 1 行）
├── ...
└── run-5-summary_all.csv
```

同一配置共进行 5 次重复实验，`run-N/` 即第 N 轮。每轮开始前均重新 `flush_cache`，
各轮相互独立；报告中的稳定性对照即基于这 5 轮。

### `requests.csv` 字段

| 字段                  | 含义                                                 |
| --------------------- | ---------------------------------------------------- |
| `group`               | 组别（`shared_prefix` / `dispersed_prefix`）         |
| `index`               | 请求序号 0~31                                        |
| `ok`                  | 是否成功（1/0）                                      |
| `prompt_tokens`       | 输入 token 数                                        |
| `cached_tokens`       | 命中的缓存 token 数                                  |
| `real_prefill_tokens` | 实际需 Prefill 的 token 数（= 前两者之差）           |
| `completion_tokens`   | 输出 token 数                                        |
| `ttft_ms`             | 首 token 延迟（毫秒）                                |
| `tpot_ms`             | 每输出 token 延迟（毫秒）                            |
| `e2e_ms`              | 客户端测得的端到端延迟（毫秒）                       |
| `server_e2e_ms`       | 服务端上报的 `meta_info.e2e_latency`（用于交叉核对） |
| `error`               | 出错信息（成功时为空）                               |

### `summary.csv` / `run-N-summary_all.csv` 字段

三个文件的分工：

| 文件                                 | 内容                   |
| ------------------------------------ | ---------------------- |
| `shared_prefix/run-N/summary.csv`    | 共享前缀组的汇总       |
| `dispersed_prefix/run-N/summary.csv` | 分散前缀组的汇总       |
| **`run-N-summary_all.csv`**          | 两组并排 —— 即**主表** |

前两个各描述"一组"，只有 `run-N-summary_all.csv` 构成"对照"，

#### 因此**主表取 `run-N-summary_all.csv`**。

其列与题目要求的主表内容逐项对应：

| 字段                        | 含义                         |
| --------------------------- | ---------------------------- |
| `success_rate`              | 成功率                       |
| `throughput_rps`            | 吞吐量                       |
| `cache_hit_rate`            | 缓存命中率                   |
| `total_real_prefill_tokens` | 实际执行 Prefill 的 token 数 |
| `ttft_p50` / `ttft_p95`     | TTFT 的 p50 / p95            |
| `tpot_p50` / `tpot_p95`     | TPOT 的 p50 / p95            |
| `e2e_p50` / `e2e_p95`       | 端到端延迟的 p50 / p95       |

其余字段为支撑数据，可用于验证上表各值的来源：

| 字段                  | 含义                               |
| --------------------- | ---------------------------------- |
| `n_total`             | 计划发送的测量请求条数（= 32）     |
| `n_success`           | 实际成功的条数（即 `ok=1` 的行数） |
| `wall_time_s`         | 该组 32 条测量请求的墙钟耗时（秒） |
| `total_prompt_tokens` | 32 条输入的 token 总数             |
| `total_cached_tokens` | 32 条命中缓存的 token 总数         |

关于 `wall_time_s`：其计时范围仅覆盖 32 条测量请求，不含服务预热、`flush_cache`
与前缀预热（见 `bench.py` 的 `run_group()`）。若有额外开销被计入，两组被计入的内容不同，
吞吐量即不可比。

##### 对应关系：

​	**1) shared_prefix/run-N/requests.csv   (共享前缀组的逐一请求结果)**

​	**2) shared_prefix/run-N/summary.csv   (共享前缀组的汇总)**

​	**3) shared_prefix/run-N/requests.csv  （分散前缀组的逐一请求结果)**

​	**4) dispersed_prefix/run-N/summary.csv   (分散前缀组的汇总)**

​	**5) run-N-summary_all.csv   (两组并排,即主表)**



