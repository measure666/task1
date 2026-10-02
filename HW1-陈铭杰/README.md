# task1
为完成福州大学504实验室本科生纳新第一次挑战（研究方向：大模型推理优化和AI for Science）

### 任务要求：

**在本地环境中启动 SGLangOpenAI-compatible 服务，访问 /v1/models，并完成一次推理请求**

### 提交物：

##### 最终需要的pdf文件在文件“./提交的PDF文件”中：[提交的PDF文件](./提交的PDF文件)

1.[操作保存.pdf](./提交的PDF文件/操作保存.pdf)

2.[流程图.pdf](./提交的PDF文件/流程图.pdf)

3.[重点回答.pdf](./提交的PDF文件/重点回答.pdf)

4.[作业感受.pdf](./提交的PDF文件/作业感受.pdf)

5.[AI 使用说明情况.pdf](./提交的PDF文件/AI使用说明情况.pdf)

6.[阅读文献笔记.pdf](./提交的PDF文件/阅读文献笔记.pdf)



**以下为完成步骤：**

### 1.配置环境

**按照题目（在 [第一关挑战.pdf](./第一关挑战.pdf)中）要求在Windows11内置Linux虚拟机WSL2，安装Ubuntu**

参考视频:[Windows跑AI Agent，WSL才是终极答案，别羡慕Mac了，  WSL保姆级全攻略，海量实战教程，一期视频精通]( https://www.bilibili.com/video/BV1pYNm69EPm/?share_source=copy_web&vd_source=61034729728380e19bafb453a33312d7)

**通过anaconda管理不同python环境：**

相关文档：[Anaconda 教程 | 菜鸟教程](https://www.runoob.com/python-qt/anaconda-tutorial.html)

![conda-env](/image/conda-env.png)

**环境1(sglang)**: SGLang 0.5.14

```bash
# 【环境1】跑 SGLang 服务端
conda create -n sglang python=3.12 -y
conda activate sglang
pip install "sglang==0.5.14"
```

**环境2(client)**:Ray 2.56.0 Httpx 0.28.1

```bash
# 【环境2】跑客户端（Ray + 压测脚本）
conda create -n client python=3.12 -y
conda activate client
pip install "ray==2.56.0" "httpx==0.28.1"
```

在sglang环境中使用modelscope包，部署Qwen3-0.6B

建一个文件夹models用来存放Qwen3

```bash
# 利用modelscope包，部署Qwen3-0.6B
pip install modelscope
mkdir -p ~/models
modelscope download --model Qwen/Qwen3-0.6B --local_dir ~/models/Qwen3-0.6B
```

### 2.启动服务并发送推理请求

根据官方文档以及ai帮助[QwenLM/Qwen3: Qwen3 is the large language model series developed by Qwen team, Alibaba Cloud.](https://github.com/QwenLM/Qwen3)

得到以下SGLang配置Qwen3指令：

![sglang-qwen3](/image/sglang-qwen3.png)

```bash
python -m sglang.launch_server \
  --model-path ~/models/Qwen3-0.6B \
  --host 0.0.0.0 --port 30000 \
  --mem-fraction-static 0.70 \
  --context-length 8192 \
  --reasoning-parser qwen3
```

**在环境1(sglang)中启动 SGLangOpenAI-compatible 服务：**

![sglang-start](/image/sglang-start.png)

出现以上内容表示启动成功

**在环境2(client)中访问 /v1/models：**

![client-v1-models](/image/client-v1-models.png)

出现以上内容表示访问成功

接着

**发送一次推理请求**

![client-chat](/image/client-chat.png)

**推理内容：**

**用户**：“用一句话介绍一下你自己”

**模型**：\u6211\u662fAI\u52a9\u624b\uff0c\u53ef\u4ee5\u5e2e\u4f60\u89e3\u7b54\ u95ee\u9898\u6216\u63d0\u4f9b\u652f\u6301\u3002（我是 AI 助手，可以帮 你解答问题或提供支持。）

出现以上内容表示成功，并截图保存为截图1放入[操作保存.pdf](./提交的PDF文件/操作保存.pdf)

### 3.Mooncake trace 采样 workload

克隆[kvcache-ai/Mooncake: Mooncake is the serving platform for Kimi, a leading LLM service provided by Moonshot AI.](https://github.com/kvcache-ai/Mooncake)中Mooncake仓库到环境2(client)中

接着从 Mooncake FAST’25 trace的arxiv-trace/mooncake_trace.jsonl 中采样 20条请求记录。根据 input_length 和 output_length 构造 synthetic prompt，用 Poisson 到达过程生成请求时间，并将这组小 workload 发送到 SGLang。记录每个请求的 input tokens、output tokens、status、TTFT 、latency。

运行程序[03_mooncake_bench.py](./code/03_mooncake_bench.py)(由claude生成)得到以下数据：![cilent-trace](/image/cilent-trace.png)

截图保存为截图 2放入[操作保存.pdf](./提交的PDF文件/操作保存.pdf)

### 其它任务内容放在[提交的PDF文件](./提交的PDF文件)

