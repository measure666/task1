# -*- coding: utf-8 -*-
"""第三关 target3：4 个独立 SGLang 后端 + 1 head + 4 逻辑 worker 的 Ray Serve 应用。

拓扑
    Client --POST /generate--> Serve HTTP Proxy(:8000)
           --> Replica 0..3（每个独占一个逻辑节点）
           --> SGLang 30000..30003（一一对应）

四组路由共用这一份代码，只换参数，保证四轮之间除了被比较的变量以外
完全相同：

    A  python app.py --round A_default
    B  python app.py --round B_cand8                      # 候选 1
    B  python app.py --round B_cand32                     # 候选 2
    C  python app.py --round C_affinity  --max-ongoing <B 胜出值>
    D  python app.py --round D_improved  --max-ongoing <B 胜出值>

进程前台常驻，Ctrl-C 才拆掉 Ray 集群（每轮重新建集群 = 每轮冷启动，
四轮起点一致，比较更公平）。流量发生器请在另一个终端跑。
"""

import argparse
import os
import re
import sys
import time

import aiohttp
import ray
from ray import serve
from ray.cluster_utils import Cluster
from ray.serve.config import RequestRouterConfig
from starlette.responses import Response, StreamingResponse

# ----------------------------------------------------------------------
# 让自定义路由类在任何启动方式下都能被 Ray worker import
# ----------------------------------------------------------------------
# RequestRouterConfig.request_router_class 是「点分路径字符串」，
# Ray 在 **HTTP Proxy 进程**（它本身是个 Ray worker 进程）里用 importlib
# 解析它。所以 router_d 必须出现在那个进程的 sys.path 上，否则 D 组部署时
# 直接失败 —— 而它在哪个进程里失败，驱动进程这边是看不到的。
#
# 三件事都不能指望：
#   1) 启动时的 cwd：用 `python /绝对路径/app.py` 从别处启动时，cwd 就不是这里；
#   2) 外层 shell 的 PYTHONPATH：换一台机器就没了；
#   3) 环境里手工塞的 .pth 文件：那是本机调试的残留，不会跟着 src/ 一起交付。
#
# 实测（Ray 2.56，本地 Cluster）：在 Cluster()/ray.init() **之前**把本目录写进
# os.environ["PYTHONPATH"]，raylet 会继承它，worker 进程的 PYTHONPATH 也就有了。
# 所以在模块加载时就显式声明，不等、不猜。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)          # 驱动进程自己也要能 import，供下面自检

_existing_paths = [
    p for p in (os.environ.get("PYTHONPATH") or "").split(os.pathsep) if p
]
if _HERE not in _existing_paths:
    os.environ["PYTHONPATH"] = os.pathsep.join([_HERE] + _existing_paths)

# ----------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------
NUM_BACKENDS = 4          # 四个 SGLang 后端
BASE_PORT = 30000         # 端口 30000..30003
APP_NAME = "sglang_router_lab"
HTTP_PORT = 8000          # 和负载 README 的 --base-url 对应

# Ray 2.56.0 里三个路由类的导入路径（已核对源码 / 实测可 import）
P2C_ROUTER = (
    "ray.serve._private.request_router."
    "pow_2_router.PowerOfTwoChoicesRequestRouter"
)
CHR_ROUTER = (
    "ray.serve.experimental."
    "consistent_hash_router.ConsistentHashRouter"
)
CUSTOM_ROUTER = "router_d.PrefixAffinityLoadAwareRouter"

# 本轮配置表。max_ongoing=None 表示由 --max-ongoing 决定（C/D 必须给）。
ROUNDS = {
    "A_default": dict(
        desc="A 默认请求路由器（P2C），max_ongoing_requests=5",
        router_class=P2C_ROUTER,
        router_kwargs={},
        max_ongoing=5,
    ),
    "B_cand8": dict(
        desc="B 候选 1：P2C，max_ongoing_requests=8",
        router_class=P2C_ROUTER,
        router_kwargs={},
        max_ongoing=8,
    ),
    "B_cand32": dict(
        desc="B 候选 2：P2C，max_ongoing_requests=32",
        router_class=P2C_ROUTER,
        router_kwargs={},
        max_ongoing=32,
    ),
    "C_affinity": dict(
        desc="C 一致性哈希路由，num_fallback_replicas=0（严格亲和）",
        router_class=CHR_ROUTER,
        router_kwargs={
            "num_virtual_nodes": 100,
            "num_fallback_replicas": 0,
        },
        max_ongoing=None,
    ),
    "D_improved": dict(
        desc="D 自定义：前缀亲和优先，主副本超预留额度或明显更忙时让给最闲副本",
        router_class=CUSTOM_ROUTER,
        router_kwargs={
            "num_virtual_nodes": 100,
            "num_fallback_replicas": 0,
            # 主副本最多用一半槽位接亲和流量，排队数被压在半满，
            # 判据不会像绝对阈值那样被饱和信号恒真化（见 router_d.py）
            "reserve_ratio": 0.5,
            "margin_ratio": 0.125,
        },
        max_ongoing=None,
    ),
}


# ----------------------------------------------------------------------
# 一、搭 Ray 集群：1 head + 4 逻辑 worker
# ----------------------------------------------------------------------
def start_cluster() -> Cluster:
    """起 1 个 head + 4 个逻辑 worker 节点。

    关键点：head 节点**不挂** `sglang_backend` 资源，只有 4 个 worker 挂。
    配合部署上的 STRICT_SPREAD 放置组，4 个 Replica 必然一个节点一个，
    且不会落到 head 上 —— 「4 个 worker 均处于活跃状态」由此保证。
    """
    # Cluster() 默认 initialize_head=True，head 已经起好了
    cluster = Cluster(head_node_args={"num_cpus": 2})

    for i in range(NUM_BACKENDS):
        port = BASE_PORT + i
        cluster.add_node(
            num_cpus=4,
            resources={
                # 标志位：本节点可以跑一个 SGLang 转发的 Replica
                "sglang_backend": 1,
                # 端口标签：Replica 靠它认出自己该转发到哪个后端
                f"backend_port_{port}": 1,
            },
        )

    ray.init(address=cluster.address, ignore_reinit_error=True)
    return cluster


def build_node_map() -> dict:
    """读出 {Ray 节点 ID: SGLang 端口} 的映射，交给 Replica 用。

    Replica 在 __init__ 里拿自己的 node_id 去查这张表，就知道该转发到
    哪个端口；这样「Replica ↔ 节点 ↔ 后端」是三向一一对应的。
    """
    mapping = {}
    for node in ray.nodes():
        for key in (node.get("Resources") or {}):
            m = re.fullmatch(r"backend_port_(\d+)", key)
            if m:
                mapping[node["NodeID"]] = int(m.group(1))
    if len(mapping) != NUM_BACKENDS:
        raise RuntimeError(
            f"期望 4 个带 backend_port_* 的节点，实际 {len(mapping)} 个。"
            "检查 start_cluster() 是否被改动。"
        )
    return mapping


# ----------------------------------------------------------------------
# 二、Replica：把 /generate 原样透传给对应的 SGLang 后端
# ----------------------------------------------------------------------
class SGLangProxy:
    """一个 Replica 固定绑一个 SGLang 后端，只做 SSE 透传 + 加三个头。"""

    def __init__(self, node_map: dict):
        node_id = ray.get_runtime_context().get_node_id()
        if node_id not in node_map:
            raise RuntimeError(
                "本 Replica 落在了没有 SGLang 后端的节点上"
                f"（node_id={node_id}），放置组配置有问题。"
            )
        self.port = node_map[node_id]
        self.idx = self.port - BASE_PORT

        # README 要求：三个值分别稳定标识 Replica、Ray 节点、SGLang 后端。
        # 因为 STRICT_SPREAD 保证 Replica↔节点↔后端一一对应，
        # 用同一个下标派生三个不同的字符串即可，且重启后依然稳定。
        self.replica_label = f"replica-{self.idx}"
        self.node_label = f"node-{self.idx}"
        self.backend_label = f"sglang-{self.port}"

        # 连接池必须放开上限：默认 100，而负载压到 2048 并发。
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0),
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
        )

    def _out_headers(self) -> dict:
        return {
            "X-Ray-Replica-ID": self.replica_label,
            "X-Ray-Node-ID": self.node_label,
            "X-SGLang-Backend": self.backend_label,
        }

    async def __call__(self, request):
        body = await request.body()
        url = f"http://127.0.0.1:{self.port}/generate"

        # 逐跳头不能转发；accept-encoding 去掉是为了避免压缩干扰流式。
        drop = {
            "host", "content-length", "connection",
            "transfer-encoding", "accept-encoding",
        }
        fwd = {
            k: v for k, v in request.headers.items()
            if k.lower() not in drop
        }
        fwd["Accept"] = "text/event-stream"
        fwd["Content-Type"] = "application/json"

        try:
            resp = await self._session.post(url, data=body, headers=fwd)
        except Exception as exc:  # 后端没起来 / 连接失败
            return Response(
                content=(
                    f"转发到 {self.backend_label} 失败: {exc}"
                ).encode("utf-8"),
                status_code=502,
                headers=self._out_headers(),
            )

        if resp.status != 200:
            text = await resp.text()
            resp.release()
            return Response(
                content=text.encode("utf-8"),
                status_code=resp.status,
                headers=self._out_headers(),
            )

        async def body_iter():
            # 必须**逐块**往外吐，不能先 read() 再一次性返回：
            # 一次性返回会让客户端测到的 TTFT 等于总时长，指标就废了。
            try:
                async for chunk in resp.content.iter_any():
                    yield chunk
            finally:
                resp.release()

        return StreamingResponse(
            body_iter(),
            status_code=200,
            headers=self._out_headers(),
            media_type="text/event-stream",
        )


# ----------------------------------------------------------------------
# 三、按本轮配置组装部署
# ----------------------------------------------------------------------
def build_app(cfg: dict, max_ongoing: int, node_map: dict):
    router_cfg = RequestRouterConfig(
        request_router_class=cfg["router_class"],
        request_router_kwargs=dict(cfg["router_kwargs"]),
    )

    deployment = serve.deployment(
        num_replicas=NUM_BACKENDS,
        max_ongoing_requests=max_ongoing,
        # —— 让 4 个副本落在一个节点一个 ——
        #
        # 踩过的坑（2026-10-06）：先用 placement_group_bundles +
        # placement_group_strategy="STRICT_SPREAD"，在本机 Ray 2.56 上
        # 放置组建出来了，但副本根本没进组 —— 4 个副本全被塞进同一个
        # 节点。后果很隐蔽：4 个副本各自查自己节点的端口，全查到
        # 30000，于是 2048 个请求全打到 sglang-30000，另外三个后端
        # 基本闲置，跑五分钟才发现数据作废。
        #
        # max_replicas_per_node=1 是直接约束调度器的，不经过放置组，
        # 配合「只有 worker 才有的 sglang_backend 资源」，效果是
        # 「4 个副本、4 个 worker 节点、一个节点一个」。
        max_replicas_per_node=1,
        ray_actor_options={
            "num_cpus": 0.1,
            # head 节点没有这个资源，所以副本不可能落到 head 上
            "resources": {"sglang_backend": 0.01},
        },
        request_router_config=router_cfg,
    )(SGLangProxy)

    return deployment.bind(node_map)


# ----------------------------------------------------------------------
# 四、入口
# ----------------------------------------------------------------------
def deploy(app, port: int) -> None:
    """部署 Serve 应用。

    `serve.run` 的参数各家 Ray 版本不一致（2.56 已经不吃 host/port 了），
    所以从「参数最全」到「什么都不给」逐个试，第一个不报 TypeError 的就是
    这个版本支持的写法。

    端口不用我们操心：默认就是 127.0.0.1:8000，正好是负载 README 要求的
    `--base-url http://127.0.0.1:8000`。
    """
    candidates = [
        (
            "serve.run(app, name=, route_prefix=, host=, port=)",
            dict(name=APP_NAME, route_prefix="/",
                 host="0.0.0.0", port=port),
        ),
        (
            "serve.run(app, name=, route_prefix=)",
            dict(name=APP_NAME, route_prefix="/"),
        ),
        (
            "serve.run(app, route_prefix=)",
            dict(route_prefix="/"),
        ),
        ("serve.run(app)", dict()),
    ]

    for label, kwargs in candidates:
        try:
            serve.run(app, **kwargs)
        except TypeError:
            print(f"[部署] {label} 不被支持，换下一种写法", flush=True)
            continue
        print(f"[部署] 本机生效的写法：{label}", flush=True)
        return

    raise RuntimeError("serve.run 的每一种写法都报 TypeError，请把上面"
                       "几行贴给 Claude。")


def verify_placement(node_map: dict, timeout_s: float = 90.0) -> None:
    """确认 4 个副本真的落在 4 个不同节点上、算出 4 个不同端口。

    这道检查是必须的：一旦 4 个副本挤在同一节点，它们会算出同一个
    后端端口、全部转发到同一个 SGLang，另外三个后端闲置 —— 跑完
    2048 个请求才发现白跑。所以在发流量之前就先炸。
    """
    from ray.util.state import list_actors

    deadline = time.time() + timeout_s
    last_seen = None
    while time.time() < deadline:
        # 注意：Serve 副本的 class_name 是
        # "ServeReplica:<app>:<deployment>"，不是裸的部署名，
        # 所以这里必须用子串匹配。
        ports = sorted(
            node_map.get(a.node_id, -1)
            for a in list_actors()
            if "SGLangProxy" in (a.class_name or "")
            and a.state == "ALIVE"
        )
        last_seen = ports
        if len(ports) == NUM_BACKENDS:
            print(f"[放置检查] 4 个副本 -> 后端端口 {ports}", flush=True)
            if -1 in ports or len(set(ports)) != NUM_BACKENDS:
                raise RuntimeError(
                    "4 个副本没有均匀落在 4 个节点上"
                    f"（实际后端端口 = {ports}）。调度器没分散开，"
                    "继续跑只会得到无效数据，先停下。"
                )
            return
        print(f"[放置检查] 当前只看到 {len(ports)} 个副本，继续等…",
              flush=True)
        time.sleep(5)

    raise RuntimeError(
        f"{timeout_s} 秒内只看到 {last_seen}，没能确认 4 个副本各就各位"
    )


def ensure_router_module() -> None:
    """D 组的自定义路由类必须 import 得到，先自检，别等集群搭好了才炸。

    这个检查只覆盖**驱动进程**；真正需要 router_d 的是 HTTP Proxy 进程，
    它靠上面设置的 PYTHONPATH 拿到同一个目录。驱动这边能 import，
    Proxy 那边通常也就没问题；若这里就失败，说明两个文件没放在一起。
    """
    import importlib

    module_name = CUSTOM_ROUTER.rsplit(".", 1)[0]
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        sys.exit(
            f"[错误] 自定义路由模块 `{module_name}` 无法 import：{exc}\n"
            f"        D 组需要它和 app.py 放在**同一个目录**（当前是 {_HERE}）。\n"
            "        A / B / C 三组不需要这个文件，可以先跑那三组。"
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="第三关 target3：Ray Serve + 4 路 SGLang 路由对比"
    )
    p.add_argument(
        "--round", required=True, choices=sorted(ROUNDS),
        help="本轮跑哪一组（见 ROUNDS）",
    )
    p.add_argument(
        "--max-ongoing", type=int, default=None,
        help="覆盖 max_ongoing_requests；C/D 必须显式给（用 B 的胜出值）",
    )
    p.add_argument(
        "--reserve-ratio", type=float, default=None,
        help="仅 D 用：主副本拿这个比例的槽位接亲和流量，超出就让位",
    )
    p.add_argument(
        "--margin-ratio", type=float, default=None,
        help="仅 D 用：主副本排队数要比最闲副本多出这个比例才判为过忙",
    )
    p.add_argument("--port", type=int, default=HTTP_PORT)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ROUNDS[args.round]

    # D 组依赖同目录下的 router_d.py，先自检再动手搭集群
    if cfg["router_class"] == CUSTOM_ROUTER:
        ensure_router_module()

    max_ongoing = args.max_ongoing or cfg["max_ongoing"]
    if max_ongoing is None:
        sys.exit(
            f"[错误] {args.round} 必须显式给 --max-ongoing "
            "（填 B 组选出的那个值）"
        )

    if args.reserve_ratio is not None:
        cfg["router_kwargs"]["reserve_ratio"] = args.reserve_ratio
    if args.margin_ratio is not None:
        cfg["router_kwargs"]["margin_ratio"] = args.margin_ratio

    banner = [
        "",
        "=" * 62,
        f"  本轮：{args.round}",
        f"  说明：{cfg['desc']}",
        f"  路由类：{cfg['router_class']}",
        f"  路由参数：{cfg['router_kwargs']}",
        f"  max_ongoing_requests：{max_ongoing}",
        f"  HTTP 端口：{args.port}",
        "=" * 62,
    ]
    for line in banner:
        print(line, flush=True)

    cluster = None
    try:
        cluster = start_cluster()
        node_map = build_node_map()
        print(f"\n[集群] head + {len(node_map)} 个 worker 已就绪", flush=True)
        for nid, port in sorted(node_map.items(), key=lambda kv: kv[1]):
            print(f"        {nid[:16]}... -> SGLang :{port}", flush=True)

        app = build_app(cfg, max_ongoing, node_map)
        deploy(app, args.port)
        verify_placement(node_map)

        print(
            "\n[就绪] 部署完成。HTTP 口在 127.0.0.1:8000，"
            "现在到**另一个终端**跑 run_workload.py。"
            "\n       本终端 Ctrl-C 结束本轮并拆掉集群。\n",
            flush=True,
        )
        while True:
            time.sleep(3600)

    except KeyboardInterrupt:
        print("\n[退出] 收到 Ctrl-C，正在拆集群……", flush=True)
    finally:
        # 收尾顺序是有讲究的：必须先断开 Serve，再断开驱动与集群的连接，
        # 最后才拆节点。驱动还连着的时候，Ray 拒绝移除驱动所在的那个节点
        # （"Removing a node that is connected to this Ray client"）。
        try:
            serve.shutdown()
        except Exception:
            pass
        try:
            ray.shutdown()
        except Exception:
            pass
        if cluster is not None:
            try:
                cluster.shutdown()
            except Exception as exc:
                print(
                    f"\n[提示] 集群没能自动拆干净：{exc}\n"
                    "       下一轮开始前先执行： ray stop --force\n",
                    flush=True,
                )


if __name__ == "__main__":
    main()
