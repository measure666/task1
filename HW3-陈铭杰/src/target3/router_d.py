# -*- coding: utf-8 -*-
"""D 组：前缀亲和 + 负载感知退让的自定义路由。

依据全部来自 Ray 2.56.0 源码（读过，不是猜的）：

1) `RequestRouter.choose_replicas` 返回的是「候选名次表」
   `List[List[RunningReplica]]`，语义由基类的
   `_select_from_candidate_replicas`（_private/request_router/request_router.py:1023）
   决定：

       - 外层是名次先后：第 0 名里所有副本都不可用时，才退到第 1 名；
       - 内层是同一名次内的多个副本：框架**自动挑排队最短的那个**。

   所以「退让」不需要自己实现挑选，只要多给一个名次即可。
   C 组用 num_fallback_replicas=0，名次表里只有 [[主副本]]，
   主副本一忙就原地退避死等 —— 这正是 C 的热点问题来源。

2) 负载信号用 `RequestRouter._replica_queue_len_cache`，类型
   `ReplicaQueueLengthCache`（_private/request_router/common.py:66），
   `get(replica_id)` 返回 `Optional[int]`，过期或不存在时给 None。
   **这份缓存就是 A/B 组 P2C 路由用的同一份**
   （request_router.py:1044），所以 D 和它们比是同一把尺子。

3) 主副本用 `ConsistentHashRouter._lookup_ranked_replicas(key)` 算，
   哈希键是 `_routing_key(pending_request)` —— 优先取 `session_id`，
   也就是 HTTP 头 `X-Session-Id`（负载里它等于前缀族 ID）。

策略：主副本仍在预留额度内、且不比最闲副本明显更忙，就吃满亲和；
否则把流量让给最闲的副本。

—— 为什么第二版要改成「相对判据 + 预留额度」——

第一版用的是绝对阈值（排队数 >= max_ongoing_requests * 0.6 就退让）。
实测在本负载下失效：2048 个请求在 29.5 秒内到齐，而系统排空要 115 秒以上，
**整个到达窗口内系统全程饱和**，四个副本的排队数全都顶在 max_ongoing（32）。
于是「排队数 >= 19」这个判据恒为真，亲和那一支永远走不到 —— 实际日志
35840 次决策全是 shed，D 退化成单纯的「挑最闲副本」，白白丢掉前缀亲和。
根因不是阈值取多少，而是**队列长度这个信号在饱和时对所有副本都取同一个
饱和值，本身没有区分度**。

所以第二版换两件事：
  a) 预留额度（reserve）：主副本只用一部分槽位承接亲和流量。它的排队数
     会被压到预留线附近而不是顶到上限，信号重新变得可区分；
  b) 相对比较（margin）：只在主副本「比当前最闲的副本还忙出余量」时才让位。
     绝对繁忙本身不构成让位理由 —— 大家都忙的时候让位没有意义。
"""

import logging
from typing import Dict, List, Optional

from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.request_router.replica_wrapper import RunningReplica
from ray.serve.experimental.consistent_hash_router import (
    ConsistentHashRouter,
)

logger = logging.getLogger(SERVE_LOGGER_NAME)

# 主副本最多用这个比例的槽位承接「亲和流量」。超出部分让给别的副本。
# 0.5 的含义：一半槽位留给「回自己家」的请求，另一半留给溢出流量，
# 这样主副本的排队数会被压在半满附近，判据始终可区分。
DEFAULT_RESERVE_RATIO = 0.5

# 主副本的排队数要比「当前最闲的副本」还多出这个比例，才判为「过忙」。
DEFAULT_MARGIN_RATIO = 0.125


class PrefixAffinityLoadAwareRouter(ConsistentHashRouter):
    """先按一致性哈希找前缀亲和副本，副本超出预留额度或明显更忙时让位。"""

    def initialize_state(self, **kwargs) -> None:
        # 先摘掉自己的参数，其余原样交给父类（num_virtual_nodes /
        # num_fallback_replicas 仍由父类处理）。
        self._reserve_ratio = float(
            kwargs.pop("reserve_ratio", DEFAULT_RESERVE_RATIO)
        )
        self._margin_ratio = float(
            kwargs.pop("margin_ratio", DEFAULT_MARGIN_RATIO)
        )
        super().initialize_state(**kwargs)
        # 诊断用：三类决策各发生多少次。这是报告里「D 到底有没有让位」
        # 的直接证据 —— 第一版就是靠它发现亲和分支从未被走到。
        self._n_affinity = 0
        self._n_shed = 0
        self._n_no_signal = 0
        self._n_decisions = 0
        self._n_cache_hit = 0

    # ------------------------------------------------------------------
    # 负载读取
    # ------------------------------------------------------------------
    def _queue_len(self, replica: RunningReplica) -> Optional[int]:
        """读副本当前排队数；缓存没值或已过期时返回 None。"""
        cache = getattr(self, "_replica_queue_len_cache", None)
        if cache is None:
            return None
        return cache.get(replica.replica_id)

    # ------------------------------------------------------------------
    # 核心：决定名次表
    # ------------------------------------------------------------------
    async def choose_replicas(
        self,
        candidate_replicas: List[RunningReplica],
        pending_request=None,
    ) -> List[List[RunningReplica]]:
        replicas = list(candidate_replicas)

        # 只有一个副本、或这轮没有请求上下文时，没有选择余地。
        if len(replicas) <= 1 or pending_request is None:
            return [replicas] if replicas else []

        # 1) 哈希键 → 环上主副本（ranked[0] 恒为主副本）
        key = self._routing_key(pending_request)
        ranked_ids = self._lookup_ranked_replicas(key) if key else []
        by_id: Dict = {r.replica_id: r for r in replicas}
        primary = by_id.get(ranked_ids[0]) if ranked_ids else None
        if primary is None:
            # 环还没建好（首次请求）或候选集与环不一致，交给框架自由选。
            return [replicas]

        others = [
            r for r in replicas
            if r.replica_id != primary.replica_id
        ]

        # 2) 读负载：主副本的排队数，以及其余副本里最闲的那个。
        #    「最闲副本」是让位的比较基准 —— 让位只有让给更闲的副本才有意义。
        q_primary = self._queue_len(primary)
        alt_queues = [
            q for q in (self._queue_len(r) for r in others)
            if q is not None
        ]
        q_min = min(alt_queues) if alt_queues else None

        # 预留额度与比较余量，都跟 B 组选定的 max_ongoing_requests 挂钩。
        reserve = max(
            1, int(primary.max_ongoing_requests * self._reserve_ratio)
        )
        margin = max(
            1, int(primary.max_ongoing_requests * self._margin_ratio)
        )

        self._n_decisions += 1
        if q_primary is not None:
            self._n_cache_hit += 1

        # 3) 分派
        if q_primary is None:
            # 读数不可得 → 保守吃亲和。此时行为与 C 组一致，
            # 仍然可解释，而不是随机乱选。
            self._n_no_signal += 1
            decision = "affinity(no-signal)"
            ranks = [[primary], others]
        elif q_primary < reserve and (q_min is None or q_primary <= q_min + margin):
            # 主副本还在自己的预留额度内，且不比最闲副本忙出余量
            # → 吃满前缀亲和，第 2 名次放其余副本兜底（不像 C 组死等）。
            self._n_affinity += 1
            decision = "affinity"
            ranks = [[primary], others]
        else:
            # 主副本超出预留额度，或明显比最闲副本忙
            # → 第 0 名次换成其余副本，框架自动挑最闲的；主副本降到
            # 第 1 名次兜底。前缀缓存受损，但避免热点。
            self._n_shed += 1
            decision = "shed"
            ranks = [others, [primary]]

        # 每 256 次打一条日志，作为报告里「退让确实发生了」的证据。
        if self._n_decisions % 256 == 0:
            logger.info(
                f"[router_d] 第 {self._n_decisions} 次决策：{decision}，"
                f"主副本队列={q_primary}，最闲副本队列={q_min}，"
                f"预留={reserve}，余量={margin}，"
                f"亲和 {self._n_affinity} / 让位 {self._n_shed} / "
                f"无读数 {self._n_no_signal}"
            )

        return ranks
