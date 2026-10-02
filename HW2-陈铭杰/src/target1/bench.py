#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench.py —— 任务一压测：测量 SGLang 前缀缓存的效果

【职责】
    读 build_loads.py 产出的 loads.json，按题目规定的流程发 64 条请求，
    采集指标，输出「逐请求明细」和「汇总」两张 CSV。

【跑法】
    先确保 SGLang 服务已在 127.0.0.1:30000 上跑着（不要加 --disable-radix-cache）

    python bench.py                  # 结果写到 run-1/
    python bench.py --run run-2      # 再跑一轮，写到 run-2/

【输出】
    results/target1/<组名>/<run-N>/requests.csv    逐请求明细（32 行）
    results/target1/<组名>/<run-N>/summary.csv     该组汇总（1 行）
    results/target1/<run-N>-summary_all.csv        两组对照表

    题目允许"对同一配置进行多次实验"，建立 run-1/、run-2/ 等子目录，
    所以每次运行都要用 --run 指定轮次，避免覆盖上一轮的数据。

【依赖】
    aiohttp —— 装在 client 环境里

【实验流程（题目规定）】
    服务预热（只做一次）
    ── 共享前缀组 ──
      flush_cache 确认成功 → 前缀预热 1 条（不计入） → 32 条测量
    ── 分散前缀组 ──
      flush_cache 确认成功 → 不预热            → 32 条测量

【计时口径】
    T0   = 真正把请求发出去的时刻（拿到并发槽之后，见 send_one 的注释）
    TTFT = T0 → 第一个"含内容的 chunk"到达
    E2E  = T0 → 流结束
    TPOT = (E2E - TTFT) / (completion_tokens - 1)
    吞吐量 = 成功条数 / 测量总耗时（只框住 32 条测量请求，不含预热与 flush）
"""

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

import aiohttp

# ═══════════════════════════ 配置 ═══════════════════════════

DEFAULT_BASE_URL = "http://127.0.0.1:30000"

# 题目规定的统一采样参数
SAMPLING_PARAMS = {
    "temperature": 0,
    "max_new_tokens": 16,
    "ignore_eos": True,       # 必定生成满 16 个，实际输出长度 = 计划输出长度
    "sampling_seed": 2026,
}

CONCURRENCY = 8        # 题目规定：最大并发数为 8
WARMUP_ROUNDS = 5      # 服务预热发几条

# 服务预热用的短请求（内容无所谓，短就行）
WARMUP_IDS = [9707, 11, 1879, 525, 498, 30]

# 相对项目根定位路径，这样在任何目录下跑都能找对文件。
# __file__ 是本文件路径；.resolve() 转成绝对路径；parents[2] 往上数三层：
#   parents[0] = src/target1   parents[1] = src   parents[2] = 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ═══════════════════════════ HTTP 辅助 ═══════════════════════════

async def flush_cache(session, url_base: str, retries: int = 5) -> None:
    """清空 RadixCache，并确认成功。

    服务端逻辑（scheduler.py:3640）：
        if self.is_fully_idle():   ← 必须"完全空闲"才会真的清
            self.tree_cache.reset()
            success = True
        else:
            logging.warning("Cache not flushed because there are pending requests...")

    也就是说：只要还有请求在跑或在排队，flush 会静默地不执行。
    HTTP 层用状态码区分 —— 成功 200，失败 400。

    所以这里必须检查状态码，失败就重试 —— 因为服务端可能只是
    "上一批请求刚跑完，状态还没收拾干净"。
    """
    last_text = ""
    for attempt in range(1, retries + 1):
        async with session.post(f"{url_base}/flush_cache") as resp:
            last_text = await resp.text()
            if resp.status == 200:
                return
        # 没成功，等一会儿再试
        await asyncio.sleep(0.5)

    raise RuntimeError(f"flush_cache 连续 {retries} 次失败：{last_text.strip()}")


async def warmup_service(session, url_base: str, rounds: int) -> None:
    """服务预热：连发几条短请求，让 CUDA / 显存池进入稳定状态。

    这些请求不计入结果。

    为什么需要：服务刚启动时，第一条请求要付 CUDA context 初始化、
    kernel 首次编译、显存池首次分配等"一次性开销"，可能比正常慢几十倍。
    如果不预热，32 条测量请求里的第 0 条会撞上冷启动，
    把 p95 抬得很难看 —— 而那个"尾延迟"是假的，跟缓存机制毫无关系。

    观察要点：第 1 条明显慢，之后迅速稳定。这里把每条的 e2e 打出来，
    你能亲眼看到"热机"的过程。
    """
    payload = {
        "input_ids": WARMUP_IDS,
        "sampling_params": SAMPLING_PARAMS,
        "stream": False,
    }
    for i in range(rounds):
        async with session.post(f"{url_base}/generate", json=payload) as resp:
            resp.raise_for_status()
            out = await resp.json()
        e2e_ms = out["meta_info"]["e2e_latency"] * 1000
        print(f"    第 {i + 1}/{rounds} 条: e2e = {e2e_ms:7.2f} ms")


async def warmup_prefix(session, url_base: str, prefix_ids: list) -> None:
    """前缀预热：发 1 条"只含共享前缀"的请求，把 2048 个 token 写进 radix 树。

    这条请求不计入结果 —— 它的唯一作用是"把路铺好"。

    为什么需要：radix 树一开始是空的（刚 flush 过）。如果不预热，
    32 条测量请求里的第 0 条就是"建缓存"的那一条，它的 cached_tokens = 0、
    TTFT 特别高，跟其余 31 条不是一路数据，会污染平均值和分位数。
    """
    payload = {
        "input_ids": prefix_ids,
        "sampling_params": SAMPLING_PARAMS,
        "stream": False,
    }
    async with session.post(f"{url_base}/generate", json=payload) as resp:
        resp.raise_for_status()
        out = await resp.json()

    meta = out["meta_info"]
    print(
        f"    已预热 {meta['prompt_tokens']} 个共享前缀 token "
        f"(本条 cached={meta['cached_tokens']}，不计入结果)"
    )


# ═══════════════════════════ 单条测量 ═══════════════════════════

async def send_one(session, sem: asyncio.Semaphore, url_base: str, input_ids: list) -> dict:
    """发一条流式测量请求，返回一条明细记录。

    ── 流式响应格式（已从源码核实，http_server.py:772）──
        yield b"data: " + dumps_json(out) + b"\n\n"   # 每个 chunk
        yield b"data: [DONE]\n\n"                     # 结尾
    所以按行读、剥掉 "data: " 前缀、遇到 [DONE] 就停。

    ── 计时口径（重要设计决定）──
    先拿并发槽，再开计时器（t0 在 async with sem 之后）。

    理由：TTFT 应该反映"服务端处理这条请求有多快"，而不是
    "客户端排队等槽位等了多久"。后者是我们自己的测试工具造成的假象，
    不属于被测系统。两组负载的并发数相同，这样处理对两组一视同仁。
    （如果反过来把排队也算进 TTFT，8 并发的排队时间会远大于 prefill 的
     差异，反而把要测量的效果淹没了。）
    """
    payload = {
        "input_ids": input_ids,
        "sampling_params": SAMPLING_PARAMS,
        "stream": True,
    }

    rec = {
        "ok": 0,
        "prompt_tokens": "",
        "cached_tokens": "",
        "real_prefill_tokens": "",
        "completion_tokens": "",
        "ttft_ms": "",
        "tpot_ms": "",
        "e2e_ms": "",
        "server_e2e_ms": "",
        "error": "",
    }

    try:
        async with sem:                       # 并发闸门：最多 8 条同时在飞
            t0 = time.monotonic()             # ← 计时起点
            ttft = None
            meta = None

            async with session.post(f"{url_base}/generate", json=payload) as resp:
                resp.raise_for_status()

                # resp.content 是字节流，async for 按行迭代。
                # 一个 chunk 可能被 TCP 分片，但按行读能自动拼回来。
                async for raw_line in resp.content:
                    line = raw_line.strip()
                    if not line or not line.startswith(b"data:"):
                        continue

                    body = line[len(b"data:"):].strip()
                    if body == b"[DONE]":
                        break

                    obj = json.loads(body)

                    # 每个 chunk 都带 meta_info，最后一个最完整。
                    # （已核实：流式下只有 logprob 类字段会被逐块切分，
                    #   cached_tokens / prompt_tokens 都是完整值。）
                    if "meta_info" in obj:
                        meta = obj["meta_info"]

                    # 第一个"含内容"的 chunk 到达 → TTFT
                    # 不能用"收到任何 chunk"判定：开头可能有空 delta 的 chunk。
                    if ttft is None and obj.get("text"):
                        ttft = time.monotonic() - t0

            t_end = time.monotonic()          # ← 计时终点

        if meta is None or ttft is None:
            raise RuntimeError("响应里没找到 meta_info 或没收到任何内容")

        pt = meta.get("prompt_tokens", 0)
        ct = meta.get("completion_tokens", 0)
        cached = meta.get("cached_tokens", 0)
        e2e = t_end - t0

        rec.update({
            "ok": 1,
            "prompt_tokens": pt,
            "cached_tokens": cached,
            "real_prefill_tokens": pt - cached,     # 实际要做 prefill 的 token 数
            "completion_tokens": ct,
            "ttft_ms": round(ttft * 1000, 3),
            "e2e_ms": round(e2e * 1000, 3),
            "server_e2e_ms": round(meta.get("e2e_latency", 0) * 1000, 3),
        })

        # TPOT = (端到端 - TTFT) / (输出 token 数 - 1)
        # 分母减 1 是因为第一个 token 的时间已经算在 TTFT 里了。
        # ct <= 1 时这个式子没有意义，留空。
        if ct > 1:
            rec["tpot_ms"] = round((e2e - ttft) * 1000 / (ct - 1), 3)

    except Exception as e:
        # 单条失败不能让整个实验崩掉 —— 记下来继续跑，
        # 最后用「成功率」体现。题目要求成功率这一项。
        rec["error"] = f"{type(e).__name__}: {e}"

    return rec


async def run_group(session, url_base: str, name: str, requests: list):
    """并发跑一组的 32 条测量请求，返回 (明细列表, 墙钟耗时秒)。"""
    sem = asyncio.Semaphore(CONCURRENCY)

    # ★ 墙钟计时只框住这 32 条测量请求。
    #   预热和 flush 都在这个函数外面，不会被算进来，
    #   否则吞吐量会偏低，而且两组偏低的程度还不一样。
    t_start = time.monotonic()

    tasks = [asyncio.create_task(send_one(session, sem, url_base, ids)) for ids in requests]
    records = await asyncio.gather(*tasks)

    t_end = time.monotonic()
    return records, (t_end - t_start)


# ═══════════════════════════ 统计 ═══════════════════════════

def percentile(values: list, p: float) -> float:
    """线性插值分位数，p 取 0~100。

    做法：先排序，再找位置 pos = (n-1) * p/100，落在两个数之间就按比例插值。
    这与 numpy.percentile 的默认算法一致。

    ★ 分位数有多种定义方式，同组数据用不同算法会差一点点。
      本项目统一用这一种，报告里注明即可。

    ★ 注意：32 个样本算 p95 本身就比较粗糙（位置 30.45，靠插值得到），
      不要过度解读小数点后的差异。
    """
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])

    pos = (len(xs) - 1) * p / 100.0
    lo = int(pos)                      # 向下取整
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return round(xs[lo] * (1 - frac) + xs[hi] * frac, 3)


def build_summary(name: str, records: list, wall_time: float) -> dict:
    """把一组的 32 条明细汇总成一行。"""
    ok = [r for r in records if r["ok"] == 1]
    n_total = len(records)
    n_ok = len(ok)

    total_prompt = sum(r["prompt_tokens"] for r in ok)
    total_cached = sum(r["cached_tokens"] for r in ok)
    total_real = sum(r["real_prefill_tokens"] for r in ok)

    ttfts = [r["ttft_ms"] for r in ok]
    tpots = [r["tpot_ms"] for r in ok if r["tpot_ms"] != ""]
    e2es = [r["e2e_ms"] for r in ok]

    return {
        "group": name,
        "n_total": n_total,
        "n_success": n_ok,
        # 成功率
        "success_rate": round(n_ok / n_total, 4) if n_total else 0,
        "wall_time_s": round(wall_time, 3),
        # 吞吐量口径：成功条数 / 测量总耗时（req/s）
        # 本实验输出长度固定为 16，所以 token 吞吐 = 本值 × 16，是同一个信息。
        "throughput_rps": round(n_ok / wall_time, 3) if wall_time > 0 else 0,
        "total_prompt_tokens": total_prompt,
        "total_cached_tokens": total_cached,
        # 题目给的定义：先求和再相除，不是"每条算完取平均"
        "cache_hit_rate": round(total_cached / total_prompt, 4) if total_prompt else 0,
        "total_real_prefill_tokens": total_real,
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p95": percentile(ttfts, 95),
        "tpot_p50": percentile(tpots, 50),
        "tpot_p95": percentile(tpots, 95),
        "e2e_p50": percentile(e2es, 50),
        "e2e_p95": percentile(e2es, 95),
    }


# ═══════════════════════════ 输出 ═══════════════════════════

def write_csv(path: Path, rows: list, fieldnames: list) -> None:
    """写 CSV。

    ★ encoding="utf-8-sig" 比普通的 "utf-8" 多一个 BOM 头，
      作用是让 Excel 在 Windows 上正确识别中文表头，不显示成乱码。
    ★ newline="" 是 csv 模块的要求，避免 Windows 上多出空行。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


REQ_FIELDS = [
    "group", "index", "ok", "prompt_tokens", "cached_tokens",
    "real_prefill_tokens", "completion_tokens",
    "ttft_ms", "tpot_ms", "e2e_ms", "server_e2e_ms", "error",
]

SUM_FIELDS = [
    "group", "n_total", "n_success", "success_rate",
    "wall_time_s", "throughput_rps",
    "total_prompt_tokens", "total_cached_tokens", "cache_hit_rate",
    "total_real_prefill_tokens",
    "ttft_p50", "ttft_p95",
    "tpot_p50", "tpot_p95",
    "e2e_p50", "e2e_p95",
]


def print_comparison(summaries: list) -> None:
    """打印对照表 —— 报告的核心结论就在这两行里。"""
    if len(summaries) < 2:
        return
    a, b = summaries[0], summaries[1]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    print("\n" + "=" * 62)
    print(f"{'指标':<26}{a['group']:<18}{b['group']:<18}")
    print("-" * 62)
    for key in [
        "n_success", "success_rate", "throughput_rps",
        "cache_hit_rate", "total_real_prefill_tokens",
        "ttft_p50", "ttft_p95", "tpot_p50", "tpot_p95", "e2e_p50", "e2e_p95",
    ]:
        print(f"{key:<26}{fmt(a[key]):<18}{fmt(b[key]):<18}")
    print("=" * 62)


# ═══════════════════════════ 主流程 ═══════════════════════════

async def main_async(args) -> None:
    loads = json.loads(Path(args.loads).read_text(encoding="utf-8"))
    out_root = Path(args.out_root)

    # 连接数不额外限制（真正的限流靠下面的 Semaphore）
    connector = aiohttp.TCPConnector(limit=0)
    # 单条请求的总超时设为 5 分钟，避免慢请求把脚本挂死
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # ── 步骤 1：服务预热（整个实验只做一次）──
        print(f"[1] 服务预热（{args.warmup_rounds} 条短请求，不计入结果）")
        await warmup_service(session, args.url, args.warmup_rounds)

        # 两组负载：共享前缀组带预热前缀，分散前缀组不带
        plans = [
            ("shared_prefix",    loads["shared_requests"],    loads["shared_prefix"]),
            ("dispersed_prefix", loads["dispersed_requests"], None),
        ]

        summaries = []

        for idx, (name, requests, prefix) in enumerate(plans, start=2):
            print(f"\n[{idx}] {name}")

            # ── 步骤 2：等空闲后 flush_cache 并确认成功 ──
            # 到这里上一组已经 await 完了，服务端应该已空闲；
            # flush_cache 内部若失败会自动重试。
            await flush_cache(session, args.url)
            print("    flush_cache 成功（缓存已清空，起点干净）")

            # ── 步骤 3：前缀预热 ──
            if prefix is not None:
                await warmup_prefix(session, args.url, prefix)
            else:
                print("    分散前缀组：按要求不做前缀预热")

            # ── 步骤 4：32 条测量请求（并发 ≤ 8）──
            records, wall = await run_group(session, args.url, name, requests)

            n_ok = sum(1 for r in records if r["ok"] == 1)
            print(f"    完成 {n_ok}/{len(records)} 条，墙钟 {wall:.2f} s")

            # ── 步骤 5：写 CSV ──
            rows = []
            for i, r in enumerate(records):
                row = dict(r)
                row["group"] = name
                row["index"] = i
                rows.append(row)

            summary = build_summary(name, records, wall)
            summaries.append(summary)

            # 每次实验的结果写进对应轮次的子目录：
            #   results/target1/<组名>/<run-1|run-2>/...
            run_dir = out_root / name / args.run
            write_csv(run_dir / "requests.csv", rows, ["group", "index"] + REQ_FIELDS[2:])
            write_csv(run_dir / "summary.csv", [summary], SUM_FIELDS)
            print(f"    → {run_dir / 'requests.csv'}")
            print(f"    → {run_dir / 'summary.csv'}")

        # ── 步骤 6：对照表 ──
        all_path = out_root / f"{args.run}-summary_all.csv"
        write_csv(all_path, summaries, SUM_FIELDS)
        print_comparison(summaries)
        print(f"\n对照表已写出：{all_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="任务一压测：测量 SGLang 前缀缓存")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="SGLang 服务地址")
    parser.add_argument(
        "--loads",
        default=str(PROJECT_ROOT / "src" / "target1" / "loads.json"),
        help="build_loads.py 产出的负载文件",
    )
    parser.add_argument(
        "--out-root",
        default=str(PROJECT_ROOT / "results" / "target1"),
        help="结果输出根目录",
    )
    parser.add_argument(
        "--run",
        default="run-1",
        help="实验轮次，决定写到哪个子目录（如 run-1 / run-2）。"
             "题目允许对同一配置重复实验以取稳定值。",
    )
    parser.add_argument("--warmup-rounds", type=int, default=WARMUP_ROUNDS, help="服务预热条数")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
