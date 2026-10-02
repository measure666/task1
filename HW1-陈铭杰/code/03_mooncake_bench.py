#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_mooncake_bench.py —— Mooncake FAST'25 trace workload 回放

做四件事：
  1. 从 Mooncake trace 采样 N 条请求（按 input/output 长度过滤）
  2. 按 hash_ids 构造带「真实共享前缀」的 synthetic prompt（token 数对齐 input_length）
  3. 用泊松过程生成到达时刻，流式发给 SGLang 的 OpenAI 兼容接口
  4. 记录 input/output tokens、status、TTFT、latency，打印表格并保存 CSV

设计说明见同目录 ../Mooncake-Workload-回放指南.md

依赖：aiohttp（必须）；transformers（可选，用于精确对齐 token 数）

用法：
  # 先 dry-run，不发请求
  python 03_mooncake_bench.py --dry-run --n 20 --seed 42

  # 正式跑
  python 03_mooncake_bench.py --n 20 --seed 42 --rate 1.0 --out run1.csv --tag run1
"""

import argparse
import asyncio
import csv
import json
import os
import random
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass, asdict, field

try:
    import aiohttp
except ImportError:
    # dry-run 不需要网络库，所以这里不直接退出；真正发送前再检查（见 main）
    aiohttp = None

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# Mooncake trace 里 hash_ids 的块大小（token）。这是 trace 自带的约定。
BLOCK_TOKENS = 512

DEFAULT_TRACE = "~/trace/Mooncake/FAST25-release/arxiv-trace/mooncake_trace.jsonl"
DEFAULT_URL = "http://localhost:30000/v1/chat/completions"

# 用于生成填充文本的词表。英文常见词在 Qwen tokenizer 下大约 1 词 ≈ 1.35 token。
WORD_POOL = (
    "system context document passage retrieval knowledge base instruction assistant "
    "analysis summary report section chapter paragraph sentence reference source citation "
    "method result experiment observation dataset model training evaluation metric baseline "
    "approach framework pipeline component module service request response latency throughput "
    "memory compute storage network cluster node scaling batch schedule queue cache prefix "
    "token sequence attention layer head dimension embedding projection residual normalize "
    "temperature sampling decoding generation prompt completion stream chunk client server "
    "config parameter option flag environment dependency version release build test deploy "
    "monitor trace profile benchmark workload pattern distribution variance threshold limit"
).split()


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class Row:
    """从 trace 里采样出来的一条请求记录。"""
    line_no: int
    input_length: int
    output_length: int
    hash_ids: list


@dataclass
class Item:
    """构造好、准备发送的一条请求。"""
    idx: int
    row: Row
    prompt: str
    prompt_tokens_est: int      # 本地 tokenizer 数出来的正文 token 数
    exact: bool = False         # False 表示上面这个数字只是粗略估算，不可信


@dataclass
class Result:
    """一条请求的实测结果。"""
    idx: int
    planned_in: int = 0         # trace 的 input_length
    actual_in: int = 0          # usage.prompt_tokens（含 chat template 开销）
    prompt_tokens: int = 0      # 本地数出来的正文 token 数
    planned_out: int = 0        # trace 的 output_length
    actual_out: int = 0         # usage.completion_tokens
    status: str = ""
    ttft_ms: float = float("nan")
    latency_ms: float = float("nan")
    tpot_ms: float = float("nan")
    arrival_s: float = 0.0      # 计划的泊松到达时刻
    send_s: float = 0.0         # 实际发出时刻（用来检查分发有没有被阻塞）
    error: str = ""
    tag: str = ""


# ---------------------------------------------------------------------------
# 终端宽度对齐（中文字符占 2 列）
# ---------------------------------------------------------------------------

def _disp_width(s):
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in str(s))


def _pad(s, width, align=">"):
    s = str(s)
    gap = width - _disp_width(s)
    if gap <= 0:
        return s
    if align == ">":
        return " " * gap + s
    if align == "<":
        return s + " " * gap
    left = gap // 2
    return " " * left + s + " " * (gap - left)


# ---------------------------------------------------------------------------
# 1. 读 trace + 过滤 + 采样
# ---------------------------------------------------------------------------

def load_trace(path, min_input, max_input, max_output):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            il = rec.get("input_length")
            ol = rec.get("output_length")
            if il is None or ol is None:
                continue
            if not (min_input <= il <= max_input):
                continue
            if not (1 <= ol <= max_output):
                continue
            rows.append(Row(line_no, il, ol, rec.get("hash_ids") or []))
    return rows


def sample_rows(rows, n, seed):
    if len(rows) < n:
        sys.exit(
            f"过滤后只剩 {len(rows)} 条，不够采 {n} 条。\n"
            f"放宽 --max-input / --max-output，或减小 --n。"
        )
    rng = random.Random(seed)
    # 故意不排序：保持随机顺序，避免「短请求全排在前面」给实验引入系统性偏差
    return rng.sample(rows, n)


# ---------------------------------------------------------------------------
# 2. 构造 synthetic prompt
# ---------------------------------------------------------------------------

def get_tokenizer(model_name, quiet=False):
    """尝试加载 tokenizer。失败返回 None，退化为按词数估算。"""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if not quiet:
            print(f"[info] 已加载 tokenizer: {model_name}（token 数将精确对齐）",
                  file=sys.stderr)
        return tok
    except Exception as e:
        if not quiet:
            print(f"[warn] 无法加载 tokenizer（{type(e).__name__}: {e}）\n"
                  f"       退化为按 1.35 token/词 估算，「计划 in」会有偏差。\n"
                  f"       想精确对齐：pip install transformers",
                  file=sys.stderr)
        return None


def make_chunk(block_hash, n_tokens, tok, seed_base):
    """
    为一个 block 生成填充文本。

    **关键**：文本完全由 (seed_base, block_hash) 决定 —— 相同 hash 的 block 一定
    生成完全相同的文本。这正是让 RadixCache 能真正命中共享前缀的机制。
    """
    # Python 3.11+ 不再接受 tuple 作为种子，用字符串拼一个稳定种子
    rng = random.Random(f"{seed_base}:{int(block_hash)}")

    if tok is None:
        # 无 tokenizer：按经验比例估算词数
        n_words = max(1, int(round(n_tokens / 1.35)))
        return " ".join(rng.choices(WORD_POOL, k=n_words))

    # 有 tokenizer：先生成一个足够长的词序列，再二分成最接近目标 token 数的前缀。
    # 用「同一个已生成序列的前缀」做二分，保证单调性，收敛才可靠。
    max_words = int(n_tokens * 1.2) + 200
    words = rng.choices(WORD_POOL, k=max_words)

    best = words[0]
    lo, hi = 1, max_words
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = " ".join(words[:mid])
        if len(tok.encode(cand)) <= n_tokens:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def build_prompt(row, tok, seed_base):
    """
    按 hash_ids 逐块构造 prompt。

    hash_ids 是前缀块的哈希列表。我们让每个 block 的文本由它的 hash 唯一决定，
    于是：hash_ids[0] 相同的请求 → 第 0 个 block 文本完全相同 → 真实共享前缀。
    """
    ids = row.hash_ids
    if not ids:
        ids = [0]

    pieces = []
    remaining = row.input_length
    i = 0
    while remaining > 0:
        h = ids[i] if i < len(ids) else ids[-1]
        take = min(BLOCK_TOKENS, remaining)
        pieces.append(make_chunk(h, take, tok, seed_base))
        remaining -= take
        i += 1

    text = "\n".join(pieces)
    if tok is None:
        # 没有 tokenizer 时无法真正数出 token 数。
        # 这里给的数字是由 input_length 反推的，只作占位；真实值以运行时
        # usage.prompt_tokens 为准（见 Item.exact 标志）。
        return text, row.input_length
    return text, len(tok.encode(text))


def build_workload(rows, tok, seed_base):
    items = []
    for i, r in enumerate(rows, 1):
        text, n_tok = build_prompt(r, tok, seed_base)
        items.append(Item(i, r, text, n_tok, exact=(tok is not None)))
    return items


# ---------------------------------------------------------------------------
# 3. 泊松到达时刻
# ---------------------------------------------------------------------------

def poisson_arrivals(n, rate, seed):
    """
    指数分布间隔：t_{k+1} = t_k + Exp(rate)，Exp(rate) = -ln(U)/rate。

    注意这里生成的是「绝对到达时刻」列表，配合下面的分发循环使用 ——
    不要写成「发一条 sleep 一条」，那样服务时间会混进间隔，泊松过程就失真了。
    """
    rng = random.Random(seed + 1000)
    times, t = [], 0.0
    for _ in range(n):
        times.append(t)
        t += rng.expovariate(rate)
    return times


# ---------------------------------------------------------------------------
# 4. 发送 + 计时
# ---------------------------------------------------------------------------

def build_payload(args, item):
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": item.prompt}],
        "max_tokens": item.row.output_length,
        "temperature": args.temperature,
        "stream": True,
        # 流式下要拿到 usage 必须显式要求
        "stream_options": {"include_usage": True},
        # 小模型加速：不在返回值里带 logprobs / 其他多余字段
        "ignore_eos": args.ignore_eos,
        # Qwen3 默认开 thinking，会让 output_length 完全对不上，必须关
        "chat_template_kwargs": {"enable_thinking": False},
    }


async def send_one(session, url, payload, item, arrival_s, t_start, timeout, tag):
    res = Result(
        idx=item.idx,
        planned_in=item.row.input_length,
        planned_out=item.row.output_length,
        prompt_tokens=item.prompt_tokens_est,
        arrival_s=arrival_s,
        tag=tag,
    )

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    t0 = time.perf_counter()
    res.send_s = t0 - t_start

    ttft = None
    n_chunks = 0
    usage = None

    try:
        ctimeout = aiohttp.ClientTimeout(total=timeout, sock_read=timeout)
        async with session.post(url, data=data, headers=headers,
                                timeout=ctimeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                res.status = f"HTTP {resp.status}"
                res.error = body[:200].replace("\n", " ")
                res.latency_ms = (time.perf_counter() - t0) * 1000
                return res

            buf = b""
            async for raw in resp.content.iter_any():
                buf += raw
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == b"[DONE]":
                        continue
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue

                    if obj.get("usage"):
                        usage = obj["usage"]

                    for choice in obj.get("choices") or []:
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            n_chunks += 1
                            if ttft is None:
                                # 第一个「真正有内容」的 chunk 才算 TTFT
                                ttft = time.perf_counter() - t0

        res.latency_ms = (time.perf_counter() - t0) * 1000
        res.ttft_ms = ttft * 1000 if ttft is not None else float("nan")
        res.status = "success"

        if usage:
            res.actual_in = usage.get("prompt_tokens", 0) or 0
            res.actual_out = usage.get("completion_tokens", 0) or 0
        else:
            # 服务端没返回 usage，退化为本地数字 / chunk 计数
            res.actual_in = item.prompt_tokens_est
            res.actual_out = n_chunks

        if res.actual_out > 1 and ttft is not None:
            res.tpot_ms = (res.latency_ms - res.ttft_ms) / (res.actual_out - 1)

    except asyncio.TimeoutError:
        res.status = "TIMEOUT"
        res.error = f">{timeout}s"
        res.latency_ms = (time.perf_counter() - t0) * 1000
    except aiohttp.ClientError as e:
        res.status = "CLIENT_ERR"
        res.error = f"{type(e).__name__}: {e}"[:200]
        res.latency_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        res.status = type(e).__name__
        res.error = str(e)[:200]
        res.latency_ms = (time.perf_counter() - t0) * 1000

    return res


async def warmup(session, url, args, item):
    """跑一条热身请求，把 CUDA graph 捕获 / 首次 JIT 等一次性开销排除在测量之外。"""
    payload = build_payload(args, item)
    payload["max_tokens"] = 8
    payload["ignore_eos"] = False
    try:
        async with session.post(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=args.timeout),
        ) as resp:
            async for _ in resp.content.iter_any():
                pass
        print("[info] 热身完成", file=sys.stderr)
    except Exception as e:
        print(f"[warn] 热身失败（忽略，继续跑）：{type(e).__name__}: {e}", file=sys.stderr)


async def run_workload(args, items, arrivals):
    sem = asyncio.Semaphore(args.concurrency) if args.concurrency > 0 else None
    if sem:
        print(f"[warn] --concurrency={args.concurrency} 会推迟后续请求的发出时刻，"
              f"使泊松到达失真。推荐用 --concurrency 0（默认）。", file=sys.stderr)

    conn = aiohttp.TCPConnector(limit=0)  # 不限连接数
    async with aiohttp.ClientSession(connector=conn) as session:
        await warmup(session, args.url, args, items[0])

        t_start = time.perf_counter()
        tasks = []

        for item, arr in zip(items, arrivals):
            delay = (t_start + arr) - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

            async def guarded(it=item, a=arr):
                if sem:
                    async with sem:
                        return await send_one(session, args.url, build_payload(args, it),
                                              it, a, t_start, args.timeout, args.tag)
                return await send_one(session, args.url, build_payload(args, it),
                                      it, a, t_start, args.timeout, args.tag)

            tasks.append(asyncio.create_task(guarded()))

        print(f"[info] {len(tasks)} 条请求已按泊松过程发出，等待完成……", file=sys.stderr)
        results = await asyncio.gather(*tasks)

    results.sort(key=lambda r: r.idx)
    return results


# ---------------------------------------------------------------------------
# 5. 输出
# ---------------------------------------------------------------------------

HEADERS = ["#", "计划in", "实际in", "计划out", "实际out", "状态",
           "TTFT(ms)", "总延迟(ms)", "TPOT(ms)", "到达(s)"]
WIDTHS = [3, 8, 8, 8, 8, 10, 10, 12, 10, 9]
ALIGNS = ["^", ">", ">", ">", ">", "<", ">", ">", ">", ">"]


def print_table(results, args, trace_path):
    line = "-+-".join("-" * w for w in WIDTHS)
    total_w = sum(WIDTHS) + 3 * (len(WIDTHS) - 1)

    print()
    print("=" * total_w)
    print(f" Mooncake Trace -> SGLang 回放   tag={args.tag or '-'}  model={args.model}")
    print(f" trace={trace_path}")
    print(f" n={args.n}  rate={args.rate} req/s  seed={args.seed}  "
          f"input=[{args.min_input},{args.max_input}]  output<= {args.max_output}")
    print("=" * total_w)
    print(" " + " | ".join(_pad(h, w, a) for h, w, a in zip(HEADERS, WIDTHS, ALIGNS)))
    print(line)

    for r in results:
        cells = [
            r.idx,
            r.planned_in,
            r.actual_in if r.actual_in else "-",
            r.planned_out,
            r.actual_out if r.actual_out else "-",
            r.status,
            f"{r.ttft_ms:.1f}" if r.ttft_ms == r.ttft_ms else "-",
            f"{r.latency_ms:.1f}" if r.latency_ms == r.latency_ms else "-",
            f"{r.tpot_ms:.1f}" if r.tpot_ms == r.tpot_ms else "-",
            f"{r.arrival_s:.2f}",
        ]
        print(" " + " | ".join(_pad(c, w, a) for c, w, a in zip(cells, WIDTHS, ALIGNS)))

    print(line)

    ok = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status != "success"]

    print("-" * total_w)
    if ok:
        ttfts = sorted(r.ttft_ms for r in ok if r.ttft_ms == r.ttft_ms)
        lats = [r.latency_ms for r in ok if r.latency_ms == r.latency_ms]
        out_tok = sum(r.actual_out for r in ok)
        in_tok = sum(r.actual_in for r in ok)
        wall = max(r.send_s + r.latency_ms / 1000.0 for r in ok) - \
            min(r.send_s for r in ok)

        p95 = ttfts[min(int(len(ttfts) * 0.95), len(ttfts) - 1)] if ttfts else float("nan")

        print(f" 汇总: 成功 {len(ok)}/{len(results)}"
              f" | 平均 TTFT {statistics.mean(ttfts):.1f}ms"
              f" | P50 TTFT {statistics.median(ttfts):.1f}ms"
              f" | P95 TTFT {p95:.1f}ms")
        print(f"       平均延迟 {statistics.mean(lats):.1f}ms"
              f" | 总输入 tokens {in_tok:,}"
              f" | 总输出 tokens {out_tok:,}")
        if wall > 0:
            print(f"       回放总时长 {wall:.2f}s"
                  f" | 输出吞吐 {out_tok / wall:.1f} tok/s"
                  f" | 请求吞吐 {len(ok) / wall:.2f} req/s")

        # 计划 vs 实际：差值主要是 chat template 开销 + 采样随机性
        d_in = [r.actual_in - r.planned_in for r in ok if r.actual_in]
        d_out = [r.actual_out - r.planned_out for r in ok if r.actual_out]
        if d_in:
            print(f"       input  偏差: 均值 {statistics.mean(d_in):+.1f} token "
                  f"(chat template 开销)")
        if d_out:
            print(f"       output 偏差: 均值 {statistics.mean(d_out):+.1f} token "
                  f"(ignore_eos 下应接近 0)")

    if failed:
        print(f"\n ⚠ 失败 {len(failed)} 条：")
        for r in failed[:10]:
            print(f"   #{r.idx} {r.status}  {r.error}")

    print("=" * total_w)


def save_csv(results, path):
    fields = ["tag", "idx", "planned_in", "actual_in", "prompt_tokens",
              "planned_out", "actual_out", "status", "ttft_ms", "latency_ms",
              "tpot_ms", "arrival_s", "send_s", "error"]
    # utf-8-sig：Windows 上双击 Excel 打开不乱码
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            d = asdict(r)
            w.writerow({k: d.get(k, "") for k in fields})
    print(f" 结果已保存到: {os.path.abspath(path)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Mooncake trace workload 回放到 SGLang",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--trace", default=DEFAULT_TRACE, help="trace jsonl 路径")
    p.add_argument("--url", default=DEFAULT_URL, help="SGLang chat/completions 地址")
    p.add_argument("--model", default="Qwen3-0.6B", help="模型名（写进请求体）")
    p.add_argument("--n", type=int, default=20, help="采样条数（作业要求 10-30）")
    p.add_argument("--seed", type=int, default=42,
                   help="采样种子。两遍对比实验必须一致")
    p.add_argument("--rate", type=float, default=1.0, help="泊松到达率 λ (req/s)")
    p.add_argument("--min-input", type=int, default=890, help="input_length 下界")
    p.add_argument("--max-input", type=int, default=2048, help="input_length 上界")
    p.add_argument("--max-output", type=int, default=256, help="output_length 上界")
    p.add_argument("--concurrency", type=int, default=0,
                   help="客户端并发上限。0=不限，交给服务端调度（推荐）")
    p.add_argument("--timeout", type=float, default=120.0, help="单请求超时（秒）")
    p.add_argument("--temperature", type=float, default=0.0, help="采样温度")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B",
                   help="用于对齐 token 数的 tokenizer；设为空字符串则跳过")
    p.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false",
                   help="不强制生成满 max_tokens（默认强制）")
    p.add_argument("--out", default="mooncake_bench.csv", help="CSV 输出路径")
    p.add_argument("--tag", default="", help="本次运行的标记，写进 CSV")
    p.add_argument("--dry-run", action="store_true",
                   help="只采样 + 构造 prompt，不发请求")
    return p.parse_args()


def main():
    args = parse_args()
    trace_path = os.path.expanduser(args.trace)

    if not os.path.isfile(trace_path):
        sys.exit(f"找不到 trace 文件：{trace_path}\n"
                 f"确认 Mooncake 仓库已拷贝到 ~/trace/Mooncake/")

    # --- 采样 ---
    print(f"[info] 读取 {trace_path}", file=sys.stderr)
    all_rows = load_trace(trace_path, args.min_input, args.max_input, args.max_output)
    print(f"[info] 过滤后剩 {len(all_rows)} 条 "
          f"(input {args.min_input}-{args.max_input}, output<= {args.max_output})",
          file=sys.stderr)
    rows = sample_rows(all_rows, args.n, args.seed)

    # --- 构造 prompt ---
    tok = get_tokenizer(args.tokenizer, quiet=args.dry_run) if args.tokenizer else None
    items = build_workload(rows, tok, args.seed)

    # --- dry-run：只看采样和 prompt，不发请求 ---
    if args.dry_run:
        print(f"\n{'#':>3} | {'line':>6} | {'input_len':>9} | {'output_len':>10} | "
              f"{'prompt_tok':>10} | {'n_blocks':>8} | {'hash_ids[0]':>11}")
        print("-" * 76)
        for it in items:
            r = it.row
            h0 = r.hash_ids[0] if r.hash_ids else "-"
            tok_str = str(it.prompt_tokens_est) if it.exact else f"~{it.prompt_tokens_est}"
            print(f"{it.idx:>3} | {r.line_no:>6} | {r.input_length:>9} | "
                  f"{r.output_length:>10} | {tok_str:>10} | "
                  f"{len(r.hash_ids):>8} | {h0:>11}")
        inputs = [it.row.input_length for it in items]
        print("-" * 76)
        print(f"input_length: min={min(inputs)} max={max(inputs)} "
              f"mean={statistics.mean(inputs):.0f}")
        shared = sum(1 for it in items
                     if it.row.hash_ids and it.row.hash_ids[0] == items[0].row.hash_ids[0])
        print(f"hash_ids[0] 与第一条相同的: {shared}/{len(items)}  "
              f"→ 这些请求会共享首个 {BLOCK_TOKENS} token 前缀")
        if tok is None:
            print("\n[注意] 没有 tokenizer，上面带 ~ 的 prompt_tok 只是占位数字"
                  "（直接沿用了 trace 的 input_length），不构成任何验证。\n"
                  "       真实 token 数由服务端 usage.prompt_tokens 给出，"
                  "跑完看「实际in」列即可。\n"
                  "       想要本地精确对齐：pip install transformers")
        print("\n[DRY RUN] 未发送任何请求。去掉 --dry-run 即可正式开跑。")
        return

    # --- 发请求 ---
    if aiohttp is None:
        sys.exit("缺少 aiohttp。请在 client 环境执行：pip install aiohttp")

    arrivals = poisson_arrivals(len(items), args.rate, args.seed)
    try:
        results = asyncio.run(run_workload(args, items, arrivals))
    except KeyboardInterrupt:
        sys.exit("\n已中断")

    print_table(results, args, trace_path)
    save_csv(results, args.out)


if __name__ == "__main__":
    main()
