#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_loads.py —— 构造任务一的两组测试负载，输出 loads.json

【职责】
    只造数据，不发请求。压测脚本 bench.py 读它产出的 JSON。

【为什么用 input_ids 而不是文字】
    题目要求精确控制 token 数量（2048 / 2112），而"一段文字有多少个 token"
    很难凑准。直接给出 token id 列表就能一个不多一个不少。

【跑法】
    python build_loads.py                     # 默认输出 ./loads.json
    python build_loads.py --out /tmp/x.json   # 指定输出路径

【依赖】
    只用标准库（random / json / argparse），不需要 transformers，任何 Python 环境都能跑。
"""

import argparse
import json
import random
from pathlib import Path

# ═══════════════════════════ 实验参数 ═══════════════════════════

# Qwen3-0.6B 的词表大小。实测特殊 token 的最小 id 就是 151643，
# 所以取 [0, 151643) 的随机数能天然避开 <|im_end|> 这类控制符，
# 不需要额外做排除。（混入控制符会让模型提前截断，实验就失真了。）
VOCAB_SIZE = 151643

# 固定种子。同一种子永远产出同一份负载 —— 这是"可复现"的前提，
# 也是"32 条请求共享完全相同的那 2048 个 token"的前提。
SEED = 2026

SHARED_PREFIX_LEN = 2048  # 共享前缀长度（题目规定）
SUFFIX_LEN = 64           # 每条请求的独立后缀长度（题目规定）
DISPERSED_LEN = 2112      # 分散前缀组每条的长度（题目规定）
NUM_REQUESTS = 32         # 每组请求数（题目规定）

# 两组输入长度必须一致，否则不可比。这行在 import 时就会执行，
# 参数填错了立刻炸掉，不会等到跑完才发现数据是废的。
assert SHARED_PREFIX_LEN + SUFFIX_LEN == DISPERSED_LEN, "两组输入长度必须一致！"


# ═══════════════════════════ 工具函数 ═══════════════════════════

def make_tokens(rng: random.Random, n: int) -> list:
    """用指定的随机引擎造 n 个合法 token id。

    注意参数是 rng 对象，不是直接用全局的 random 模块。
    这样每个用途各有独立的随机序列，改动一处不会串到另一处。

    ★ Python 对照：random.Random(seed) 相当于 C++ 的 std::mt19937 gen(seed)，
      是"一个独立的随机引擎对象"。而 random.seed(seed) 相当于 C 的 srand()，
      操作的是全局状态，模块之间会互相干扰 —— 不要用那个。
    """
    # 下面的写法叫"列表推导式"，等价于：
    #   result = []
    #   for _ in range(n):
    #       result.append(rng.randrange(VOCAB_SIZE))
    # randrange(k) 返回 [0, k) 的整数，相当于 uniform_int_distribution。
    return [rng.randrange(VOCAB_SIZE) for _ in range(n)]


# ═══════════════════════════ 两组负载 ═══════════════════════════

def build_shared(rng_prefix: random.Random, rng_suffixes: random.Random):
    """共享前缀组：32 条请求共享同一段 2048 token，各自接 64 token 后缀。

    返回 (prefix, requests)。prefix 单独返回是为了给预热请求用 ——
    预热时只发这 2048 个 token，把前缀先写进缓存。
    """
    prefix = make_tokens(rng_prefix, SHARED_PREFIX_LEN)

    requests = []
    for i in range(NUM_REQUESTS):
        suffix = make_tokens(rng_suffixes, SUFFIX_LEN)
        # ★ Python 的 + 用来拼接列表，相当于 C++ vector 的 insert(end, ...)，
        #   但这里返回的是新列表，不修改原对象。
        requests.append(prefix + suffix)

    return prefix, requests


def build_dispersed(rng: random.Random) -> list:
    """分散前缀组：32 条请求各 2112 token，首 token 两两不同。

    首 token 不同 → radix tree 从根的第一个分支就分叉 → 谁也无法命中别人的缓存。
    这正是要和"共享前缀组"形成的对照。
    """
    requests = []
    for i in range(NUM_REQUESTS):
        # 首 token 显式指派，而不是靠随机生成。
        # 原因：32 个随机数在 15 万的空间里仍有约 0.3% 概率撞车（生日问题）。
        # 概率虽小，但一旦撞车这两条请求就悄悄共享了前缀，
        # "分散前缀组"就不再纯粹，实验结论会被污染。
        # 直接按序号指派，100% 保证两两不同，且都远小于 VOCAB_SIZE。
        first_token = 1000 + i

        body = make_tokens(rng, DISPERSED_LEN - 1)
        requests.append([first_token] + body)

    return requests


# ═══════════════════════════ 自检 ═══════════════════════════

def check(prefix: list, shared_requests: list, dispersed_requests: list) -> None:
    """写出文件前的自检。

    宁可在造数据时炸掉，也不要产出一份"看起来正常但其实是废的"负载 ——
    那种数据跑完实验才发现问题，重跑成本极高。
    """

    # ── 长度检查 ──
    assert len(prefix) == SHARED_PREFIX_LEN, f"共享前缀长度应为 {SHARED_PREFIX_LEN}，实为 {len(prefix)}"
    assert len(shared_requests) == NUM_REQUESTS
    assert len(dispersed_requests) == NUM_REQUESTS

    for req in shared_requests:
        assert len(req) == DISPERSED_LEN, f"共享前缀组某条长度 {len(req)} != {DISPERSED_LEN}"
        # 每条的前 2048 个 token 必须逐元素等于 prefix —— 这是"共享"的定义。
        # prefix[:2048] 是切片语法，取前 2048 个，相当于取出子数组。
        assert req[:SHARED_PREFIX_LEN] == prefix, "共享前缀组内部前缀不一致！"

    for req in dispersed_requests:
        assert len(req) == DISPERSED_LEN, f"分散前缀组某条长度 {len(req)} != {DISPERSED_LEN}"

    # ── 分散前缀组：首 token 必须两两不同 ──
    # set() 是 Python 的集合，自动去重，相当于 Java 的 HashSet。
    # 去重后个数仍是 32，就说明没有重复。
    first_tokens = [req[0] for req in dispersed_requests]
    assert len(set(first_tokens)) == NUM_REQUESTS, "分散前缀组存在重复首 token！"

    # ── token id 范围检查 ──
    # 全部落在 [0, VOCAB_SIZE) 就不会是特殊 token，也一定在词表内。
    for req in shared_requests + dispersed_requests:
        for t in req:
            assert 0 <= t < VOCAB_SIZE, f"非法 token id：{t}"

    # ── 两组总长度一致（题目完成标准之一）──
    assert sum(len(r) for r in shared_requests) == sum(len(r) for r in dispersed_requests)


# ═══════════════════════════ 主流程 ═══════════════════════════

def main() -> None:
    # argparse 是标准库的命令行参数解析，相当于 C 的 getopt。
    parser = argparse.ArgumentParser(description="构造任务一（前缀缓存测量）的测试负载")
    parser.add_argument("--out", default="loads.json", help="输出 JSON 路径（默认 ./loads.json）")
    args = parser.parse_args()

    # 三个互相独立的随机引擎，各自负责一块。
    # 好处：想只改分散前缀组的生成逻辑时，共享前缀组的数据不会跟着变，
    # 避免"改一处、两组数据全变、无法对比"。
    rng_prefix = random.Random(SEED)
    rng_suffixes = random.Random(SEED + 1)
    rng_dispersed = random.Random(SEED + 2)

    prefix, shared_requests = build_shared(rng_prefix, rng_suffixes)
    dispersed_requests = build_dispersed(rng_dispersed)

    # 先自检，不通过就直接抛异常，不会写出坏文件
    check(prefix, shared_requests, dispersed_requests)

    data = {
        # meta 记录生成参数，方便日后核对"这份负载是用什么参数造的"
        "meta": {
            "vocab_size": VOCAB_SIZE,
            "seed": SEED,
            "shared_prefix_len": SHARED_PREFIX_LEN,
            "suffix_len": SUFFIX_LEN,
            "dispersed_len": DISPERSED_LEN,
            "num_requests": NUM_REQUESTS,
        },
        "shared_prefix": prefix,
        "shared_requests": shared_requests,
        "dispersed_requests": dispersed_requests,
    }

    out = Path(args.out)
    # ★ with 语句保证文件用完自动关闭，即使中途抛异常也会关。
    #   相当于 C++ 的 RAII（lock_guard 那套）或 Java 的 try-with-resources。
    #   encoding="utf-8" 显式指定编码，避免 Windows 上的默认编码把内容写乱。
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    # ★ f"..." 是 f-string，把 {} 里的变量插进字符串，相当于 Java 的 String.format。
    print(f"已写出：{out}")
    print(f"  共享前缀      : {len(prefix)} token")
    print(f"  共享前缀组    : {len(shared_requests)} 条 × {len(shared_requests[0])} token")
    print(f"  分散前缀组    : {len(dispersed_requests)} 条 × {len(dispersed_requests[0])} token")
    print(f"  首 token 互异 : {len(set(r[0] for r in dispersed_requests))} 个")


# ★ 这个判断的意思：只有"直接运行本文件"时才执行 main()，
#   被别的文件 import 时不执行。相当于 Java 的 public static void main。
if __name__ == "__main__":
    main()
