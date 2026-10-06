#!/usr/bin/env python3
"""minesolver -- 扫雷棋盘求解助手（纯确定性逻辑，无暴力 CSP）

棋盘文本格式（每行等长）：
    0-8   已翻开的数字
    .     未翻开的格子
    F     已标记的旗子

用法：
    python -m minesolver board.txt      # 分析棋盘，输出安全格/必雷
    python -m minesolver --guess board.txt   # 无确定走法时给一个猜测建议
    cat board.txt | python -m minesolver      # 从 stdin 读

求解逻辑：
    1. 基础规则：数字 N，相邻旗子 f，未翻开邻格 h
       - N - f == 0      → h 全是安全格
       - N - f == len(h) → h 全是雷
    2. 子集规则：若数字 A 的未翻开邻格集合是 B 的子集，
       则差集上的雷数 = (B 剩余雷数 - A 剩余雷数)，
       差集雷数为 0 → 全安全；差集雷数 == 差集大小 → 全是雷。
       经典 1-2-1 模式是子集规则的特例（本工具直接覆盖）。
    3. --guess：无确定走法时，给每个前沿未翻开格估算
       p = max(相邻数字的 剩余雷数/未翻开邻格数)，选 p 最小的；
       若没有前沿格，任选一个未翻开格。
"""

import argparse
import sys

DIRS = [(-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)]


def parse_board(text):
    """解析棋盘文本，返回 (grid, rows, cols)。非法字符抛 ValueError。"""
    lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip() != ""]
    if not lines:
        raise ValueError("棋盘为空")
    width = len(lines[0])
    grid = []
    for i, ln in enumerate(lines):
        if len(ln) != width:
            raise ValueError(f"第 {i + 1} 行长度 {len(ln)} 与首行 {width} 不一致")
        for j, ch in enumerate(ln):
            if ch != "." and ch != "F" and not ch.isdigit():
                raise ValueError(f"第 {i + 1} 行第 {j + 1} 列非法字符: {ch!r}")
        grid.append(list(ln))
    return grid, len(grid), width


def neighbors(r, c, rows, cols):
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def analyze(grid):
    """返回 (safe: set, mines: set, constraints: list)。

    constraints: [(frozenset(未翻开邻格), 剩余雷数), ...] 供 --guess 用。
    确定性逻辑：基础规则 + 子集规则 + 约束传播，迭代到不动点。
    经典 1-2-1 模式是子集规则的特例（本工具直接覆盖）。
    """
    rows, cols = len(grid), len(grid[0])
    safe, mines = set(), set()
    cons = []  # {"cells": set, "rem": int}，可变约束

    for r in range(rows):
        for c in range(cols):
            ch = grid[r][c]
            if not ch.isdigit():
                continue
            n = int(ch)
            hidden = set()
            flags = 0
            for nr, nc in neighbors(r, c, rows, cols):
                g = grid[nr][nc]
                if g == "F":
                    flags += 1
                elif g == ".":
                    hidden.add((nr, nc))
            remaining = n - flags
            if remaining < 0 or remaining > len(hidden):
                raise ValueError(f"第 {r + 1} 行第 {c + 1} 列数字 {n} 与棋盘矛盾")
            cons.append({"cells": hidden, "rem": remaining})

    changed = True
    while changed:
        changed = False
        for c in cons:
            # 把已知结论折进约束
            known_mines = c["cells"] & mines
            if known_mines:
                c["cells"] -= known_mines
                c["rem"] -= len(known_mines)
                changed = True
            known_safe = c["cells"] & safe
            if known_safe:
                c["cells"] -= known_safe
                changed = True
            if c["rem"] < 0 or c["rem"] > len(c["cells"]):
                raise ValueError("棋盘矛盾：某数字的剩余雷数超出可能范围")
            # 基础规则
            if c["cells"]:
                if c["rem"] == 0:
                    new = c["cells"] - safe
                    if new:
                        safe |= new
                        changed = True
                elif c["rem"] == len(c["cells"]):
                    new = c["cells"] - mines
                    if new:
                        mines |= new
                        changed = True
        # 子集规则
        for i in range(len(cons)):
            for j in range(len(cons)):
                if i == j:
                    continue
                si, ri = cons[i]["cells"], cons[i]["rem"]
                sj, rj = cons[j]["cells"], cons[j]["rem"]
                if si and si < sj:  # 真子集
                    diff = sj - si
                    d = rj - ri
                    if d == 0:
                        new = diff - safe
                        if new:
                            safe |= new
                            changed = True
                    elif d == len(diff):
                        new = diff - mines
                        if new:
                            mines |= new
                            changed = True
    return safe, mines, [(frozenset(c["cells"]), c["rem"]) for c in cons]


def guess_cell(grid, infos, safe, mines):
    """无确定走法时的猜测：返回 (r, c) 或 None（无未知格）。

    排除已判定为安全格/必雷的格子。
    """
    rows, cols = len(grid), len(grid[0])
    unknown = [(r, c) for r in range(rows) for c in range(cols)
               if grid[r][c] == "." and (r, c) not in safe and (r, c) not in mines]
    if not unknown:
        return None
    # 每个前沿格的粗略雷概率
    prob = {}
    for cell_set, remaining in infos:
        if not cell_set or remaining <= 0:
            continue
        p = remaining / len(cell_set)
        for cell in cell_set:
            if prob.get(cell, 0.0) < p:
                prob[cell] = p
    if prob:
        return min(prob, key=lambda cell: prob[cell])
    # 棋盘上一个数字都没有：随便猜一个未知格
    return unknown[0]


def fmt(cell):
    return f"({cell[0] + 1}, {cell[1] + 1})"  # 1-based，方便人看


def render(grid, safe, mines):
    """把结论画回棋盘：安全格标 S，必雷标 X。"""
    out = [row[:] for row in grid]
    for r, c in safe:
        out[r][c] = "S"
    for r, c in mines:
        out[r][c] = "X"
    return "\n".join("".join(row) for row in out)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="扫雷棋盘求解助手：确定性逻辑找出安全格和必雷")
    ap.add_argument("board", nargs="?",
                    help="棋盘文本文件（省略则从 stdin 读）")
    ap.add_argument("--guess", action="store_true",
                    help="无确定走法时给出一个猜测建议")
    args = ap.parse_args(argv)

    if args.board:
        with open(args.board, encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    try:
        grid, rows, cols = parse_board(text)
        safe, mines, infos = analyze(grid)
    except ValueError as e:
        print(f"棋盘错误：{e}", file=sys.stderr)
        return 2

    print(f"棋盘 {rows}x{cols}，分析结果：")
    print()
    print(render(grid, safe, mines))
    print()
    if safe:
        print(f"安全格（{len(safe)}）："
              + ", ".join(fmt(c) for c in sorted(safe)))
    else:
        print("安全格：无（确定性逻辑找不到）")
    if mines:
        print(f"必雷（{len(mines)}）："
              + ", ".join(fmt(c) for c in sorted(mines)))
    else:
        print("必雷：无（确定性逻辑找不到）")

    if not safe and not mines and args.guess:
        g = guess_cell(grid, infos, safe, mines)
        if g is None:
            print("\n棋盘已无未知格。")
        else:
            print(f"\n猜测建议：点 {fmt(g)}（前沿格中估算雷概率最低）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
