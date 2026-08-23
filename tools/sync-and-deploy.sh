#!/bin/bash
# sync-and-deploy.sh — PR 合并后的收尾同步 + 生产部署 + 健康验证
#
# 背景(2026-08-23 复盘): dev 仓库→PR→合并→同步→pinned 部署→重启→健康检查
# 这套流程今天手打了 14 遍, 期间出过 detached HEAD(本地 main 指针虽然
# 一直是对的, 但 HEAD 曾游离指向某次诊断用的 checkout, 没人当场发现)、
# 一次分支清理误推到了另一个仓库(cwd 依赖脆弱)。这个脚本只封装"合并
# 之后"的纯重复收尾段 —— 创建分支/写 commit/发 PR/审批合并这些每次
# 内容不同、需要人工判断的步骤不在这个脚本里, 留在交互流程中。
#
# 用法:
#   bash tools/sync-and-deploy.sh                          # 同步+部署+验证
#   bash tools/sync-and-deploy.sh --delete-branch <name>    # 额外清理已合并分支(本地+远程)

set -euo pipefail

DEV_REPO="$(cd "$(dirname "$0")/.." && pwd)"
PINNED_REPO="${AETHERFORGE_PINNED_REPO:-$HOME/aetherforge-final-ae3570f}"
PLIST="${AETHERFORGE_PLIST:-$HOME/Library/LaunchAgents/com.aetherforge.gateway.plist}"
HEALTH_URL="${AETHERFORGE_HEALTH_URL:-http://127.0.0.1:4000/health}"

DELETE_BRANCH=""
if [ "${1:-}" = "--delete-branch" ]; then
    DELETE_BRANCH="${2:?--delete-branch 需要跟分支名}"
fi

cd "$DEV_REPO"

echo "== [1/5] dev 仓库前置检查 =="
if [ -n "$(git status --short)" ]; then
    echo "❌ dev 仓库工作树不干净, 停止 —— 先处理未提交改动:"
    git status --short
    exit 1
fi
CURRENT_BRANCH="$(git branch --show-current)"
if [ -z "$CURRENT_BRANCH" ]; then
    echo "⚠️  检测到 detached HEAD(今天真实撞过这个坑), 自动切回 main"
fi

echo "== [2/5] dev 仓库同步到 origin/main =="
git switch main
git pull --ff-only origin main
DEV_SHA="$(git rev-parse HEAD)"
echo "   dev main -> ${DEV_SHA:0:7}"

if [ -n "$DELETE_BRANCH" ]; then
    echo "== 清理已合并分支: $DELETE_BRANCH =="
    git branch -d "$DELETE_BRANCH" 2>&1 || echo "   (本地分支不存在或已删, 跳过)"
    git push origin --delete "$DELETE_BRANCH" 2>&1 || echo "   (远程分支不存在或已删, 跳过)"
fi

echo "== [3/5] 同步 pinned 部署仓库 =="
cd "$PINNED_REPO"
git fetch origin main
git checkout "$DEV_SHA"
echo "   pinned HEAD -> $(git rev-parse --short HEAD)"

echo "== [4/5] 重启 gateway =="
launchctl unload "$PLIST" 2>&1 || true
sleep 2
launchctl load "$PLIST" 2>&1
sleep 8

echo "== [5/5] 健康验证(最多重试 5 次, 间隔 3s) =="
for i in 1 2 3 4 5; do
    if curl -sf -m 10 "$HEALTH_URL" 2>&1; then
        echo ""
        echo "✅ 同步部署完成: ${DEV_SHA:0:7}, 健康检查通过(第 $i 次尝试)"
        exit 0
    fi
    echo "   第 $i 次健康检查未通过, 3s 后重试..."
    sleep 3
done

echo "❌ 健康检查 5 次均未通过 —— 部署完成但服务未就绪, 需要人工排查"
exit 1
