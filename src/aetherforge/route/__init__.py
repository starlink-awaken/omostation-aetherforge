"""aetherforge.route — RouteScheduler 路由核心 (ARCHITECTURE-v2 §2.2).

三级路由 (模型 → Provider → 节点) + 策略评分 (cost/speed/quota/affinity)
+ FallbackManager 自动降级. 消费 routing_policy 命名空间.

TASK-02788FE2 治本实现. 设计: projects/aetherforge/ARCHITECTURE-v2.md:88-134.
"""

from __future__ import annotations

from .scheduler import (
    FallbackManager,
    Model,
    Node,
    Provider,
    Route,
    RouteRequest,
    RouteScheduler,
    RoutingPolicy,
    ScoredCandidate,
)

__all__ = [
    "FallbackManager",
    "Model",
    "Node",
    "Provider",
    "Route",
    "RouteRequest",
    "RouteScheduler",
    "RoutingPolicy",
    "ScoredCandidate",
]
