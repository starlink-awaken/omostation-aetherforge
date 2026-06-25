"""swarm 基础测试"""


class TestSwarm:
    """Swarm 功能测试"""

    def test_initialization(self):
        """测试初始化"""
        from aetherforge.swarm import __init__

        assert __init__ is not None
