"""swarm 基础测试"""

from aetherforge.swarm import cli


class TestSwarmCLI:
    """Swarm CLI 测试"""

    def test_import(self):
        """测试导入"""
        assert cli is not None


class TestSwarm:
    """Swarm 功能测试"""

    def test_initialization(self):
        """测试初始化"""
        from aetherforge.swarm import __init__
        assert __init__ is not None
