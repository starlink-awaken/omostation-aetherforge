"""gateway 基础测试"""

from aetherforge.gateway import cli


class TestGatewayCLI:
    """Gateway CLI 测试"""
    
    def test_import(self):
        """测试导入"""
        assert cli is not None


class TestGateway:
    """Gateway 功能测试"""
    
    def test_initialization(self):
        """测试初始化"""
        from aetherforge.gateway import __init__
        assert __init__ is not None
