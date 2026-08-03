"""mesh 基础测试"""

from aetherforge.mesh import cli


class TestMeshCLI:
    """Mesh CLI 测试"""

    def test_import(self):
        """测试导入"""
        assert cli is not None


class TestMesh:
    """Mesh 功能测试"""

    def test_initialization(self):
        """测试初始化"""
        from aetherforge.mesh import __init__  # type: ignore[reportAttributeAccessIssue]

        assert __init__ is not None
