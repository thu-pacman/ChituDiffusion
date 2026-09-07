"""SGLang integration entry points.

Importing this package does not import SGLang.  The registry mutation is
performed only when :func:`install_sglang_h3_adapter` is called.
"""

from .bootstrap import install_sglang_h3_adapter

__all__ = ["install_sglang_h3_adapter"]

