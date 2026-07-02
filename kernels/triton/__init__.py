"""Triton kernels for the structured-sparse Expander SAE.

Importing this package does not import Triton itself; importing individual
kernel modules does. This lets you build the autograd wrapper on a CPU-only
host and dispatch to the right backend at module-load time.
"""
