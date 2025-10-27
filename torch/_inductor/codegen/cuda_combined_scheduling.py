# mypy: allow-untyped-defs
from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING, Union

from torch._inductor.ir import FusableUserDefinedTritonKernel, IRNode, TensorBox

from torch._inductor.ir import MutationOutput

from ..scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusableUserDefinedKernelSchedulerNode,
    FusedSchedulerNode,
    Scheduler,
    SchedulerBuffer,
    SchedulerNode,
)
from .cuda.cuda_cpp_scheduling import CUDACPPScheduling
from .cutedsl.cutedsl_scheduling import CuteDSLScheduling
from .rocm.rocm_cpp_scheduling import ROCmCPPScheduling
from .triton import TritonScheduling


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing_extensions import TypeAlias

    from sympy import Expr

    import torch
    from torch.utils._ordered_set import OrderedSet

    from .common import BackendFeature

    _IntLike: TypeAlias = Union[int, Expr]


class CUDACombinedScheduling(BaseScheduling):
    """
    Scheduler for CUDA Kernels, which delegates calls as appropriate
    to the CUDA-C++ and Triton Schedulers, which both work for CUDA devices
    and use a unified-wrapper for codegen.

    If Scheduling code needs to be specialized for the case of mixed Triton / CUDA C++ code,
    this would also be the place to do it.
    """

    def __init__(self, scheduler: Optional[Scheduler]) -> None:
        super().__init__(scheduler)
        self._triton_scheduling = TritonScheduling(scheduler)
        self._cuda_cpp_scheduling = CUDACPPScheduling(scheduler)
        self._rocm_cpp_scheduling = ROCmCPPScheduling(scheduler)
        self._cutedsl_scheduling = CuteDSLScheduling(scheduler)

    def get_backend_features(self, device: torch.device) -> OrderedSet[BackendFeature]:
        return self._triton_scheduling.get_backend_features(device)

    def choose_node_backend(self, node: BaseSchedulerNode) -> BaseScheduling:
        if self._cuda_cpp_scheduling.is_cuda_cpp_template(node):
            return self._cuda_cpp_scheduling
        if self._rocm_cpp_scheduling.is_rocm_cpp_template(node):
            return self._rocm_cpp_scheduling
        if self._cutedsl_scheduling.is_cutedsl_template(node):
            return self._cutedsl_scheduling
        return self._triton_scheduling

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        if self._cuda_cpp_scheduling.can_fuse_vertical(node1, node2):
            return True
        elif self._cuda_cpp_scheduling.is_cuda_cpp_template(
            node1
        ) or self._cuda_cpp_scheduling.is_cuda_cpp_template(node2):
            return False
        # CuteDSL doesn't support vertical fusion currently
        elif self._cutedsl_scheduling.is_cutedsl_template(
            node1
        ) or self._cutedsl_scheduling.is_cutedsl_template(node2):
            return False
        return self._triton_scheduling.can_fuse_vertical(node1, node2)

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        for node in (node1, node2):
            if self._cuda_cpp_scheduling.is_cuda_cpp_template(node):
                return self._cuda_cpp_scheduling.can_fuse_horizontal(
                    node1, node2
                )  # always False at the moment
            if self._cutedsl_scheduling.is_cutedsl_template(node):
                return self._cutedsl_scheduling.can_fuse_horizontal(
                    node1, node2
                )  # always False at the moment
        return self._triton_scheduling.can_fuse_horizontal(node1, node2)

    def group_fn(
        self, sizes: Sequence[Sequence[_IntLike]]
    ) -> tuple[tuple[_IntLike, ...], ...]:
        return self._triton_scheduling.group_fn(sizes)

    def codegen_fusable_user_defined_triton_kernel(
        self,
        scheduler_node
    ):
        """
        NOTE: This is for a simple epilogue fusion (user-defined = producer, scheduler-node = consumer).

        Two main goals, in order:

            1. BUFFER MANAGEMENT: 
                - Resolve intermediate buffers, 
                - and correct input/output buffers of `FusedSchedulerNode`.
            2. MERGE COMPUTATION:
                - Combine the epilogue's computation into the user kernel's source code.

        """
        print("INSIDE CODEGEN_FUSABLE_USER_DEFINED_TRITON_KERNEL")
        from ..virtualized import V

        nodes = scheduler_node.get_nodes()

        user_defined_scheduler_node = nodes[0]
        assert isinstance(user_defined_scheduler_node, FusableUserDefinedKernelSchedulerNode)

        iir_node = user_defined_scheduler_node.node
        assert isinstance(iir_node, FusableUserDefinedTritonKernel)
        assert iir_node is not None # For LSP

        kernel_wrapper, render = iir_node.make_kernel_render()

        # Remaning nodes are epilogue nodes
        epilogue_nodes = [n for n in nodes[1:] if isinstance(n, SchedulerNode)]

        # Triton kernels do not return an output, but rather mutate arguments.
        # However, this is somewhat represented as an output in the dep-graph.

        # We will work from the persepctive of the consumer (epilogue kernel), to avoid 
        # changing dependencies of subeqeuent operations (that is, the subsequent to this fused kernel). 
        # Thus, we will work backwards from the epilogue: 
        #       (1) get the ouput buffer of the subsequent.
        #       (2) redirect the user-kernel with (1)'s buffer.
        #           this is apart of the mutated buffers of the user-kernels.
        #       (3) mark intermediate buffers as removed, so they're not allocated.

        user_kernel_output_buf = user_defined_scheduler_node.get_outputs()[0]
        assert isinstance(user_kernel_output_buf, SchedulerBuffer), f"{type(user_kernel_output_buf)}"

        # Get mutable arguments of user-kernel. These are in the form of TensorBox.
        # For our example we only have one argument/buffer that is mutated.
        assert len(iir_node.mutable_args) == 1
        mutated_buffer_name = iir_node.mutable_args[0].get_name()
        print(iir_node.mutable_args)
        assert isinstance(iir_node.mutable_args[0], TensorBox)
        
        # Get epilogue SchedulerBuffer.
        epilogue_output_buf = epilogue_nodes[0].get_outputs()[0]
        epilogue_output_name = epilogue_output_buf.get_name()
        
        # Replace in kwargs.
        epilogue_ir_buf = epilogue_output_buf.node
        # TODO: I dont like the way this is done. Perhaps we add some mapping to the iir_node
        #       during initialisation or analysis.
        for key, value in iir_node.kwargs.items():
            # TODO: Is there a case where an IRNode for Triton kernel will not have this implemented?
            if isinstance(value, IRNode) and value.get_name() == mutated_buffer_name:
                iir_node.kwargs[key] = epilogue_ir_buf
                break
        else:
            assert False
        
        # Mark intermediate buffers as removed
        kernel_wrapper.removed_buffers.add(mutated_buffer_name)  # buf1
        kernel_wrapper.removed_buffers.add(user_kernel_output_buf.get_name())  # buf2

        # Currently, the render() function and hooks are hardcoded for the
        # specific GeLU + scale example. A general implementation would need
        # to introspect both the user-defined and epilogue bodies and codegen the
        # appropriate Triton code.
        with kernel_wrapper:
            partial_code = render()
            partial_code.finalize_hook("<KERNEL_BODY>")
            partial_code.finalize_hook("<EPILOGUE_FUSION>")

            src_code = partial_code.code

            print(src_code)

        # kernel_wrapper.jit_kernel.__dict__['src'] = src_code
        # kernel_wrapper.jit_kernel.src = src_code
        wrapper = V.graph.wrapper_code
        user_kernel = kernel_wrapper.user_defined_kernel

        (
            kernel_name,
            triton_meta,
            extra_launch_args,
        ) = wrapper.define_user_defined_triton_kernel(
            kernel_wrapper.jit_kernel,
            kernel_wrapper.configs,
            user_kernel.kwargs,
            kernel_wrapper.restore_value_args,
            kernel_wrapper.reset_to_zero_args,
            user_kernel.grid,
            src_code # TODO: CHECK THIS LOGIC !!!!
        )

        kernel_wrapper.kernel_name = kernel_name
        kernel_wrapper.extra_launch_args = extra_launch_args

        with V.set_kernel_handler(kernel_wrapper):
            for node in scheduler_node.get_nodes():
                node.mark_run()

        # kernel_wrapper.call_kernel(kernel_name, scheduler_node)
        kernel_wrapper.call_kernel(kernel_name)

        V.graph.removed_buffers |= kernel_wrapper.removed_buffers
        V.graph.inplaced_to_remove |= kernel_wrapper.inplaced_to_remove
        self.free_buffers_in_scheduler()


    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ) -> Optional[str]:
        if self._cuda_cpp_scheduling.is_cuda_cpp_template(template_node):
            assert not prologue_nodes
            return self._cuda_cpp_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )
        elif self._rocm_cpp_scheduling.is_rocm_cpp_template(template_node):
            assert not epilogue_nodes
            assert not prologue_nodes
            return self._rocm_cpp_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )
        elif self._cutedsl_scheduling.is_cutedsl_template(template_node):
            # TODO remove this when we add epilogue support
            assert not epilogue_nodes
            assert not prologue_nodes
            return self._cutedsl_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )
        else:
            return self._triton_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )

    def codegen_node(self, node: Union[FusedSchedulerNode, SchedulerNode]) -> None:
        return self._triton_scheduling.codegen_node(node)

    def codegen_sync(self) -> None:
        return self._triton_scheduling.codegen_sync()

    def flush(self) -> None:
        return self._triton_scheduling.flush()

    def codegen_combo_kernel(self, *args: Any, **kwargs: Any) -> None:
        return self._triton_scheduling.codegen_combo_kernel(*args, **kwargs)

    def benchmark_fused_nodes(
        self, nodes: Sequence[BaseSchedulerNode]
    ) -> tuple[float, str]:
        return self._triton_scheduling.benchmark_fused_nodes(nodes)

    def benchmark_codegened_module(self, module):
        return self._triton_scheduling.benchmark_codegened_module(module)

    def generate_kernel_code_from_nodes(
        self,
        nodes: Sequence[Any],
        benchmark_kernel: bool = False,
        hint_override: Optional[int] = None,
    ) -> str:
        return self._triton_scheduling.generate_kernel_code_from_nodes(
            nodes, benchmark_kernel, hint_override=hint_override
        )

    def benchmark_combo_kernel(
        self, node_list: Sequence[BaseSchedulerNode]
    ) -> tuple[float, float, list[Optional[str]]]:
        return self._triton_scheduling.benchmark_combo_kernel(node_list)
