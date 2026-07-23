import copy
import warnings
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
from torch import Tensor
from torch.nn import Module, ModuleDict, ModuleList, Sequential

try:
    from torch.fx import Graph, GraphModule, Node
except (ImportError, ModuleNotFoundError, AttributeError):
    GraphModule, Graph, Node = 'GraphModule', 'Graph', 'Node'


class Transformer:
    def __init__(
        self,
        module: Module,
        input_map: Optional[Dict[str, str]] = None,
        debug: bool = False,
    ):
        self.module = module
        self.gm = symbolic_trace(module)
        self.input_map = input_map
        self.debug = debug


    def placeholder(self, node: Node, target: Any, name: str):
        pass

    def get_attr(self, node: Node, target: Any, name: str):
        pass

    def call_message_passing_module(self, node: Node, target: Any, name: str):
        pass

    def call_global_pooling_module(self, node: Node, target: Any, name: str):
        pass

    def call_module(self, node: Node, target: Any, name: str):
        pass

    def call_method(self, node: Node, target: Any, name: str):
        pass

    def call_function(self, node: Node, target: Any, name: str):
        pass

    def output(self, node: Node, target: Any, name: str):
        pass

    def init_submodule(self, module: Module, target: str) -> Module:
        return module


    @property
    def graph(self) -> Graph:
        return self.gm.graph

    def transform(self) -> GraphModule:
        r"""Transforms :obj:`self.module` and returns a transformed
        :class:`torch.fx.GraphModule`.
        """
        if self.debug:
            self.graph.print_tabular()
            print()
            code = self.graph.python_code('self')
            print(code.src if hasattr(code, 'src') else code)

        self._state = copy.copy(self.input_map or {})

        for node in list(self.graph.nodes):
            if node.op == 'call_function' and 'training' in node.kwargs:
                warnings.warn(
                    f"Found function '{node.name}' with keyword "
                    f"argument 'training'. During FX tracing, this "
                    f"will likely be baked in as a constant value. "
                    f"Consider replacing this function by a module "
                    f"to properly encapsulate its training flag.",
                    stacklevel=2)

            if node.op == 'placeholder':
                if node.name not in self._state:
                    if 'edge' in node.name or 'adj' in node.name:
                        self._state[node.name] = 'edge'
                    else:
                        self._state[node.name] = 'node'
            elif is_message_passing_op(self.module, node.op, node.target):
                self._state[node.name] = 'node'
            elif is_global_pooling_op(self.module, node.op, node.target):
                self._state[node.name] = 'graph'
            elif node.op in ['call_module', 'call_method', 'call_function']:
                if self.has_edge_level_arg(node):
                    self._state[node.name] = 'edge'
                elif self.has_node_level_arg(node):
                    self._state[node.name] = 'node'
                else:
                    self._state[node.name] = 'graph'

        for node in list(self.graph.nodes):
            op = node.op
            if is_message_passing_op(self.module, op, node.target):
                op = 'call_message_passing_module'
            elif is_global_pooling_op(self.module, op, node.target):
                op = 'call_global_pooling_module'
            getattr(self, op)(node, node.target, node.name)

        for node in reversed(list(self.graph.nodes)):
            try:
                if node.op not in ['placeholder', 'output']:
                    self.graph.erase_node(node)
            except RuntimeError:
                pass

        for target, submodule in dict(self.module._modules).items():
            self.gm._modules[target] = self._init_submodule(submodule, target)

        del self._state

        if self.debug:
            self.gm.graph.print_tabular()
            print()
            code = self.graph.python_code('self')
            print(code.src if hasattr(code, 'src') else code)

        self.gm.graph.lint()
        self.gm.recompile()

        return self.gm

    def _init_submodule(self, module: Module, target: str) -> Module:
        if isinstance(module, ModuleList) or isinstance(module, Sequential):
            return ModuleList([
                self._init_submodule(submodule, f'{target}.{i}')
                for i, submodule in enumerate(module)
            ])
        elif isinstance(module, ModuleDict):
            return ModuleDict({
                key:
                self._init_submodule(submodule, f'{target}.{key}')
                for key, submodule in module.items()
            })
        else:
            return self.init_submodule(module, target)

    def _is_level(self, node: Node, name: str) -> bool:
        return self._state[node.name] == name

    def _has_level_arg(self, node: Node, name: str) -> bool:
        def _recurse(value: Any) -> bool:
            if isinstance(value, Node):
                return getattr(self, f'is_{name}_level')(value)
            elif isinstance(value, dict):
                return any([_recurse(v) for v in value.values()])
            elif isinstance(value, (list, tuple)):
                return any([_recurse(v) for v in value])
            else:
                return False

        return (any([_recurse(value) for value in node.args])
                or any([_recurse(value) for value in node.kwargs.values()]))

    def is_node_level(self, node: Node) -> bool:
        return self._is_level(node, name='node')

    def is_edge_level(self, node: Node) -> bool:
        return self._is_level(node, name='edge')

    def is_graph_level(self, node: Node) -> bool:
        return self._is_level(node, name='graph')

    def has_node_level_arg(self, node: Node) -> bool:
        return self._has_level_arg(node, name='node')

    def has_edge_level_arg(self, node: Node) -> bool:
        return self._has_level_arg(node, name='edge')

    def has_graph_level_arg(self, node: Node) -> bool:
        pass

    def find_by_name(self, name: str) -> Optional[Node]:
        for node in self.graph.nodes:
            if node.name == name:
                return node
        return None

    def find_by_target(self, target: Any) -> Optional[Node]:
        for node in self.graph.nodes:
            if node.target == target:
                return node
        return None

    def replace_all_uses_with(self, to_replace: Node, replace_with: Node):
        pass


def symbolic_trace(
        module: Module,
        concrete_args: Optional[Dict[str, Any]] = None) -> GraphModule:

    try:
        import torch.fx._symbolic_trace as st
    except (ImportError, ModuleNotFoundError):
        import torch.fx.symbolic_trace as st

    from torch_geometric.nn import Aggregation

    class Tracer(torch.fx.Tracer):
        def is_leaf_module(self, module: Module, *args, **kwargs) -> bool:
            return not isinstance(module, torch.nn.Sequential)

        @st.compatibility(is_backward_compatible=True)
        def trace(self, root: Union[torch.nn.Module, Callable[..., Any]],
                  concrete_args: Optional[Dict[str, Any]] = None) -> Graph:

            if isinstance(root, torch.nn.Module):
                self.root = root
                fn = type(root).forward
                self.submodule_paths = {
                    mod: name
                    for name, mod in root.named_modules()
                }
            else:
                self.root = torch.nn.Module()
                fn = root

            tracer_cls: Optional[Type['Tracer']] = getattr(
                self, '__class__', None)
            self.graph = Graph(tracer_cls=tracer_cls)

            self.tensor_attrs: Dict[Union[Tensor, st.ScriptObject], str] = {}

            def collect_tensor_attrs(m: torch.nn.Module,
                                     prefix_atoms: List[str]):
                for k, v in m.__dict__.items():
                    if isinstance(v, (Tensor, st.ScriptObject)):
                        self.tensor_attrs[v] = '.'.join(prefix_atoms + [k])
                for k, v in m.named_children():
                    collect_tensor_attrs(v, prefix_atoms + [k])

            collect_tensor_attrs(self.root, [])

            assert isinstance(fn, st.FunctionType)

            fn_globals = fn.__globals__  # run before it gets patched
            fn, args = self.create_args_for_root(
                fn, isinstance(root, torch.nn.Module), concrete_args)

            parameter_proxy_cache: Dict[str, st.Proxy] = {
            }  # Reduce number of get_attr calls

            @st.functools.wraps(st._orig_module_getattr)
            def module_getattr_wrapper(mod, attr):
                pass

            @st.functools.wraps(st._orig_module_call)
            def module_call_wrapper(mod, *args, **kwargs):
                pass

            with st._Patcher() as patcher:
                patcher.patch_method(torch.nn.Module, "__getattr__",
                                     module_getattr_wrapper, deduplicate=False)
                patcher.patch_method(torch.nn.Module, "__call__",
                                     module_call_wrapper, deduplicate=False)
                patcher.patch_method(Aggregation, "__call__",
                                     module_call_wrapper, deduplicate=False)
                st._patch_wrapped_functions(patcher)
                st._autowrap_check(patcher, fn_globals,
                                   self._autowrap_function_ids)
                for module in self._autowrap_search:
                    st._autowrap_check(patcher, module.__dict__,
                                       self._autowrap_function_ids)
                self.create_node(
                    'output', 'output', (self.create_arg(fn(*args)), ), {},
                    type_expr=fn.__annotations__.get('return', None))

            self.submodule_paths = None

            return self.graph

    return GraphModule(module, Tracer().trace(module, concrete_args))


def get_submodule(module: Module, target: str) -> Module:
    out = module
    for attr in target.split('.'):
        out = getattr(out, attr)
    return out


def is_message_passing_op(module: Module, op: str, target: str) -> bool:
    from torch_geometric.nn import MessagePassing
    if op == 'call_module':
        return isinstance(get_submodule(module, target), MessagePassing)
    return False


def is_global_pooling_op(module: Module, op: str, target: str) -> bool:
    from torch_geometric.nn import Aggregation
    if op == 'call_module':
        return isinstance(get_submodule(module, target), Aggregation)
    return False
