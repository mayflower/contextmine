from contextmine_core.models import TwinLayer
from contextmine_core.twin.service import infer_edge_layers, infer_node_layers


def test_infer_node_layers_for_code_entities_exposes_architecture_layers() -> None:
    layers = infer_node_layers("function")
    assert TwinLayer.CODE_CONTROLFLOW in layers
    assert TwinLayer.COMPONENT_INTERFACE in layers
    assert TwinLayer.DOMAIN_CONTAINER in layers


def test_infer_edge_layers_for_symbol_edges_exposes_architecture_layers() -> None:
    layers = infer_edge_layers("file_defines_symbol")
    assert TwinLayer.CODE_CONTROLFLOW in layers
    assert TwinLayer.COMPONENT_INTERFACE in layers
    assert TwinLayer.DOMAIN_CONTAINER in layers

