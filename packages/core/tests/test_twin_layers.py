from contextmine_core.models import TwinLayer
from contextmine_core.twin.service import infer_edge_layers, infer_node_layers


def test_infer_node_layers_for_code_entities_remains_code_layer_only() -> None:
    layers = infer_node_layers("function")
    assert layers == {TwinLayer.CODE_CONTROLFLOW}


def test_infer_edge_layers_for_symbol_edges_remains_code_layer_only() -> None:
    layers = infer_edge_layers("file_defines_symbol")
    assert layers == {TwinLayer.CODE_CONTROLFLOW}
