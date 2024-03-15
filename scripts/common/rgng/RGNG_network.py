from networkx import Graph

class RGNG_Graph(Graph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
    
    def add_node(self, node_for_adding, **attr):
        for i in self.nodes():
            self.nodes[i]['prenode_ranking'] += 1
        return super().add_node(node_for_adding, prenode_ranking=0, **attr)