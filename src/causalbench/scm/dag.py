from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

@dataclass(frozen=True)
class DAG:
    """implementation of a DAG as an adjacency dict"""
    _children: Dict[str, Tuple[str,...]]
    _parents: Dict[str, Tuple[str,...]]

    @staticmethod
    def from_edges(edges: Iterable[Tuple[str, str]]) -> "DAG":
        """Construct a DAG from an iterable of (parent, child) edges."""
        edges = list(edges)
        # infer nodes
        nodes = set()
        for p, c in edges:
            nodes.add(p)
            nodes.add(c)

        children_map: Dict[str, list[str]] = {n: [] for n in nodes}
        parents_map: Dict[str, list[str]] = {n: [] for n in nodes}

        for p, c in edges:
            if p == c:
                raise ValueError(f"self loops not allowed: {p} -> {c}")
            
            children_map[p].append(c)
            parents_map[c].append(p)

        return DAG(
            _children={k: tuple(children_map[k]) for k in nodes},
            _parents={k: tuple(parents_map[k]) for k in nodes},
        )
    
    def nodes(self) -> Tuple[str, ...]:
        """return the nodes in the DAG"""
        return tuple(self._children.keys())
    
    def children_of(self, node: str) -> Tuple[str, ...]:
        """return the children of a node"""
        return self._children[node]
    
    def parents_of(self, node: str) -> Tuple[str, ...]:
        """return the parents of a node"""
        return self._parents[node]
    
    def edges(self) ->Tuple[Tuple[str, str], ...]:
        """return the edges in the DAG"""
        edges = []
        for parent,children in self._children.items():
            for child in children:
                edges.append((parent, child))
        return tuple(edges)
    
    def topological_sort(self) -> Tuple[str, ...]:
        """ sort the dag topologically , also check for cycles"""

        sorted_nodes = []
        visited = set()
        visiting = set()

        def dfs(node: str):
            if node in visited:
                return
            if node in visiting:
                raise ValueError(" cycle detected ")
            
            visiting.add(node)
            for child in self.children_of(node):
                dfs(child)
            visiting.remove(node)
            visited.add(node)
            sorted_nodes.append(node)

        for node in self.nodes():
            if node not in visited:
                dfs(node)
        sorted_nodes.reverse()
        return(tuple(sorted_nodes))
    

            
         

        

        

    


    
