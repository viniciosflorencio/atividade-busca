from puzzle.base_search import BaseSearch
from puzzle.state import State
from puzzle.result import SearchResult

DEFAULT_DEPTH_LIMIT = 50


class DFS(BaseSearch):
    """Busca em Profundidade com limite: expande o nó mais fundo (pilha LIFO).

    Não é ótima: devolve a primeira solução encontrada dentro do limite.
    """

    def __init__(self, depth_limit: int = DEFAULT_DEPTH_LIMIT):
        self.depth_limit = depth_limit

    def search(self, initial: State) -> SearchResult:
        stack = [(initial, 0)]
        # Menor profundidade em que cada estado já foi alcançado. Um conjunto
        # simples de visitados não basta: um estado visto perto do limite
        # bloquearia um caminho mais curto até ele, que ainda teria
        # profundidade sobrando para chegar ao objetivo.
        best_depth = {initial.tiles: 0}
        unreached = self.depth_limit + 1  # profundidade "infinita": além do limite
        nodes_expanded = 0
        nodes_generated = 1  # o estado inicial
        max_frontier_size = 1

        while stack:
            state, depth = stack.pop()

            # Entrada obsoleta: o estado já foi alcançado por caminho mais curto.
            if depth > best_depth[state.tiles]:
                continue

            if state.is_goal:
                return SearchResult(
                    solution=state,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    max_frontier_size=max_frontier_size,
                    depth=depth,
                )

            if depth >= self.depth_limit:
                continue

            nodes_expanded += 1
            child_depth = depth + 1
            # reversed: o primeiro vizinho fica no topo e é explorado primeiro.
            for child in reversed(state.neighbors()):
                nodes_generated += 1
                if child_depth < best_depth.get(child.tiles, unreached):
                    best_depth[child.tiles] = child_depth
                    stack.append((child, child_depth))

            max_frontier_size = max(max_frontier_size, len(stack))

        return SearchResult(
            solution=None,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=max_frontier_size,
        )
