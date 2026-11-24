from collections import deque
from puzzle.base_search import BaseSearch
from puzzle.state import State
from puzzle.result import SearchResult


class BFS(BaseSearch):
    """Busca em Largura: expande sempre o nó mais raso (fronteira FIFO).

    Completa e ótima no 8-puzzle, pois todo movimento tem custo 1.
    """

    def search(self, initial: State) -> SearchResult:
        frontier = deque([initial])
        visited = {initial}
        nodes_expanded = 0
        nodes_generated = 1  # o estado inicial
        max_frontier_size = 1

        while frontier:
            state = frontier.popleft()

            if state.is_goal:
                return SearchResult(
                    solution=state,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    max_frontier_size=max_frontier_size,
                    depth=state.cost - initial.cost,
                )

            nodes_expanded += 1
            for child in state.neighbors():
                nodes_generated += 1
                # Marcar ao enfileirar evita o mesmo estado repetido na fila.
                if child not in visited:
                    visited.add(child)
                    frontier.append(child)

            max_frontier_size = max(max_frontier_size, len(frontier))

        return SearchResult(
            solution=None,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=max_frontier_size,
        )
