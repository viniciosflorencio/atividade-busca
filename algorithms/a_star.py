import heapq
from puzzle.base_search import BaseSearch
from puzzle.state import State, GOAL_STATE
from puzzle.result import SearchResult

# Posição (linha, coluna) de cada peça no estado objetivo.
GOAL_POSITION = {tile: divmod(index, 3) for index, tile in enumerate(GOAL_STATE)}


class AStar(BaseSearch):
    """Busca A*: expande o nó de menor f(n) = g(n) + h(n).

    Ótima, pois a distância de Manhattan é admissível e consistente.
    """

    def heuristic(self, state: State) -> int:
        """Distância de Manhattan: soma das distâncias de cada peça ao objetivo.

        O espaço vazio não entra na soma. Cada movimento desloca uma única
        peça em uma casa, então o valor nunca superestima o custo real.
        """
        total = 0
        for index, tile in enumerate(state.tiles):
            if tile == 0:
                continue
            row, col = divmod(index, 3)
            goal_row, goal_col = GOAL_POSITION[tile]
            total += abs(row - goal_row) + abs(col - goal_col)
        return total

    def search(self, initial: State) -> SearchResult:
        # Fila de prioridade de tuplas (f, h, ordem, estado). Em empates de f,
        # prefere o menor h (mais perto do objetivo); `order` é o desempate final.
        h = self.heuristic(initial)
        order = 0
        frontier = [(initial.cost + h, h, order, initial)]
        best_cost = {initial: initial.cost}
        nodes_expanded = 0
        nodes_generated = 1  # o estado inicial
        max_frontier_size = 1

        while frontier:
            _, _, _, state = heapq.heappop(frontier)

            # Entrada obsoleta: o estado já foi alcançado com custo menor.
            if state.cost > best_cost[state]:
                continue

            # O teste é feito ao retirar da fila (e não ao gerar o nó):
            # é isso que garante que a solução seja ótima.
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
                if child.cost < best_cost.get(child, float("inf")):
                    best_cost[child] = child.cost
                    h = self.heuristic(child)
                    order += 1
                    heapq.heappush(frontier, (child.cost + h, h, order, child))

            max_frontier_size = max(max_frontier_size, len(frontier))

        return SearchResult(
            solution=None,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=max_frontier_size,
        )
