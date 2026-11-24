import numpy as np


class AntColonyPathPlanner:

  def __init__(
      self,
      grid,
      start,
      goal,
      n_ants=40,
      n_best=5,
      n_iterations=60,
      decay=0.15,
      alpha=1.0,
      beta=2.5,
      max_steps=200,
  ):
    self.grid = grid
    self.rows, self.cols = grid.shape
    self.start = start
    self.goal = goal
    self.n_ants = n_ants
    self.n_best = n_best
    self.n_iterations = n_iterations
    self.decay = decay
    self.alpha = alpha
    self.beta = beta
    self.max_steps = max_steps

    # Initialize pheromone matrix (0 on obstacles, 1.0 on free cells)
    self.pheromone = np.where(grid == 0, 1.0, 0.0)

    # 8-directional movement (orthogonal + diagonal)
    self.directions = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )

  def run(self):
    best_path = None
    best_cost = np.inf

    for _ in range(self.n_iterations):
      valid_paths = []

      for _ in range(self.n_ants):
        path, cost = self.build_path()
        if path is not None:
          valid_paths.append((path, cost))
          if cost < best_cost:
            best_cost = cost
            best_path = path

      self.pheromone *= 1.0 - self.decay

      if valid_paths:
        self.update_pheromone(valid_paths)

    return best_path, float(best_cost)

  def build_path(self):
    curr = self.start
    path = [curr]
    visited = {curr}

    for _ in range(self.max_steps):
      if curr == self.goal:
        return self.prune_loops(path)

      neighbors = self.get_valid_neighbors(curr, visited)
      if not neighbors:
        return None, np.inf

      next_cell = self.pick_move(neighbors)
      path.append(next_cell)
      visited.add(next_cell)
      curr = next_cell

    return None, np.inf

  def get_valid_neighbors(self, cell, visited):
    neighbors = []
    r, c = cell
    for dr, dc in self.directions:
      nr, nc = r + dr, c + dc
      if 0 <= nr < self.rows and 0 <= nc < self.cols:
        if self.grid[nr, nc] == 0 and (nr, nc) not in visited:
          neighbors.append((nr, nc))
    return neighbors

  def pick_move(self, neighbors):
    gr, gc = self.goal
    weights = []

    for nr, nc in neighbors:
      tau = self.pheromone[nr, nc] ** self.alpha
      dist_to_goal = np.hypot(gr - nr, gc - nc)
      eta = (1.0 / (dist_to_goal + 1e-5)) ** self.beta
      weights.append(tau * eta)

    weights_arr = np.array(weights, dtype=float)
    probs = weights_arr / weights_arr.sum()

    idx = int(np.random.choice(len(neighbors), p=probs))
    return neighbors[idx]

  def prune_loops(self, path):
    pruned = [path[0]]
    i = 0
    while i < len(path) - 1:
      r1, c1 = path[i]
      furthest = i + 1
      for j in range(len(path) - 1, i + 1, -1):
        r2, c2 = path[j]
        if abs(r2 - r1) <= 1 and abs(c2 - c1) <= 1:
          furthest = j
          break
      pruned.append(path[furthest])
      i = furthest

    total_cost = 0.0
    for a, b in zip(pruned, pruned[1:]):
      r1, c1 = a
      r2, c2 = b
      total_cost += float(np.hypot(r2 - r1, c2 - c1))

    return pruned, total_cost

  def update_pheromone(self, valid_paths):
    sorted_paths = sorted(valid_paths, key=lambda item: item[1])
    for path, cost in sorted_paths[: self.n_best]:
      deposit = 10.0 / cost
      for r, c in path:
        self.pheromone[r, c] += deposit


# --- Example Usage ---
if __name__ == "__main__":
  np.random.seed(42)

  # 10x10 Occupancy Grid (0 = free space, 1 = obstacle)
  grid_layout = """
        0 0 0 0 0 0 0 0 0 0
        0 1 1 1 0 0 1 1 0 0
        0 0 0 1 0 0 1 0 0 0
        0 0 0 1 0 0 1 0 1 0
        0 1 0 0 0 0 0 0 1 0
        0 1 1 1 1 0 1 1 1 0
        0 0 0 0 1 0 0 0 0 0
        0 0 1 0 1 0 1 1 0 0
        0 0 1 0 0 0 0 1 0 0
        0 0 0 0 0 0 0 0 0 0
    """
  occupancy_grid = np.array(
      [list(map(int, line.split())) for line in grid_layout.strip().splitlines()]
  )

  start_pos = (0, 0)
  goal_pos = (9, 9)

  planner = AntColonyPathPlanner(
      grid=occupancy_grid,
      start=start_pos,
      goal=goal_pos,
      n_ants=40,
      n_best=5,
      n_iterations=60,
      decay=0.15,
      alpha=1.0,
      beta=2.5,
  )

  best_path, path_length = planner.run()

  if best_path:
    print(f"Collision-Free Path: {best_path}")
    print(f"Total Path Length:   {path_length:.2f}")

    visual = np.full(occupancy_grid.shape, ".", dtype=str)
    visual[occupancy_grid == 1] = "#"
    for r, c in best_path:
      visual[r, c] = "*"
    visual[start_pos] = "S"
    visual[goal_pos] = "G"

    print("\nGrid Map:")
    for row in visual:
      print(" ".join(row))
  else:
    print("No collision-free path found.")
