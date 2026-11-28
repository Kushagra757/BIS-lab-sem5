import numpy as np
from concurrent.futures import ThreadPoolExecutor

# ==============================================================================
# STEP 1: DEFINE THE PROBLEM (Multi-Objective Delivery Time & Fuel Cost)
# ==============================================================================
class FoodDeliveryProblem:
    def __init__(self, depot, orders, weights, traffic_factors, n_drivers=2, fuel_rate=0.08):
        self.depot = np.array(depot)
        self.orders = np.array(orders)            # (N, 2) coordinates of customers
        self.weights = np.array(weights)          # (N,) order weights in kg
        self.traffic = np.array(traffic_factors)  # (N,) congestion multiplier (>= 1.0)
        self.n_orders = len(orders)
        self.n_drivers = n_drivers
        self.fuel_rate = fuel_rate                # Base liters per km

    def decode_cell(self, cell_state):
        """
        Maps continuous cell vector [0, 1]^N to discrete driver assignments & route sequence:
        - Integer part of (state * n_drivers) -> assigned driver ID
        - Fractional part -> visit priority order
        """
        scaled = cell_state * self.n_drivers
        driver_ids = np.floor(scaled).astype(int) % self.n_drivers
        priorities = scaled - np.floor(scaled)

        routes = []
        for d in range(self.n_drivers):
            assigned = np.where(driver_ids == d)[0]
            # Sort assigned orders by fractional priority
            ordered_stops = assigned[np.argsort(priorities[assigned])]
            routes.append(ordered_stops)
        return routes

    def evaluate(self, cell_state, time_weight=0.6, fuel_weight=0.4):
        """Minimizes weighted sum of Total Delivery Time (mins) and Fuel Used (liters)."""
        routes = self.decode_cell(cell_state)
        total_time = 0.0
        total_fuel = 0.0
        base_speed_km_min = 0.5  # 30 km/h average city speed

        for route in routes:
            if len(route) == 0:
                continue
            curr_pos = self.depot
            curr_load = np.sum(self.weights[route])  # Total food weight carried at start

            for order_idx in route:
                dest_pos = self.orders[order_idx]
                dist = np.linalg.norm(dest_pos - curr_pos)

                # Delivery time increases with traffic congestion factor
                leg_time = (dist / base_speed_km_min) * self.traffic[order_idx]
                total_time += leg_time

                # Fuel consumption increases with distance and carried payload weight
                load_factor = 1.0 + 0.015 * curr_load
                total_fuel += dist * self.fuel_rate * load_factor

                # Drop off order and update position
                curr_load -= self.weights[order_idx]
                curr_pos = dest_pos

            # Return trip to restaurant/depot (empty payload)
            return_dist = np.linalg.norm(self.depot - curr_pos)
            total_fuel += return_dist * self.fuel_rate

        fitness = (time_weight * total_time) + (fuel_weight * total_fuel * 10.0)
        return fitness, total_time, total_fuel


# ==============================================================================
# STEPS 2–7: PARALLEL CELLULAR ALGORITHM (PCA)
# ==============================================================================
class ParallelCellularOptimizer:
    def __init__(self, problem, grid_shape=(8, 8), max_iters=120, mutation_rate=0.15, seed=42):
        self.problem = problem
        self.rows, self.cols = grid_shape
        self.dim = problem.n_orders
        self.max_iters = max_iters
        self.mutation_rate = mutation_rate
        self.rng = np.random.default_rng(seed)

    def get_moore_neighbors(self, r, c):
        """Returns coordinates of the 8 surrounding cells on a wrap-around (toroidal) grid."""
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbors.append(((r + dr) % self.rows, (c + dc) % self.cols))
        return neighbors

    def _update_single_cell(self, args):
        """Cellular transition rule: diffuse state from best neighbor + local perturbation."""
        r, c, grid, fitness_grid, step_scale, local_seed = args
        cell_rng = np.random.default_rng(local_seed)

        # Find the fittest neighbor in the 3x3 Moore neighborhood
        neighbors = self.get_moore_neighbors(r, c)
        best_n_r, best_n_c = min(neighbors, key=lambda pos: fitness_grid[pos[0], pos[1]])
        best_neighbor = grid[best_n_r, best_n_c]

        current_cell = grid[r, c]
        current_fit = fitness_grid[r, c]

        # Information diffusion rule: move toward best local neighbor + stochastic exploration
        alpha = cell_rng.uniform(0.3, 0.8)
        candidate = current_cell + alpha * (best_neighbor - current_cell)

        # Apply cellular mutation to escape local optima
        mut_mask = cell_rng.random(self.dim) < self.mutation_rate
        candidate[mut_mask] += cell_rng.normal(0.0, step_scale, size=np.sum(mut_mask))
        candidate = np.clip(candidate, 0.0, 0.9999)

        cand_fit, _, _ = self.problem.evaluate(candidate)

        # Synchronous greedy state update
        if cand_fit < current_fit:
            return r, c, candidate, cand_fit
        return r, c, current_cell, current_fit

    def optimize(self):
        # Step 3: Initialize 2D Cellular Grid Population
        grid = self.rng.random((self.rows, self.cols, self.dim))
        fitness_grid = np.zeros((self.rows, self.cols))

        # Step 4: Initial Fitness Evaluation
        for r in range(self.rows):
            for c in range(self.cols):
                fitness_grid[r, c], _, _ = self.problem.evaluate(grid[r, c])

        best_idx = np.unravel_index(np.argmin(fitness_grid), fitness_grid.shape)
        global_best_cell = grid[best_idx].copy()
        global_best_fit = fitness_grid[best_idx]

        # Steps 5 & 6: Parallel Synchronous Cellular Evolution Loop
        with ThreadPoolExecutor() as executor:
            for t in range(self.max_iters):
                step_scale = 0.25 * (1.0 - t / self.max_iters)  # Decay perturbation over time
                tasks = [
                    (r, c, grid, fitness_grid, step_scale, int(self.rng.integers(0, 1_000_000)))
                    for r in range(self.rows)
                    for c in range(self.cols)
                ]

                next_grid = np.empty_like(grid)
                next_fitness = np.empty_like(fitness_grid)

                # Execute cell state transitions concurrently
                for r, c, updated_cell, updated_fit in executor.map(self._update_single_cell, tasks):
                    next_grid[r, c] = updated_cell
                    next_fitness[r, c] = updated_fit

                grid, fitness_grid = next_grid, next_fitness

                # Track best global cell across the grid
                curr_best_idx = np.unravel_index(np.argmin(fitness_grid), fitness_grid.shape)
                if fitness_grid[curr_best_idx] < global_best_fit:
                    global_best_fit = fitness_grid[curr_best_idx]
                    global_best_cell = grid[curr_best_idx].copy()

        # Step 7: Output Best Solution
        best_routes = self.problem.decode_cell(global_best_cell)
        _, final_time, final_fuel = self.problem.evaluate(global_best_cell)
        return {
            "routes": best_routes,
            "fitness": global_best_fit,
            "total_time_mins": final_time,
            "total_fuel_liters": final_fuel,
        }


# ==============================================================================
# EXECUTION & SAMPLE SCENARIO (8 Customer Orders, 2 Delivery Drivers)
# ==============================================================================
if __name__ == "__main__":
    depot_loc = [0.0, 0.0]  # Cloud kitchen / Restaurant location (km)
    customer_coords = [
        [2.1, 3.4], [5.2, 1.0], [-3.0, 4.2], [-4.5, -2.1],
        [1.5, -3.8], [6.0, 4.5], [-1.2, 2.0], [3.8, -1.9]
    ]
    order_weights_kg = [2.5, 4.0, 1.5, 3.2, 2.0, 5.0, 1.2, 3.5]
    traffic_congestion = [1.2, 1.5, 1.1, 1.3, 1.0, 1.6, 1.1, 1.4]

    problem = FoodDeliveryProblem(
        depot=depot_loc,
        orders=customer_coords,
        weights=order_weights_kg,
        traffic_factors=traffic_congestion,
        n_drivers=2,
    )

    pca = ParallelCellularOptimizer(problem, grid_shape=(8, 8), max_iters=120, seed=42)
    result = pca.optimize()

    print("=== Optimal Food Delivery Plan (Parallel Cellular Algorithm) ===")
    for driver_id, route in enumerate(result["routes"]):
        stops = " -> ".join([f"Order #{idx + 1}" for idx in route]) if len(route) else "No Orders"
        print(f"Driver {driver_id + 1}: Depot -> {stops} -> Depot")

    print("\n=== Performance Metrics ===")
    print(f"Total Delivery Time : {result['total_time_mins']:.2f} mins")
    print(f"Total Fuel Consumed : {result['total_fuel_liters']:.2f} liters")
    print(f"Combined Objective  : {result['fitness']:.4f}")
