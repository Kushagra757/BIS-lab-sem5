
import numpy as np
import matplotlib.pyplot as plt
import math

# 1. CLASSROOM CONDITIONS

outside_temp = 32
daylight = 200
occupants = 30

MIN_TEMP = 22
MAX_TEMP = 26
REQUIRED_LUX = 400

# Variables: [AC Temperature, Light Percentage]

LOWER = np.array([18, 0])
UPPER = np.array([30, 100])

# 2. FITNESS FUNCTION

def fitness(nest):

    temp, light = nest

    # Calculate electricity consumption
    ac_power = 1 + 0.12 * abs(outside_temp - temp)
    light_power = 0.01 * light

    total_energy = ac_power + light_power

    # Calculate classroom comfort
    brightness = daylight + light * 5

    penalty = 0

    if temp < MIN_TEMP or temp > MAX_TEMP:
        penalty += 1000

    if brightness < REQUIRED_LUX:
        penalty += 1000

    return total_energy + penalty


# 3. LEVY FLIGHT

def levy_flight():

    beta = 1.5

    sigma = (
        math.gamma(1 + beta) *
        math.sin(np.pi * beta / 2) /
        (
            math.gamma((1 + beta) / 2) *
            beta * 2 ** ((beta - 1) / 2)
        )
    ) ** (1 / beta)

    u = np.random.normal(0, sigma, 2)
    v = np.random.normal(0, 1, 2)

    return u / (np.abs(v) ** (1 / beta) + 1e-10)


# 4. CUCKOO SEARCH

def cuckoo_search():

    N = 25
    ITERATIONS = 100
    PA = 0.25
    ALPHA = 0.1

    nests = np.random.uniform(
        LOWER, UPPER, (N, 2)
    )

    best = nests[0].copy()
    best_cost = fitness(best)

    history = []

    for iteration in range(ITERATIONS):

        for i in range(N):

            new_nest = (
                nests[i] +
                ALPHA * levy_flight() *
                (nests[i] - best)
            )

            new_nest = np.clip(new_nest, LOWER, UPPER)

            j = np.random.randint(N)

            if fitness(new_nest) < fitness(nests[j]):
                nests[j] = new_nest

        # Abandon worst nests
        count = int(PA * N)

        costs = np.array([fitness(n) for n in nests])

        worst = np.argsort(costs)[-count:]

        nests[worst] = np.random.uniform(
            LOWER, UPPER, (count, 2)
        )

        # Update best solution
        costs = np.array([fitness(n) for n in nests])

        index = np.argmin(costs)

        if costs[index] < best_cost:
            best = nests[index].copy()
            best_cost = costs[index]

        history.append(best_cost)

    return best, history


# 5. MAIN FUNCTION

best, history = cuckoo_search()

temp, light = best

brightness = daylight + light * 5

energy = (
    1 + 0.12 * abs(outside_temp - temp)
    + 0.01 * light
)

print("\nOPTIMIZED CLASSROOM SETTINGS")
print("AC Temperature:", round(temp, 2), "C")
print("Light Level:", round(light, 2), "%")
print("Brightness:", round(brightness, 2), "lux")
print("Total Energy:", round(energy, 2), "units/hour")

if 22 <= temp <= 26 and brightness >= 400:
    print("Status: Comfortable Classroom")
else:
    print("Status: Comfort Requirements Not Met")


# 6. CONVERGENCE GRAPH

plt.plot(history)
plt.xlabel("Iterations")
plt.ylabel("Best Fitness")
plt.title("Cuckoo Search Optimization")
plt.grid()
plt.show()
