import numpy as np
import matplotlib.pyplot as plt
import random

# Параметры модели
n = 15 # Размер квадрата n x n
blue_percentage = 0.45
red_percentage = 0.45
empty_percentage = 0.10
threshold = 2  # Порог счастья
steps = 100  # Количество шагов моделирования

# Инициализация поля
def initialize_grid(n, blue_percentage, red_percentage, empty_percentage):
    grid = np.zeros((n, n), dtype=int)
    total_cells = n * n
    blue_cells = int(total_cells * blue_percentage)
    red_cells = int(total_cells * red_percentage)
    empty_cells = total_cells - blue_cells - red_cells
    
    cells = [1] * blue_cells + [2] * red_cells + [0] * empty_cells
    random.shuffle(cells)
    
    grid = np.array(cells).reshape(n, n)
    return grid

# Проверка счастья клетки
def is_happy(grid, x, y, threshold):
    color = grid[x, y]
    if color == 0:  # Пустая клетка всегда счастлива
        return True
    
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if 0 <= x + i < n and 0 <= y + j < n:
                neighbors.append(grid[x + i, y + j])
    
    same_color_neighbors = sum(1 for neighbor in neighbors if neighbor == color)
    return same_color_neighbors >= threshold

# Моделирование
def simulate(grid, steps, threshold):
    for step in range(steps):
        unhappy_cells = [(x, y) for x in range(n) for y in range(n) if not is_happy(grid, x, y, threshold)]
        if not unhappy_cells:
            print(f"Все клетки счастливы после {step} шагов.")
            break
        
        x, y = random.choice(unhappy_cells)
        empty_cells = [(i, j) for i in range(n) for j in range(n) if grid[i, j] == 0]
        if empty_cells:
            new_x, new_y = random.choice(empty_cells)
            grid[new_x, new_y] = grid[x, y]
            grid[x, y] = 0
        
        if step % 10 == 0:
            plot_grid(grid, step)

# Визуализация
def plot_grid(grid, step):
    plt.imshow(grid, cmap='Blues')
    plt.title(f"Шаг {step}")
    plt.show()

# Инициализация и запуск моделирования
grid = initialize_grid(n, blue_percentage, red_percentage, empty_percentage)
simulate(grid, steps, threshold)
plot_grid(grid, steps)