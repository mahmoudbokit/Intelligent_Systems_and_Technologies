import csv
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm  # Импортируем tqdm для отслеживания прогресса

# Создание пустого графа
G = nx.Graph()

with open('social_graph.csv', 'r') as file:
    reader = csv.reader(file, quotechar='"')
    next(reader)

    total_lines = sum(1 for _ in file)
    file.seek(0)
    next(file)

    for row in tqdm(reader, total=total_lines, desc="Обработка строк"):
        user = int(row[0])
        friends = row[1].split(',')  # Разделение друзей по запятой

        # Условие для пропуска пользователей с более чем 100 друзьями
        if len(friends) > 100:
            continue

        G.add_edges_from((user, int(friend)) for friend in friends)  # Добавление рёбер для каждого друга

# Расчет центральности по степени
degree_centrality = nx.degree_centrality(G)

# Определение цвета узлов
node_colors = [
    'green' if degree_centrality[node] > 0.01 else 'lightblue'  # Задаем пороговое значение для центральных точек
    for node in G.nodes
]

# Визуализация всего графа
plt.figure(figsize=(50, 50))
pos = nx.spring_layout(G, iterations=50)  # Расчет положения узлов для всего графа

# Визуализация графа
nx.draw(
    G, pos,
    with_labels=False,
    node_size=50,
    node_color=node_colors,  # Используем список цветов для узлов
    font_size=8,
    font_color='black',
    edge_color='gray'
)

plt.title("Визуализация графа с выделением центральных точек")
plt.show()
