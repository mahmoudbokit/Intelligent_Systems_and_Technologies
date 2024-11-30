import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Присвоение имен членов группы узлам
members = ["Nikita", "Maxim", "Airat", "Minh", "Alexey", "Tokhir",
           "Kirill", "Danik", "Mansur"]

# Создание графика с видимыми соединениями на схеме
G_named = nx.Graph()
G_named.add_edges_from([
    (members[0], members[1]),  # Nikita - Maxim
    (members[0], members[2]),  # Nikita1 - Airat
    (members[2], members[3]),  # Airat - Minh
    (members[2], members[4]),  # Airat - Alexey
    (members[1], members[4]),  # Maxim - Alexey
    (members[4], members[5]),  # Alexey - Tokhir
    (members[5], members[6]),  # Tokhir - Kirill
    (members[5], members[7]),  # Tokhir - Danik
    (members[5], members[8])   # Tokhir - Mansur
])

# Расчет центральных положений
betweenness_named = nx.betweenness_centrality(G_named)
closeness_named = nx.closeness_centrality(G_named)
eigenvector_named = nx.eigenvector_centrality(G_named)

# Расположение узлов для аналогичной визуализации
pos_named = nx.spring_layout(G_named, seed=42)

# Визуализация аннотированного графика с центральностью близости
fig, ax = plt.subplots(figsize=(12, 8))

# Нарисовать график
nx.draw_networkx_nodes(G_named, pos_named, node_color='lightblue', node_size=800, ax=ax)
nx.draw_networkx_edges(G_named, pos_named, width=1.5, ax=ax)
nx.draw_networkx_labels(G_named, pos_named, font_size=10, ax=ax)

# Комментировать с помощью Центра близости
for node, (x, y) in pos_named.items():
    ax.text(x, y + 0.1, f"{closeness_named[node]:.2f}", fontsize=10, color="darkred", ha="center")

ax.set_title("График с центральностями (аннотированная близость)", fontsize=16)
plt.axis('off')
plt.show()

# Подготовить сравнительный график центральностей
betweenness_values_named = [betweenness_named[node] for node in members]
closeness_values_named = [closeness_named[node] for node in members]
eigenvector_values_named = [eigenvector_named[node] for node in members]

x = np.arange(len(members))  # Positions des barres
width = 0.25                 # Largeur des barres

fig, ax = plt.subplots(figsize=(12, 8))

# Столбцы для каждого типа центральности
ax.bar(x - width, betweenness_values_named, width, label='Посредничество', color='skyblue')
ax.bar(x, closeness_values_named, width, label='Близость', color='lightgreen')
ax.bar(x + width, eigenvector_values_named, width, label='Собственный вектор', color='salmon')

# Добавление меток и заголовка
ax.set_xlabel('Члены группы')
ax.set_ylabel('Значение центральности')
ax.set_title('Сравнение центральных функций для каждого участника')
ax.set_xticks(x)
ax.set_xticklabels(members, rotation=45)
ax.legend()

plt.tight_layout()
plt.show()