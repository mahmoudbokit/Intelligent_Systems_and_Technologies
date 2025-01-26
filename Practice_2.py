import requests
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Токен и информация о пользователе
ACCESS_TOKEN = "vk1.a.kdSyfkHgijrKmZG5eI2KswYnMzonRVOM-bYeCCfZkQTYDgEcwGTXG9rkvUdrv7mBzJZE0INr42nvGegJ24XsWiAi6a7BiHLg1jmSU51w90kbn6j5-RfbJd4Y4euEfNcEJOSceCD34p4TUoC7-8J50Xyijc0upBew_fUZmE-CO2ugwSROXQ-UCqVu8OzYOihoPOfIXl97wCimheLs-cP0oA"
USER_ID = "619628667"
VK_API_URL = "https://api.vk.com/method"
VERSION = "5.131"


# Функция поиска друзей пользователя
def get_friends(user_id, token):
    url = f"https://api.vk.com/method/friends.get"
    params = {
        "user_id": user_id,
        "access_token": token,
        "v": "5.131"
    }
    response = requests.get(url, params=params)
    data = response.json()

    if "error" in data:
        if data["error"]["error_code"] == 18:
            print(f"Utilisateur {user_id} supprimé ou banni. Ignoré.")
            return []  # Retourne une liste vide pour cet utilisateur
        else:
            print(f"Erreur pour l'utilisateur {user_id} : {data['error']['error_msg']}")
            return []

    return data.get("response", {}).get("items", [])


# Функция построения сети друзей и друзей друзей
def build_network(user_id, token):
    G = nx.Graph()

    # Ajout des amis directs
    friends = get_friends(user_id, token)  # Passez le token ici
    G.add_node(user_id, label="Utilisateur")
    for friend in friends:
        G.add_node(friend, label="Ami")
        G.add_edge(user_id, friend)

    # Ajout des amis des amis
    for friend in friends:
        friends_of_friend = get_friends(friend, token)  # Passez le token ici aussi
        for fof in friends_of_friend:
            if fof != user_id:  # Éviter une boucle
                G.add_node(fof, label="Ami d'ami")
                G.add_edge(friend, fof)

    return G


# Функция вычисления центральностей : опосредованная, близость, собственный вектор
def calculate_centralities(G):
    """Вычисляет центральности : посредничество, близость, собственный вектор"""
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    eigenvector = nx.eigenvector_centrality(G, max_iter=1000)

    return betweenness, closeness, eigenvector

# Функция построения сравнительного графика центральностей
def plot_centralities(betweenness, closeness, eigenvector):
    """Нарисуйте сравнительный график центральностей"""
    labels = list(betweenness.keys())
    x = np.arange(len(labels))

    plt.figure(figsize=(12, 6))

    plt.bar(x - 0.2, betweenness.values(), width=0.2, label='Médiation')
    plt.bar(x, closeness.values(), width=0.2, label='Proximité')
    plt.bar(x + 0.2, eigenvector.values(), width=0.2, label='Vecteur propre')

    plt.xlabel("Utilisateurs")
    plt.ylabel("Valeur de centralité")
    plt.title("Comparaison des centralités")
    plt.xticks(x, labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Функция построения сети
def main():
    # Token d'accès
    ACCESS_TOKEN = "vk1.a.34bvcrFA78Ncn78SzAehqAVm4aHywe8VTWQto49i6Xq0yL7EjG9voQFZri5n11lfLEfsYWXTtn0LoBwj8NPWatj4kvexXMz4Xu8WfczXfcWhKChv1StxFc2_U0HTdoeiTdgbmSTxCstAKLhwaZAaL1Qqb-mBYFoOOT07R--4T6S2hdsxnzaYn6EymbB5vnAVBK4zB_Zkh9NkjkIOW9pCaA"

    # Construire le réseau
    print("Construction du réseau...")
    G = build_network(USER_ID, ACCESS_TOKEN)

    # Calculer les centralités
    print("Calcul des centralités...")
    centralities = calculate_centralities(G)

    # Visualiser les résultats
    print("Affichage des résultats...")
    plot_centralities(centralities)

if __name__ == "__main__":
    main()

