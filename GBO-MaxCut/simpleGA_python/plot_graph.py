import matplotlib.pyplot as plt
import networkx as nx


def parse_file(file_name):
    graph = nx.Graph()

    with open(file_name, 'r') as file:
        first_line = file.readline().strip().split()
        num_vertices, num_edges = map(int, first_line)

        for line in file:
            vertex1, vertex2, weight = map(int, line.strip().split())
            graph.add_edge(vertex1, vertex2, weight=weight)

    return graph


def plot_graph(graph):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.show()


def main():
    file_name = 'maxcut-instances/setA/n0000006i00.txt'  # Change this to your file path
    graph = parse_file(file_name)
    plot_graph(graph)


if __name__ == '__main__':
    main()