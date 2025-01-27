import streamlit as st
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(FeedForwardNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer to first hidden layer
        prev_size = input_size
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Last hidden layer to output layer
        self.layers.append(nn.Linear(prev_size, output_size))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x

def visualize_network(input_size, hidden_layers, output_size):
    G = nx.DiGraph()
    pos = {}
    layer_sizes = [input_size] + hidden_layers + [output_size]
    
    # Create nodes for each layer
    node_idx = 0
    layer_nodes = []
    for layer_idx, layer_size in enumerate(layer_sizes):
        nodes = []
        for i in range(layer_size):
            node_name = f"L{layer_idx}_{i}"
            G.add_node(node_name)
            pos[node_name] = (layer_idx, i - layer_size/2)
            nodes.append(node_name)
        layer_nodes.append(nodes)
    
    # Add edges between layers
    for i in range(len(layer_sizes)-1):
        for source in layer_nodes[i]:
            for target in layer_nodes[i+1]:
                G.add_edge(source, target)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=False, node_color='lightblue', 
            node_size=500, arrowsize=10, arrows=True)
    plt.title("Neural Network Architecture")
    
    return plt

def main():
    st.title("Feedforward Neural Network Visualization")
    
    # Sidebar controls
    st.sidebar.header("Network Configuration")
    input_size = st.sidebar.slider("Input Layer Size", 1, 10, 3)
    num_hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, 2)
    
    hidden_layers = []
    for i in range(num_hidden_layers):
        hidden_size = st.sidebar.slider(f"Hidden Layer {i+1} Size", 1, 10, 4)
        hidden_layers.append(hidden_size)
    
    output_size = st.sidebar.slider("Output Layer Size", 1, 10, 2)
    
    # Create and display the network
    model = FeedForwardNN(input_size, hidden_layers, output_size)
    
    # Display network information
    st.write("### Network Architecture")
    st.write(f"Input Layer: {input_size} neurons")
    for i, hidden_size in enumerate(hidden_layers):
        st.write(f"Hidden Layer {i+1}: {hidden_size} neurons")
    st.write(f"Output Layer: {output_size} neurons")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    st.write(f"Total Parameters: {total_params}")
    
    # Visualize the network
    plt = visualize_network(input_size, hidden_layers, output_size)
    st.pyplot(plt)

if __name__ == "__main__":
    main()