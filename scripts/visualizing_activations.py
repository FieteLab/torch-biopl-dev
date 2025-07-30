import numpy as np
import torch
import pandas as pd
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import os
import json
import pickle
import matplotlib.pyplot as plt
from IPython.display import clear_output

from bioplnn.models import SpatiallyEmbeddedClassifier, SpatiallyEmbeddedAreaConfig, SpatiallyEmbeddedRNN
from bioplnn.datasets import Mazes
from bioplnn.utils import (
    initialize_dataloader,
)

maze_data_path = "/om2/user/jackking/torch-bioplnn-dev/data/mazes"
checkpoint_path = "/om2/user/jackking/torch-bioplnn-dev/train/checkpoints/"

# Torch setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

batch_size = 10

def prepare_rnn_weights(state_dict):
    """Remove 'rnn.' prefix from all keys in the state dict.
        Also remove any key starting with readout"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('rnn.'):
            new_key = key[4:]  # Remove 'rnn.' prefix
            new_state_dict[new_key] = value
        elif not key.startswith('readout'):
            new_state_dict[key] = value
    return new_state_dict
    

def combine_activations(activations):
    return torch.mean(torch.abs(activations), dim=1)

def visualize_activations(excitatory_activations, inhibitory_activations, maze):
    excitatory_activations = combine_activations(excitatory_activations)
    inhibitory_activations = combine_activations(inhibitory_activations)
    
    combined_activations = torch.sum(excitatory_activations - inhibitory_activations, dim = 0)
    return combined_activations

def plot_activations(train_loader):
    # wandb_name = "happy-microwave-331"
    # wandb_name = "elated-moon-329"
    wandb_name = "toasty-oath-332"

    try:
        full_config = pickle.load(open(checkpoint_path + f"{wandb_name}.pkl", "rb"))
        model_config = full_config["model_config"]
        num_steps = full_config["num_steps"]
    except:
        model_config = pickle.load(open(checkpoint_path + f"{wandb_name}.pkl", "rb"))
        num_steps = 20

    model = SpatiallyEmbeddedRNN(**model_config["rnn_kwargs"])
    classifier = SpatiallyEmbeddedClassifier(**model_config)
    inputs, labels = next(iter(train_loader))

    mazes = Mazes(maze_data_path)

    checkpoints = [10, 90]
    num_inputs = len(inputs)
    n, m = 2, 2  # inner activation grid size
    excitatory = False

    # Figure: width has 1 maze column + one activation column per checkpoint
    fig = plt.figure(figsize=(3 * (len(checkpoints)+1) * m, 10 * num_inputs))
    outer_gs = fig.add_gridspec(
        num_inputs,
        len(checkpoints) + 1,
        width_ratios=[1] + [2] * len(checkpoints),
        # height_ratios=[1] * num_inputs,
        wspace=0.2,
        # hspace=0.4
    )

    for i, (input_tensor, label) in enumerate(zip(inputs, labels)):
        # ——— Maze once per row ———
        ax_maze = fig.add_subplot(outer_gs[i, 0])
        maze_img = mazes.tensor_to_image(input_tensor)
        ax_maze.imshow(maze_img)
        ax_maze.set_title(f"Sample {i}", fontsize=10)
        ax_maze.axis('off')

        for j, epoch in enumerate(checkpoints):
            # load model & classifier
            state_dict = torch.load(f"{checkpoint_path}{wandb_name}_{epoch}.pth")
            model.load_state_dict(prepare_rnn_weights(state_dict))
            classifier.load_state_dict(state_dict)

            # run classifier to get correctness
            out = classifier(input_tensor.unsqueeze(0), num_steps=num_steps)
            correct = (out.argmax(dim=1).item() == label)

            # get activations
            _, neuron_states, _ = model(input_tensor.unsqueeze(0), num_steps=num_steps)
            if excitatory:
                act = model.query_neuron_states(neuron_states, 0, 0).detach().cpu()[0]
            else:
                act = model.query_neuron_states(neuron_states, 0, 1).detach().cpu()[0]
            images = combine_activations(act)
            total_steps = images.shape[0]
            
            # nest activation grid in column j+1
            sub_gs = outer_gs[i, j+1].subgridspec(n, m, hspace=0.1, wspace=0.1)
            step_idxs = np.linspace(0, total_steps-1, n*m, dtype=int)
            for k, step in enumerate(step_idxs):
                r, c = divmod(k, m)
                ax = fig.add_subplot(sub_gs[r, c])
                ax.imshow(images[step], cmap='viridis')
                ax.axis('off')
                title = f"Step {step}"
                if step == 0:
                    # only the top‐left cell gets the column header
                    status = "✓" if correct else "✗"
                    title += f" - E{epoch} {status}"
                ax.set_title(title, fontsize=8)

    plt.tight_layout()
    if excitatory:
        plt.savefig(f"/om2/user/jackking/torch-bioplnn-dev/scripts/activations/{wandb_name}_excitatory_activations.png")
    else:
        plt.savefig(f"/om2/user/jackking/torch-bioplnn-dev/scripts/activations/{wandb_name}_inhibitory_activations.png")
    plt.show()

def plot_conv_weights():
    wandb_name = "toasty-oath-332"
    epoch = 10

    state_dict = torch.load(f"{checkpoint_path}{wandb_name}_{epoch}.pth", map_location=torch.device('cpu'))

    for key in state_dict.keys():
        if "convs" in key and "weight" in key:
            weight = state_dict[key]
            total_kernels = weight.shape[0] * weight.shape[1]
            weights = weight.reshape(total_kernels, weight.shape[2], weight.shape[3])
            width = int(math.sqrt(total_kernels))
            height = total_kernels // width
            fig, axs = plt.subplots(height, width, figsize=(10, 10))
            for kernel in range(total_kernels):
                axs[kernel//width, kernel%width].imshow(weights[kernel], cmap='viridis')
                axs[kernel//width, kernel%width].axis('off')
            plt.title(key)
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    train_loader, test_loader = initialize_dataloader(
    seed=42, root="./data/mazes/", batch_size=batch_size, dataset="mazes"
    )   
    # plot_activations(train_loader)
    plot_conv_weights()