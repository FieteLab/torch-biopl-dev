# Computing gradients and backpropagating through connectome-initialized models

While the previous tutorial showed you how to initialize a neural network with connectome-determined weights, it didn't provide you with information on how to "tune" model parameters in a data-driven manner. Turns out, computing (and backpropagating) gradients in networks with both ***sparse and dense*** tensors is non-trivial. In this tutorial, we spin up a small example on how you can accomplish this in `torch-biopl`.

Goals:

- Continue from our previous example.
- Implement a wrapper on top of the `ConnectomeODERNN` to support a readout layer.
- Setup a simple training loop.

Note: For demonstration purposes, we'll use flattened MNIST images as inputs into the connectome. This is however simplistic and we do allow for arbitrarily complex input mappings. To learn how to do that please refer to the API.


```python
import os

import torch
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm

from bioplnn.models import ConnectomeODEClassifier
```


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
print("Using device: {}".format(device))
```

    Using device: cuda



```python
# Download the connectome and read it in as a torch tensor. We have pre-processed this as a sparse tensor for the purposes of this example.
save_dir = "connectivity/turaga"
os.makedirs(save_dir, exist_ok=True)
!gdown "https://drive.google.com/uc?id=18448HYpYrm60boziHG73bxN4CK5jG-1g" -O "{save_dir}/turaga-dros-visual-connectome.pt"
connectome = torch.load(
    os.path.join(save_dir, "turaga-dros-visual-connectome.pt"),
    weights_only=True,
)

from bioplnn.utils import create_sparse_projection

# since we are feeding in MNIST images
input_size = 28 * 28
num_neurons = connectome.shape[0]

input_projection_matrix = create_sparse_projection(
    size=input_size,
    num_neurons=num_neurons,
    indices=torch.randint(high=num_neurons, size=(input_size,)),
    mode="ih",
)

# for now, lets just read outputs from all neurons
output_projection_matrix = None
```

    Downloading...
    From (original): https://drive.google.com/uc?id=18448HYpYrm60boziHG73bxN4CK5jG-1g
    From (redirected): https://drive.google.com/uc?id=18448HYpYrm60boziHG73bxN4CK5jG-1g&confirm=t&uuid=d8cf0a1e-b8ee-4271-b52e-09e3885e30a2
    To: /net/vast-storage/scratch/vast/mcdermott/lakshmin/torch-bioplnn-dev/examples/connectivity/turaga/turaga-dros-visual-connectome.pt
    100%|█████████████████████████████████████████| 111M/111M [00:01<00:00, 111MB/s]


## Setting up the classifier wrapper

Here, we have written a simple utility that adds an output projecting layer from the connectome to the desired logit space. Again, this is simply an example. Please feel free to add sophistication to this as you please.


```python
model = ConnectomeODEClassifier(
    rnn_kwargs={
        "input_size": input_size,
        "num_neurons": num_neurons,
        "connectome": torch.abs(connectome),
        "input_projection": input_projection_matrix,
        "output_projection": output_projection_matrix,
        "neuron_nonlinearity": "Sigmoid",
        "batch_first": False,
        "compile_solver_kwargs": {
            "mode": "max-autotune",
            "dynamic": False,
            "fullgraph": True,
        },
    },
    num_classes=10,
    fc_dim=256,
    dropout=0.5,
).to(device)
print(model)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define the loss function
criterion = nn.CrossEntropyLoss()
```

    ConnectomeODEClassifier(
      (rnn): ConnectomeODERNN(
        (nonlinearity): Sigmoid()
        (hh): SparseLinear()
        (ih): SparseLinear()
        (ho): Identity()
        (solver): OptimizedModule(
          (_orig_mod): AutoDiffAdjoint(step_method=Dopri5(
            (term): ODETerm()
          ), step_size_controller=IntegralController(
            (term): ODETerm()
          ), max_steps=None, backprop_through_step_size_control=True)
        )
        (neuron_nonlinearity): Sigmoid()
      )
      (out_layer): Sequential(
        (0): Flatten(start_dim=1, end_dim=-1)
        (1): Linear(in_features=47521, out_features=256, bias=True)
        (2): ReLU()
        (3): Dropout(p=0.5, inplace=False)
        (4): Linear(in_features=256, out_features=10, bias=True)
      )
    )



```python
# Dataloader setup
transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
train_data = MNIST(root="data", train=True, transform=transform, download=True)
train_loader = DataLoader(
    train_data, batch_size=8, num_workers=0, shuffle=True
)
```


```python
n_epochs = 1
# print training statistics every five batches
log_frequency = 50
model = model.train()
```

## Note: Things to look out for

- Depending on the GPU you have access to and the number of CPUs you will have to adjust `batch_size` and `num_workers` in the DataLoader, as well as `num_steps` in the model forward pass. For reference, on a A100 GPU and a single CPU `batch_size = 256`, `num_workers = 0`, and `num_steps=5` is a reasonable estimate.
- In this example, we have used the torch compiler with the `max-autotune` flag. This means the first few steps in the first epoch WILL BE EXTREMELY SLOW. But, this will dramatically improve as the training goes on. Fret not, early on!


```python
running_loss, running_correct, running_total = 0, 0, 0

for epoch in range(n_epochs):
    for i, (x, labels) in enumerate(train_loader):
        x = x.to(device)
        labels = labels.to(device)
        torch._inductor.cudagraph_mark_step_begin()
        logits = model(x, num_evals=2)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate running accuracy and loss
        _, predicted = torch.max(logits, 1)
        running_total += labels.size(0)
        running_correct += (predicted == labels).sum().item()
        running_loss += loss.item()

        running_acc = running_correct / running_total
        if (i + 1) % log_frequency == 0:
            print(
                f"Training | Epoch: {epoch} | "
                + f"Loss: {running_loss:.4f} | "
                + f"Acc: {running_acc:.2%}"
            )
            running_loss, running_correct, running_total = 0, 0, 0
```

    Training | Epoch: 0 | Loss: 533.8668 | Acc: 11.25%
    Training | Epoch: 0 | Loss: 115.2019 | Acc: 9.50%
    Training | Epoch: 0 | Loss: 115.0685 | Acc: 8.00%
    Training | Epoch: 0 | Loss: 115.0090 | Acc: 11.75%
    Training | Epoch: 0 | Loss: 115.2327 | Acc: 8.75%
    Training | Epoch: 0 | Loss: 115.1742 | Acc: 13.50%
    Training | Epoch: 0 | Loss: 115.1926 | Acc: 10.00%
    Training | Epoch: 0 | Loss: 115.1818 | Acc: 10.25%
    Training | Epoch: 0 | Loss: 115.1044 | Acc: 11.25%
    Training | Epoch: 0 | Loss: 115.0677 | Acc: 10.75%
    Training | Epoch: 0 | Loss: 115.1434 | Acc: 12.00%
    Training | Epoch: 0 | Loss: 115.0964 | Acc: 12.50%
    Training | Epoch: 0 | Loss: 115.0139 | Acc: 12.75%
    Training | Epoch: 0 | Loss: 115.0094 | Acc: 13.50%
    Training | Epoch: 0 | Loss: 115.0079 | Acc: 11.75%
    Training | Epoch: 0 | Loss: 114.9612 | Acc: 12.50%
    Training | Epoch: 0 | Loss: 115.1430 | Acc: 10.50%
    Training | Epoch: 0 | Loss: 115.0567 | Acc: 12.00%
    Training | Epoch: 0 | Loss: 115.0514 | Acc: 12.75%
    Training | Epoch: 0 | Loss: 114.9140 | Acc: 14.00%
    Training | Epoch: 0 | Loss: 115.0926 | Acc: 11.00%
    Training | Epoch: 0 | Loss: 115.0142 | Acc: 10.75%
    Training | Epoch: 0 | Loss: 115.2382 | Acc: 10.50%
    Training | Epoch: 0 | Loss: 115.1861 | Acc: 9.75%
    Training | Epoch: 0 | Loss: 115.0188 | Acc: 11.25%
    Training | Epoch: 0 | Loss: 115.1348 | Acc: 10.75%
    Training | Epoch: 0 | Loss: 114.8985 | Acc: 13.00%
    Training | Epoch: 0 | Loss: 115.0376 | Acc: 11.50%
    Training | Epoch: 0 | Loss: 114.9054 | Acc: 13.50%
    Training | Epoch: 0 | Loss: 115.0908 | Acc: 11.00%
    Training | Epoch: 0 | Loss: 115.0573 | Acc: 12.75%
    Training | Epoch: 0 | Loss: 115.0027 | Acc: 12.00%
    Training | Epoch: 0 | Loss: 115.0626 | Acc: 9.75%
    Training | Epoch: 0 | Loss: 115.1542 | Acc: 10.75%
    Training | Epoch: 0 | Loss: 115.0336 | Acc: 13.25%
    Training | Epoch: 0 | Loss: 115.2255 | Acc: 8.25%
    Training | Epoch: 0 | Loss: 114.9530 | Acc: 12.25%
    Training | Epoch: 0 | Loss: 115.2533 | Acc: 8.50%
    Training | Epoch: 0 | Loss: 115.2574 | Acc: 9.50%
    Training | Epoch: 0 | Loss: 114.9386 | Acc: 13.25%
    Training | Epoch: 0 | Loss: 115.0560 | Acc: 10.00%
    Training | Epoch: 0 | Loss: 115.0315 | Acc: 10.50%
    Training | Epoch: 0 | Loss: 115.2135 | Acc: 10.00%
    Training | Epoch: 0 | Loss: 115.0068 | Acc: 14.25%
    Training | Epoch: 0 | Loss: 114.9713 | Acc: 13.25%
    Training | Epoch: 0 | Loss: 115.0257 | Acc: 11.50%
    Training | Epoch: 0 | Loss: 115.0284 | Acc: 12.00%
    Training | Epoch: 0 | Loss: 114.9427 | Acc: 11.25%
    Training | Epoch: 0 | Loss: 115.1230 | Acc: 9.75%
    Training | Epoch: 0 | Loss: 115.1148 | Acc: 11.75%
    Training | Epoch: 0 | Loss: 115.1142 | Acc: 10.50%
    Training | Epoch: 0 | Loss: 114.9336 | Acc: 12.25%
    Training | Epoch: 0 | Loss: 115.0960 | Acc: 11.25%
    Training | Epoch: 0 | Loss: 115.0744 | Acc: 11.00%
    Training | Epoch: 0 | Loss: 115.0859 | Acc: 9.75%
    Training | Epoch: 0 | Loss: 114.9641 | Acc: 11.50%
    Training | Epoch: 0 | Loss: 115.1912 | Acc: 10.50%
    Training | Epoch: 0 | Loss: 115.0681 | Acc: 10.25%
    Training | Epoch: 0 | Loss: 115.2021 | Acc: 11.25%
    Training | Epoch: 0 | Loss: 115.1686 | Acc: 10.00%
    Training | Epoch: 0 | Loss: 115.1215 | Acc: 9.75%
    Training | Epoch: 0 | Loss: 115.0815 | Acc: 10.25%
    Training | Epoch: 0 | Loss: 114.8752 | Acc: 14.75%
    Training | Epoch: 0 | Loss: 115.2911 | Acc: 9.75%
    Training | Epoch: 0 | Loss: 115.2639 | Acc: 11.25%
    Training | Epoch: 0 | Loss: 115.1642 | Acc: 10.75%
    Training | Epoch: 0 | Loss: 115.1821 | Acc: 10.50%
    Training | Epoch: 0 | Loss: 115.3850 | Acc: 7.75%
    Training | Epoch: 0 | Loss: 115.2006 | Acc: 8.75%
    Training | Epoch: 0 | Loss: 114.9202 | Acc: 14.00%
    Training | Epoch: 0 | Loss: 115.0008 | Acc: 11.50%
    Training | Epoch: 0 | Loss: 115.1776 | Acc: 10.25%
    Training | Epoch: 0 | Loss: 115.1527 | Acc: 12.75%
    Training | Epoch: 0 | Loss: 115.2655 | Acc: 9.00%
    Training | Epoch: 0 | Loss: 115.0813 | Acc: 11.75%
    Training | Epoch: 0 | Loss: 115.0945 | Acc: 10.00%
    Training | Epoch: 0 | Loss: 115.1136 | Acc: 8.50%
    Training | Epoch: 0 | Loss: 115.2337 | Acc: 10.25%
    Training | Epoch: 0 | Loss: 115.1391 | Acc: 8.25%
    Training | Epoch: 0 | Loss: 115.0075 | Acc: 11.75%
    Training | Epoch: 0 | Loss: 115.0086 | Acc: 12.00%
    Training | Epoch: 0 | Loss: 115.0590 | Acc: 13.50%
    Training | Epoch: 0 | Loss: 115.1641 | Acc: 9.50%
    Training | Epoch: 0 | Loss: 115.0044 | Acc: 13.75%
    Training | Epoch: 0 | Loss: 115.1192 | Acc: 11.50%
    Training | Epoch: 0 | Loss: 115.0291 | Acc: 10.25%
    Training | Epoch: 0 | Loss: 115.1939 | Acc: 11.25%
    Training | Epoch: 0 | Loss: 115.2703 | Acc: 8.50%
    Training | Epoch: 0 | Loss: 114.8635 | Acc: 13.25%
    Training | Epoch: 0 | Loss: 115.0566 | Acc: 11.25%
    Training | Epoch: 0 | Loss: 114.9189 | Acc: 11.50%
    Training | Epoch: 0 | Loss: 115.0341 | Acc: 11.25%
    Training | Epoch: 0 | Loss: 115.0187 | Acc: 12.00%
    Training | Epoch: 0 | Loss: 115.1050 | Acc: 10.50%
    Training | Epoch: 0 | Loss: 115.0171 | Acc: 9.75%
    Training | Epoch: 0 | Loss: 114.8428 | Acc: 15.00%
    Training | Epoch: 0 | Loss: 115.1016 | Acc: 12.50%
    Training | Epoch: 0 | Loss: 115.3068 | Acc: 9.75%
    Training | Epoch: 0 | Loss: 114.9911 | Acc: 12.50%
    Training | Epoch: 0 | Loss: 115.2010 | Acc: 11.00%
    Training | Epoch: 0 | Loss: 115.0805 | Acc: 11.25%
    Training | Epoch: 0 | Loss: 115.0775 | Acc: 12.75%
    Training | Epoch: 0 | Loss: 115.2626 | Acc: 7.50%
    Training | Epoch: 0 | Loss: 114.9407 | Acc: 11.75%
    Training | Epoch: 0 | Loss: 115.1194 | Acc: 9.50%
    Training | Epoch: 0 | Loss: 115.2441 | Acc: 9.75%
    Training | Epoch: 0 | Loss: 114.8342 | Acc: 14.25%
    Training | Epoch: 0 | Loss: 114.7440 | Acc: 13.00%
    Training | Epoch: 0 | Loss: 114.9432 | Acc: 14.25%
    Training | Epoch: 0 | Loss: 115.1605 | Acc: 10.00%
    Training | Epoch: 0 | Loss: 114.8367 | Acc: 13.75%
    Training | Epoch: 0 | Loss: 115.0761 | Acc: 12.75%
    Training | Epoch: 0 | Loss: 115.1689 | Acc: 11.50%
    Training | Epoch: 0 | Loss: 115.3208 | Acc: 10.25%
    Training | Epoch: 0 | Loss: 115.2924 | Acc: 9.00%
    Training | Epoch: 0 | Loss: 115.0237 | Acc: 10.50%
    Training | Epoch: 0 | Loss: 115.0111 | Acc: 10.75%
    Training | Epoch: 0 | Loss: 115.2476 | Acc: 8.50%
    Training | Epoch: 0 | Loss: 115.0139 | Acc: 12.50%
    Training | Epoch: 0 | Loss: 115.1096 | Acc: 11.50%
    Training | Epoch: 0 | Loss: 115.2500 | Acc: 10.25%
    Training | Epoch: 0 | Loss: 115.1965 | Acc: 10.75%
    Training | Epoch: 0 | Loss: 115.2571 | Acc: 9.75%
    Training | Epoch: 0 | Loss: 115.0872 | Acc: 12.00%
    Training | Epoch: 0 | Loss: 114.8736 | Acc: 12.75%
    Training | Epoch: 0 | Loss: 115.3206 | Acc: 9.25%
    Training | Epoch: 0 | Loss: 115.2455 | Acc: 8.50%
    Training | Epoch: 0 | Loss: 114.9038 | Acc: 14.00%
    Training | Epoch: 0 | Loss: 115.1354 | Acc: 10.25%
    Training | Epoch: 0 | Loss: 114.9552 | Acc: 12.00%
    Training | Epoch: 0 | Loss: 115.0186 | Acc: 10.00%
    Training | Epoch: 0 | Loss: 114.9936 | Acc: 12.75%
    Training | Epoch: 0 | Loss: 115.1694 | Acc: 10.25%
    Training | Epoch: 0 | Loss: 115.1393 | Acc: 10.75%
    Training | Epoch: 0 | Loss: 115.0091 | Acc: 11.50%
    Training | Epoch: 0 | Loss: 115.0358 | Acc: 9.50%
    Training | Epoch: 0 | Loss: 114.9743 | Acc: 12.75%
    Training | Epoch: 0 | Loss: 115.0091 | Acc: 11.50%
    Training | Epoch: 0 | Loss: 114.8544 | Acc: 13.75%
    Training | Epoch: 0 | Loss: 115.0109 | Acc: 13.50%
    Training | Epoch: 0 | Loss: 115.1806 | Acc: 11.00%
    Training | Epoch: 0 | Loss: 114.8155 | Acc: 12.75%
    Training | Epoch: 0 | Loss: 114.9861 | Acc: 9.50%
    Training | Epoch: 0 | Loss: 115.4122 | Acc: 9.75%
    Training | Epoch: 0 | Loss: 114.9863 | Acc: 10.25%
    Training | Epoch: 0 | Loss: 114.9625 | Acc: 13.25%
    Training | Epoch: 0 | Loss: 115.2525 | Acc: 11.00%
    Training | Epoch: 0 | Loss: 114.9907 | Acc: 11.50%
    Training | Epoch: 0 | Loss: 115.1447 | Acc: 9.50%
    Training | Epoch: 0 | Loss: 115.2017 | Acc: 10.50%
