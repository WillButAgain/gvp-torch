import torch

from tqdm import tqdm

from data import NoisyDataFolder
from gvp_model import GVP_GNN

def train(dataloader, model, loss_fn, optimizer):
    subset_loss = 0
    log_iter = 100
    for batch, (x, target) in enumerate(dataloader):
        # Compute prediction error
        node_feats, edge_feats, edge_idx, coords = x
        pred = model(*x)
        # :3 for first 3 items of prediction, which we choose arbitrarily. task is to predict original coordinates given noisy ones.
        # -1 because the dataloader returns the coordinate tensor as the last element in the tuple
        loss = loss_fn(pred[0][:, :3], target[-1][:, 2, :]) # we will say that the residue coordinate is the coordiante of the alpha carbon 

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        subset_loss += loss.item()
        optimizer.step()
        
        if (1+batch) % log_iter == 0:
            yield subset_loss/log_iter

            
device = 'cuda:0'
epochs = 5

loader = NoisyDataFolder("/data/home/will/drugs-vae/data/big_files/PDB_dump", device, shuffle=True)

model = GVP_GNN(
    x_in_vector_channels=3,
    x_out_vector_channels=4,
    x_in_scalar_channels=22,
    x_out_scalar_channels=48,
    edge_in_vector_channels=1,
    edge_out_vector_channels=2,
    edge_in_scalar_channels=20,
    edge_out_scalar_channels=16,
    hidden_dim=16,
    n_layers=3
).to(device)

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(epochs):
    iterator = tqdm(train(loader, model, loss, optimizer))
    for loss in iterator:
        iterator.set_description(f'Loss: {loss}')