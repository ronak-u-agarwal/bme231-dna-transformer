
#%%
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from import_data import padded
from transformer_more import CodonTransformer
#%% Hyperparameters
## Embedding dimension, key/query dimension, number of self-attention
## heads in a single self-attention layer, etc.
#
# These are defaults, to change them, change the initialization
vocab_size = 21   # without the padding token
embed_dim = 16
kq_dim = 8
heads = 8


if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("→ Training on MPS GPU")
else:
    device = torch.device("cpu")
    print("→ Falling back to CPU")



#%% Initializing model
model = CodonTransformer(num_heads=heads)
model = model.to(device)


#%% Splitting data
train = int(padded.shape[0] * 0.8)
val = int(padded.shape[0] * 0.9)
test = int(padded.shape[0])

padded_in = padded[:, :-1]
padded_out = padded[:, 1:]

train_x = padded_in[:train]
val_x = padded_in[train:val]
test_x = padded_in[val:test - 1]

train_y = padded_out[:train]
val_y = padded_out[train:val]
test_y = padded_out[val:test - 1]

#%% Training loop

# === 1. Prepare DataLoaders ===
train_ds = TensorDataset(train_x, train_y)
val_ds = TensorDataset(val_x, val_y)

batch_size = 500
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

# === 2. Move Model and Set Up Optimizer ===
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# === 3. Training Loop ===
epochs = 10

for epoch in range(1, epochs + 1):

    # --- Training phase ---
    model.train()  # sets the model to training mode
    total_train_loss = 0.0

    for x_batch, y_batch in train_loader:
        # move batches to the chosen device
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # reset gradient information
        optimizer.zero_grad()

        # forward pass
        logits = model(x_batch)  # shape: (batch, seq_len, vocab_size)
        logits = logits.reshape(-1, vocab_size)  # shape: (batch * seq_len, vocab_size)
        targets = y_batch.reshape(-1)  # shape: (batch * seq_len)

        # compute loss, ignoring padding tokens
        loss = F.cross_entropy(logits, targets, ignore_index=pad_idx)

        # backpropagation (calculate gradients)
        loss.backward()

        # update weights
        optimizer.step()

        # accumulate loss, scaled by batch size
        total_train_loss += loss.item() * x_batch.size(0)

    # average train loss per epoch
    avg_train_loss = total_train_loss / len(train_loader.dataset)

    # --- Validation phase ---
    model.eval()  # sets the model to evaluation mode
    total_val_loss = 0.0
    print(f"Epoch {epoch:2d}: Train Loss = {avg_train_loss:.4f}")

## VALIDATION THING MOVED TO RECYCLING
#%% Saving the model
MODEL_PATH = "codon_transformer.pt"
torch.save(model.state_dict(), MODEL_PATH)

#%% Getting the model from whatever's saved
copy_model = CodonTransformer(num_heads=heads).to(device)
checkpoint = torch.load("codon_transformer.pt", map_location=device)
copy_model.load_state_dict(checkpoint)

#%% Test: test training overall

#%% Finding good batch size

def find_max_batch(model, ds, device, low=1, high=2048):
    best = low
    while low <= high:
        mid = (low + high) // 2
        loader = DataLoader(ds, batch_size=mid)
        try:
            xb, yb = next(iter(loader))
            xb, yb = xb.to(device), yb.to(device)
            _ = model(xb)
            best = mid
            low = mid + 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                high = mid - 1
            else:
                raise
    return best

# usage:
best_bs = find_max_batch(model, train_ds, device)
print("Max batch size without OOM:", best_bs)

