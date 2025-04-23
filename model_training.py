
#%%
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from import_data import padded, vocab_size, pad_idx
from transformer_more import CodonTransformer
import matplotlib.pyplot as plt
#%% Hyperparameters
## Embedding dimension, key/query dimension, number of self-attention
## heads in a single self-attention layer, etc.
#
# These are defaults, to change them, change the initialization
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

test = int(padded.shape[0])

## I think i should shuffle padded here

padded_in = padded[:, :-1]
padded_out = padded[:, 1:]

train_x = padded_in[:train]
test_x = padded_in[train:test - 1]

train_y = padded_out[:train]
test_y = padded_out[train:test - 1]

#%% Training loop

# === 1. Prepare DataLoaders ===
train_ds = TensorDataset(train_x, train_y)
test_ds = TensorDataset(test_x, test_y)

batch_size = 500
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size)

# === 2. Move Model and Set Up Optimizer ===
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# === 3. Training Loop ===
epochs = 50

train_list = []
test_list = []


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
    # model.eval()  # sets the model to evaluation mode
    total_test_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            logits = logits.reshape(-1, vocab_size)
            targets = y_batch.reshape(-1)

            loss = F.cross_entropy(logits, targets, ignore_index=pad_idx)

            total_test_loss += loss.item() * x_batch.size(0)

    # average val loss per token
    avg_test_loss = total_test_loss / len(test_loader.dataset)
    ## OOOH loss is lower, has fewer chucks, dividing here assumes they are all size 500, but actually not
    # --- Epoch summary ---
    print(f"Epoch {epoch:2d}: Train Loss = {avg_train_loss:.4f} | Test Loss = {avg_test_loss:.4f}")
    train_list.append(avg_train_loss)
    test_list.append(avg_test_loss)



#%% Plot the training curve
import matplotlib.pyplot as plt

plt.figure()
plt.plot(train_list, label='Train Loss', color='blue')
plt.plot(test_list, label='Test Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Training and Test Loss During Training")
plt.show()


#%% Reallllyyyy don't wanna run the next cell unless I'm sure


#%% Saving the model
MODEL_PATH = "codon_transformer2.pt"
torch.save(model.state_dict(), MODEL_PATH)

#%% Getting the model from whatever's saved
copy_model = CodonTransformer(num_heads=heads).to(device)
checkpoint = torch.load("codon_transformer.pt", map_location=device)
copy_model.load_state_dict(checkpoint)


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

