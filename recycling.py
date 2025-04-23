

#%% VALIDATION PART
# disable gradient calculation
with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        logits = logits.reshape(-1, vocab_size)
        targets = y_batch.reshape(-1)

        # sum loss to accurately average later
        loss = F.cross_entropy(logits, targets,
                               ignore_index=pad_idx,
                               reduction='sum')

        total_val_loss += loss.item()

# average val loss per token
avg_val_loss = total_val_loss / (len(val_loader.dataset) * (sequence_len - 1))

# --- Epoch summary ---
print(f"Epoch {epoch:2d}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

#%% Plot lengths of the genes
# Go through and convert from sequence of letters to codons

import matplotlib.pyplot as plt

# 1. Extract lengths (in bp)
lengths = [len(gene) for gene in exons]

# 2. Histogram of lengths
plt.figure()
plt.hist(lengths, bins=30)           # you can adjust bin count
plt.xlabel("Sequence length (bp)")
plt.ylabel("Count")
plt.title("Distribution of Sequence Lengths")
plt.show()



#%% Testing stuff out


#%% Super small test input

tokens_in = padded_in[0:2]
targets = padded_out[0:2]
tokens_in = tokens_in.to(device)
targets = targets.to(device)

logits = model(tokens_in)  # → (B, L-1, V)
logits = logits.reshape(-1, vocab_size)  # → (B*(L-1), V)
targets = targets.reshape(-1)  # → (B*(L-1),)
loss = F.cross_entropy(logits, targets,
                       ignore_index=pad_idx)
print(loss.item())