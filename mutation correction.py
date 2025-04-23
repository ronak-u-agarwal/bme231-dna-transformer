#%% Imports
import torch
from transformer_more import CodonTransformer
import torch.nn.functional as F
from import_data import *


#%% Setting device


if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("→ Training on MPS GPU")
else:
    device = torch.device("cpu")
    print("→ Falling back to CPU")


#%% Loading saved model
heads = 8
copy_model = CodonTransformer(num_heads=heads).to(device)
checkpoint = torch.load("codon_transformer.pt", map_location=device)
copy_model.load_state_dict(checkpoint)
copy_model.eval()


#%% Basic loss to make sure this copied model works

padded_in = padded[:, :-1]
padded_out = padded[:, 1:]

test_x = padded_in[13000:].to(device)
test_y = padded_out[13000:].to(device)
loss_num = 0
with torch.no_grad():
    logits = copy_model(test_x)  # shape: (batch, seq_len, vocab_size)
    logits = logits.reshape(-1, vocab_size)  # shape: (batch * seq_len, vocab_size)
    targets = test_y.reshape(-1)  # shape: (batch * seq_len)
    loss = F.cross_entropy(logits, targets, ignore_index=pad_idx)
    loss_num = loss.item()

print(loss_num)



#%% Start with a given DNA sequence and mutate it
rand = 12545
codon_seq = padded[rand]
bp_seq = corresponding_BP[rand]


def replace_char(s, index, char):
    return s[:index] + char + s[index + 1:]

print(bp_seq[99:102])
print(codon_seq[33])
bp_mutated = replace_char(bp_seq, 100, "A")
print(bp_mutated[99:102])

codon_mutated = seq_to_codon_indices(bp_mutated)
print(codon_mutated[33])

# Ok so basically G --> D, which is 6 --> 3 in the indices
# Also we need to pad the new guy

#%% Padding the new guy
vocab_size = 21    # no padding token again
new_tensor_seqs = [torch.tensor(codon_mutated, dtype=torch.long), codon_seq]
pad_idx = vocab_size
codon_mutated = pad_sequence(new_tensor_seqs,
                      batch_first=True,
                      padding_value=pad_idx)[0]
vocab_size += 1
print(codon_mutated.shape)


#%% Now see what token the model flags/what it thinks should go there instead
# for simplicity, now we condition on the model knowing that there is one replacement mutation
# and we want it to basically see how "cohesive" the mutated sequence is, and see where it isn't very cohesive
# so bascically, when we run the model on the first 33 codons (0-32 indices), then the prediction for the
# prediction at 33 should mismatch the "actual"
# in other words, for each position, we take the cross entropy of the prediction and actual, and when that spikes
# we think we have the mutation


in_x = codon_mutated[:-1]
in_x = in_x.unsqueeze(0)
in_x = in_x.to(device)
target_y = codon_mutated[1:]
target_y = target_y.to(device)


logits = copy_model(in_x)  # shape: (batch, seq_len, vocab_size)
logits = logits.reshape(-1, vocab_size)  # shape: (batch * seq_len, vocab_size)
targets = target_y.reshape(-1)  # shape: (batch * seq_len)


## So logits has a prediction of "next token"
# That means that logits[32] should predict 6, when we actually see 3

# logits[32]:
# tensor([-6.1451,  1.2201, -0.6001,  1.0074,  1.0891,  0.4648,  1.0900, -0.2984,
#          1.1203,  1.0387,  1.3520,  0.0109,  0.6450,  0.4014,  0.3069,  0.5385,
#          1.0073,  0.8656,  1.1052, -1.2623,  0.2994, -8.2150], device='mps:0',
#        grad_fn=<SelectBackward0>)

# SUPER interesting result: swapping bp_seq[100] doesn't every create a problem that the model
# catches, happens to give high logit to all the output codons, don't think it's a coincidence


#%% Can do again but change position 99 to T from G, so amino acid G --> C, so 6 --> 2.


bp_mutated = replace_char(bp_seq, 99, "T")
codon_mutated = seq_to_codon_indices(bp_mutated)

vocab_size = 21    # no padding token again
new_tensor_seqs = [torch.tensor(codon_mutated, dtype=torch.long), codon_seq]
pad_idx = vocab_size
codon_mutated = pad_sequence(new_tensor_seqs,
                      batch_first=True,
                      padding_value=pad_idx)[0]
vocab_size += 1



in_x = codon_mutated[:-1]
in_x = in_x.unsqueeze(0)
in_x = in_x.to(device)
target_y = codon_mutated[1:]
target_y = target_y.to(device)


logits = copy_model(in_x)  # shape: (batch, seq_len, vocab_size)
logits = logits.reshape(-1, vocab_size)  # shape: (batch * seq_len, vocab_size)
targets = target_y.reshape(-1)  # shape: (batch * seq_len)


#%% "Identifying" the 33rd index as the place with the issue
# How can I take the cross entropy loss between logits and targets,
# but in a way where the position-wise cross entropy is saved in a 249 dimensional array, so i can see where the
# mutation is likely to have occurred?
losses = F.cross_entropy(logits, targets, reduction='none')
pad_idx = 21  # or whatever you're using
mask = targets != pad_idx  # shape: [249], True where not padding
masked_losses = losses * mask

print(torch.argmax(masked_losses))

## OMG it works and returns 32, as expected !!


#%% Look at the possible BP combos



#%% Automating this process with a bayesian prior from the hamming distance


#%%
