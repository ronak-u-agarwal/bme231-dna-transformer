#%% Imports
import torch
from transformer_more import CodonTransformer
from model_training import padded


#%% Setting device and copying model here
heads = 8

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("→ Training on MPS GPU")
else:
    device = torch.device("cpu")
    print("→ Falling back to CPU")

copy_model = CodonTransformer(num_heads=heads).to(device)
checkpoint = torch.load("codon_transformer.pt", map_location=device)
copy_model.load_state_dict(checkpoint)



#%% Start with a given DNA sequence and mutate it




#%% Now see what token the model flags/what it thinks should go there instead


#%% Look at the possible BP combos



#%% Automating this process with a bayesian prior from the hamming distance


#%%
