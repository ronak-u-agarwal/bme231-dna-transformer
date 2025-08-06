# DNA Sequence Anomaly Detection with a Custom Transformer

This project is a self-directed exploration into mastering the transformer architecture by building one from first principles and applying it to a novel domain: genomics. The goal was to build a model capable of understanding the syntax of DNA sequences to implicitly detect anomalies, such as mutations.

The entire model was built from scratch using only fundamental PyTorch linear layers to gain a deep understanding of the underlying mechanics of self-attention, positional encoding, and sequence-to-sequence learning.

### Architecture & Task

The model is a small-scale decoder-only transformer, trained on the task of **next-token prediction**.

* **Data Representation:** DNA gene sequences from various bacteria were tokenized into 3-character "codons" (e.g., 'ATG', 'GCC'). A vocabulary was built mapping each unique codon to an integer.
* **Model:** The model takes a sequence of codons as input and predicts the most likely next codon in the sequence.
* **Training:** The model was trained locally on a laptop using `.fasta` files containing gene sequences. The training process and model definition can be found in `model_training.py` and `transformer_basic.py`.

### Key Result: Anomaly Detection

While not explicitly trained for mutation correction, the model's understanding of sequence probability allows it to effectively "flag" anomalies.

A proof-of-concept was demonstrated by:
1.  Taking a valid gene sequence from the test set.
2.  Manually introducing a single-codon mutation.
3.  Feeding the mutated sequence to the trained model and calculating the probability distribution for the token at the mutated position.

The result was a clear success: the model assigned a **significantly lower probability** to the incorrect, mutated codon and a higher probability to the original, correct codon. This demonstrates that the model learned the "grammar" of the gene sequences and can spot tokens that do not belong.

### Tech Stack

* **Language:** Python
* **Libraries:** PyTorch

### How to Use

A pre-trained model is saved in `codon_transformer.pt`. The script `mutation_correction.py` provides a simple interface to test the anomaly detection capability.

1.  Ensure you have PyTorch installed.
2.  Run the `mutation_correction.py` script.
3.  The script will automatically load the model and a sample gene sequence, introduce a mutation, and print the model's probability distribution for the original vs. mutated codon at that position.

```bash
# Example
python "mutation correction.py"
```

The output will show the low probability assigned to the incorrect codon, demonstrating the model's ability to identify the anomaly.
