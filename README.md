# mylstm

This repository contains a minimal example of training a PyTorch LSTM model to map
informal product synonyms to canonical names while conditioning on metadata such as
`site_id` and `category_id`.

## Requirements

* Python 3.11+
* [PyTorch](https://pytorch.org/) (CPU or GPU build)
* NumPy

Install the dependencies with:

```bash
pip install torch numpy
```

## Usage

Run the training script:

```bash
python main.py
```

The script trains a small sequence-to-sequence model on a synthetic dataset of
synonym pairs and prints sample translations such as mapping `"air jordan"` to
`"jordan"` for the Nike sneakers category.
