# TiDE-Ab: De Novo Epitope-Specific Antibody Design via SE(3) Flow Matching
This repository contains the official implementation of the paper: "De Novo Epitope-Specific Antibody Design via SE(3) Flow Matching with Time-Dependent Guidance".
TiDE-Ab is a generative framework for the de novo design of therapeutic antibodies targeting specific epitopes. By leveraging Conditional SE(3) Flow Matching and a novel Time-Dependent Classifier-Free Guidance (TD-CFG) strategy, TiDE-Ab generates physically plausible antibody backbones and accurate global binding poses without relying on pre-aligned templates.


## Data Preparation
The model is trained on antibody-antigen complexes sourced from the SAbDab database, with a temporal cutoff of April 30, 2020.
The dataset is managed through metadata files located in data/splits/.
```bash
├── data
│   └── splits          # Dataset split metadata
│       ├── metadata_train.csv
│       ├── metadata_val.csv
│       └── metadata_test.csv
```

The metadata files in data/splits/ follow this schema:
| Column | Description |
| :--- | :--- |
| **pdb_name** | Unique identifier for the complex (e.g., `1yy9_D_C_A`). |
| **num_chains** | Total number of chains in the structure. |
| **seq_len** | Total sequence length of the complex. |
| **cluster** | Interaction cluster ID used for balanced sampling. |


## Running the Code

### Training

To start training with the default configuration:
```bash
python train.py
```

You can override any parameter directly from the command line using dot notation:
```bash
python train.py optimizer.lr=0.0005
```

### Inference

#### 1. Download Pre-trained Weights
First, download the pre-trained model weights and place the file in the `checkpoints/` directory:
* [**weights.pt**](https://drive.google.com/file/d/1VApyhdaiZxcULLts_a34n9RbB0JNOFoZ/view?usp=share_link)

#### 2. Run Inference on Test Set
To generate structures on the **test set** using the downloaded checkpoint, run the following command:

```bash
python inference.py
```


## Acknowledgements

This codebase is developed based on the [**FrameFlow**](https://github.com/microsoft/protein-frame-flow/tree/main) repository. We thank the original authors for their pioneering work on $SE(3)$ flow matching for protein structures.
