### Cassava Disease Prediction

---

### Repository Structure

```
.
├── README.md
├── artefacts
│   ├── model_store  # Packaged model repository for Torchserve
│   └── models  # Exported model repository
├── data
│   └── train # Image folder (split into train and test set)
│       ├── cbb       # Disease 1
│       ├── cbsd      # Disease 2
│       ├── cgm       # Disease 3
│       ├── cmd       # Disease 4
│       └── healthy   # Disease 5
├── logs # Torchserve deployment logs
├── notebook.ipynb # Experimentation notebook
└── scripts # Script files
    ├── architectures.py
    ├── data_ops.py
    ├── handler_script.py
    ├── main.py
    ├── training_ops.py
    └── utils.py

```
