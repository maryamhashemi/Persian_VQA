# Persian VQA
Our aim in this project is to implement some state-of-the-art architectures designed for the task of the VQA and do experiments on the Persian dataset. For this purpose, no proper dataset is available hence we use the VQA v1 dataset and translate its questions and answers into Persian using Google translate and Targoman API.

## Project Directory Tree
```
├── Code 
│   ├── HieCoAttention   # source code of Hierarchical Question-Image Co-Attention model.
|   ├── ParsBERT         # source code for extracting features using ParsBERT.
│   ├── SAN              # source code of stacked attention network.
│   └── lstmQ+normI      # source code of baseline model.
├── Final Report 
├── Notebook 
│   ├── analyze_dataset.ipynb
│   └── Demo_VQA.ipynb
├── Progress Report
├── Proposal
├── .gitignore
└── Readme.md
```

