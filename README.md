# CNN-BiLSTM-GRU Model for Electric Energy Consumption Forecasting

A hybrid deep learning model combining CNN, BiLSTM, and GRU for daily electricity consumption forecasting across Brazilian regions.

## Overview

This repository contains the implementation of the CNN-BiLSTM-GRU architecture proposed in:

> **A Hybrid CNN-BiLSTM-GRU Model For Electric Energy Consumption Forecasting**
> Luciano R. S. Leal, Byron L. D. Bezerra, João F. L. de Oliveira, Carlos E. L. da Costa
> Polytechnic School of Pernambuco — University of Pernambuco

The model uses a sequential pipeline — CNN for local feature extraction, BiLSTM for bidirectional temporal dependencies, and GRU for efficient pattern refinement — to forecast daily energy load (MW) in four Brazilian subsystems: North, Northeast, South, and Southeast.

## Architecture

```
Input → Conv1D → ReLU → MaxPooling1D → BiLSTM → GRU → Dropout → Dense → Output
```

| Component | Configuration |
|---|---|
| Conv1D | 64 filters, kernel size 3, ReLU |
| MaxPooling1D | Pool size 2 |
| BiLSTM | 50 units, return sequences |
| GRU | 50 units |
| Dropout | 0.2 |
| Dense | 1 unit (regression output) |

## Dataset

Historical daily energy consumption data (2019–2023) from Brazil's [National System Operator (ONS)](https://dados.ons.org.br).

- **Training:** 2019–2022
- **Testing:** 2023 (365 days)
- **Sliding window:** 30 days → 1-day ahead prediction
- **Normalization:** MinMax scaling (−1, +1)


## Requirements

```
tensorflow
scikit-learn
pandas
numpy
matplotlib
```


```

## License

This project is available for academic and research purposes.
