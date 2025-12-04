# GNN Models - Reorganized Structure

## 📁 Directory Structure

```
src/gnn/
├── models/              # GNN model architectures
│   ├── __init__.py
│   └── sta_gcn.py      # STA-GCN implementation
│
├── trainers/           # Training scripts
│   ├── __init__.py
│   ├── train_sta_gcn.py           # Main training
│   └── incremental_train_sta_gcn.py  # Incremental training
│
├── evaluators/         # Evaluation scripts
│   ├── __init__.py
│   └── evaluate_sta_gcn.py        # Evaluation with baselines
│
├── dataset.py          # TrafficDataset class
├── graph_utils.py      # Graph operations (Chebyshev, Laplacian)
└── __init__.py
```

## 🚀 Quick Start

### Training STA-GCN
```bash
python src/gnn/trainers/train_sta_gcn.py --epochs 50 --device cpu --batch_size 4 --sigma 50.0
```

### Evaluating STA-GCN
```bash
python src/gnn/evaluators/evaluate_sta_gcn.py --checkpoint outputs/models/best_model.pt --compare_baselines --sigma 50.0
```

### Incremental Training
```bash
python src/gnn/trainers/incremental_train_sta_gcn.py --last_n_days 7
```

## 📊 Adding New Models

To add a new GNN model (e.g., GraphSAGE, GAT):

1. **Create model file**: `src/gnn/models/your_model.py`
2. **Create trainer**: `src/gnn/trainers/train_your_model.py`
3. **Create evaluator**: `src/gnn/evaluators/evaluate_your_model.py`
4. **Update `models/__init__.py`**: Add your model to exports

### Example: Adding DCRNN

```python
# src/gnn/models/dcrnn.py
class DCRNN(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Your implementation

# src/gnn/trainers/train_dcrnn.py
from models.dcrnn import DCRNN
# Training logic

# src/gnn/evaluators/evaluate_dcrnn.py
from models.dcrnn import DCRNN
# Evaluation logic
```

## 🔧 Shared Components

- **dataset.py**: TrafficDataset - reusable for all models
- **graph_utils.py**: Graph operations - reusable utilities
- **Baseline models**: Included in evaluators for comparison

## 📝 Notes

- All models share the same dataset format
- Evaluation includes baseline comparisons (Historical Avg, Last Value, Linear Regression)
- Training outputs saved to `outputs/models/`
- Evaluation outputs saved to `outputs/evaluation/`
