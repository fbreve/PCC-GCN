# LN-PCC + GCN (Label Noise Particle Competition and Cooperation + Graph Convolutional Networks)

This project implements and evaluates the integration of the **LN-PCC (Label Noise Particle Competition and Cooperation)** algorithm with **GCNs (Graph Convolutional Networks)** for semi-supervised learning on graphs in the presence of label noise.

LN-PCC is used to propagate labels, compute uncertainty, and filter/correct noisy labels prior to GCN training.

---

## 📂 Project Structure

### Core LN-PCC Files
*   **[lnpcc.py](lnpcc.py)**: Main implementation of the `LabelNoisePCC` class. Supports multiple execution backends (`cython`, `numba`, or `numpy`).
*   **[lnpcc_graph.py](lnpcc_graph.py)**: Helper functions for building label-guided k-NN graphs and graph fusion/augmentation.
*   **[lnpcc_numpy.py](lnpcc_numpy.py)**: Pure NumPy reference implementation of the LN-PCC propagation steps.
*   **[lnpcc_numba.py](lnpcc_numba.py)**: Numba-accelerated JIT implementations of the inner propagation loops.
*   **[lnpcc_step.pyx](lnpcc_step.pyx)**: High-performance Cython extension containing the critical propagation loops.
*   **[setup.py](setup.py)**: Setup script to compile the Cython extension.

### Simulation & Optimization Pipelines (Grid Search)
The optimization pipeline is split into three main incremental stages for Planetoid datasets (*Cora*, *CiteSeer*, and *PubMed*):

1.  **[lnpcc_stage1_pgrd_dexp.py](lnpcc_stage1_pgrd_dexp.py)**: Grid search over particle path parameters:
    *   `p_grd` (greediness probability in the random walk).
    *   `dexp` (distance exponent for edge weight calculation).
    *   *Note*: Uses fixed removal and relabeling thresholds at 0.5.
2.  **[lnpcc_stage2_thresholds.py](lnpcc_stage2_thresholds.py)**: Grid search to optimize the uncertainty removal threshold (`rem_thresh`) and relabeling/correction threshold (`rel_thresh`), using the best configurations from Stage 1.
3.  **[lnpcc_stage3_graph_knn.py](lnpcc_stage3_graph_knn.py)**: Grid search over graph fusion strategies combining the original graph with a k-NN graph for different values of $k$. Strategies include:
    *   `'u'` (Unused): Keep the original dataset graph only.
    *   `'s'` (Same): Add k-NN edges only between nodes that share the same LN-PCC predicted label.
    *   `'d'` (Difference): Add k-NN edges between nodes as long as they do not have different predicted labels.
    *   `'p'` (Pure): Add all k-NN edges without label constraints.

### Analysis & Visualization Scripts
*   **[plot_stage1_heatmaps.py](plot_stage1_heatmaps.py)**, **[plot_stage2_heatmaps.py](plot_stage2_heatmaps.py)**, **[plot_stage3_heatmaps.py](plot_stage3_heatmaps.py)**: Generate heatmaps comparing GCN accuracy gains (LNPCC+GCN vs. Baseline GCN trained directly on noisy labels).
*   **[select_best_config.py](select_best_config.py)**: Parses statistics CSV files to assist in selecting optimal hyperparameters.
*   **[get_best_parameters.py](get_best_parameters.py)**: Utility to extract the best performing parameters.

### Example Script
*   **[example_lnpcc.py](example_lnpcc.py)**: Simple demonstration script comparing classic PCC and LN-PCC on Scikit-Learn's *Wine* dataset.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ and the required packages installed:
```bash
pip install numpy scipy scikit-learn pandas matplotlib seaborn joblib psutil torch torch-geometric cython numba
```

### 2. Compile the Cython Extension
To compile the high-performance Cython backend module, run the following command in the root directory:
```bash
python setup.py build_ext --inplace
```
*This compiles the module into a `.pyd` (Windows) or `.so` (Linux/macOS) binary, which is automatically loaded by the `LabelNoisePCC` class.*

### 3. Run the Example
Verify the installation by running the example script on the Wine dataset:
```bash
python example_lnpcc.py
```
This loads the *Wine* dataset, simulates a semi-supervised setup with artificial label noise, and prints a classification report comparison between classic PCC and LN-PCC.

---

## 🛠️ How LN-PCC Works
Unlike the classic PCC where labeled nodes are frozen (domination potentials are fixed), **LN-PCC** incorporates:
1.  **Label-guided Graph Construction**: k-NN graph creation prioritizes edges between labeled nodes of the same class.
2.  **Active Potential Updates**: Labeled nodes actively update their dominance potentials during competitive propagation, allowing particles to overcome initial noisy labels.
3.  **Accumulated Propagation**: Runs `n_repeats` times and aggregates domination potentials to calculate final node confidence and predictions.
4.  **Uncertainty Filtering**: Node uncertainty is computed from the final ownership degrees (`owndeg`). High-uncertainty nodes are filtered out of GCN training, whereas low-uncertainty nodes that disagree with their noisy label are corrected.
