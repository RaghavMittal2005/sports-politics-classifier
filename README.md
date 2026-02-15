# Sports vs Politics Classifier

Small example project demonstrating a simple text classification pipeline
that distinguishes between "sports" and "politics" short texts. The
repository contains utilities to generate a synthetic dataset, extract
features, run experiments with multiple classifiers, and visualize results.


```
pip install -r requirements.txt
```

Quick start
1. Generate the synthetic dataset (writes `dataset.json`):

```bash
python create_dataset.py
```

2. (Optional) Inspect dataset statistics:

```bash
python analyse_dataset.py
```

3. Run classifier experiments and save results to `results.json`:

```bash
python classifier_comparison.py
```

4. Visualize results (opens matplotlib windows):

```bash
python visualize_results.py
```

5. Try the interactive demo classifier:

```bash
python demo.py
```

Files
- `create_dataset.py`: generates a balanced synthetic dataset and saves it to `dataset.json`.
- `feature_extraction.py`: tokenization, bag-of-words, TF-IDF and n-gram utilities.
- `classifier_comparison.py`: runs experiments comparing classifiers and feature methods, saves `results.json`.
- `visualize_results.py`: simple plots and textual summary from `results.json`.
- `analyse_dataset.py`: prints dataset-level statistics and examples.
- `demo.py`: trains a quick demo classifier and provides an interactive prompt.


