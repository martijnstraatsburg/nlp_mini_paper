# Comparative Evaluation of Morphological Feature Classification Methods

Mini-paper 5.1 for **Natural Language Processing (LIS060)** at Stockholm University.

Compares the Multiclass Perceptron and Multinomial Logistic Regression (MLR) on morphological feature classification for Swedish, using data from the [UniMorph](https://unimorph.github.io/) database. The full write-up is in [`NLP_Mini_paper___Comparative_evaluation_of_morphological_feature_classification_methods`](NLP_Mini_paper___Comparative_evaluation_of_morphological_feature_classification_methods.pdf).

---

## Requirements

Python 3 and its standard library only — no third-party packages.

---

## Data

The script expects two tab-separated UniMorph data files with three columns per line: `lemma`, `word form`, `morphological tag`.

```
råda	råder	V;IND;PRS;ACT
```

The Swedish training and test files (`swe-train`, `swe-test`) are available in the data folder.

---

## Usage

```
python mini_paper_5_1.py --train <train-file> --test <test-file> [options]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--train` | required | Path to training file |
| `--test` | required | Path to test file |
| `--epochs` | 3 | Training epochs for both models |
| `--lr` | 0.1 | Learning rate for logistic regression |
| `--l2` | 0.0 | L2 regularisation strength for logistic regression |
| `--bootstrap-samples` | 10000 | Number of paired bootstrap resampling iterations |

### Run command used in the paper

```
python mini_paper_5_1.py --train swe-train --test swe-test --epochs 5 --lr 0.2 --l2 0.0001
```

---

## Output

```
============================================================
Loading data...
Training examples: 129198
Test examples: 2500
Unique morph tags: 43

============================================================
Training Perceptron...
[Perceptron] epoch 1/5
[Perceptron] epoch 2/5
[Perceptron] epoch 3/5
[Perceptron] epoch 4/5
[Perceptron] epoch 5/5
Training time: 86.1s

Evaluating Perceptron on test set...
Accuracy: 0.6544 (65.4%)
Evaluation time: 0.3s

============================================================
Training Logistic Regression (lr=0.2, l2=0.0001)...
[LogReg] epoch 1/5
[LogReg] epoch 2/5
[LogReg] epoch 3/5
[LogReg] epoch 4/5
[LogReg] epoch 5/5
Training time: 291.8s

Evaluating Logistic Regression on test set...
Accuracy: 0.6940 (69.4%)
Evaluation time: 0.5s

============================================================
Results summary
----------------------------------------
Perceptron: 0.6544 - 86.1s
Logistic Regression: 0.6940 - 291.8s

Difference (Perc - LR): -0.0396 -> Logistic Regression is better

============================================================
Paired bootstrap test (R=10,000)
H0: both models have the same accuracy
H1 (one-sided): Perceptron > Logistic Regression

p(Perceptron > LR) = 1.0000
p(LR > Perceptron) = 0.0000

-> LR is significantly better than Perceptron (p < 0.05)

============================================================
Top 10 Perceptron features (highest weights):
 1. weight =  21 'ADJ suffix=a tag=POS;PL;INDF'
 2. weight =  20 'ADJ suffix=e tag=POS;MASC;SG;DEF'
 3. weight =  19 'N suffix=ets tag=GEN;SG;DEF'
 4. weight =  18 'ADJ suffix=t tag=POS;NEUT;SG;INDF'
 5. weight =  18 'N suffix=et tag=NOM;SG;DEF'
 6. weight =  16 'N suffix=ens tag=GEN;PL;DEF'
 7. weight =  16 'V suffix=es tag=SBJV;PASS;PRS'
 8. weight =  15 'N suffix=nas tag=GEN;PL;DEF'
 9. weight =  15 'N suffix=s tag=GEN;SG;INDF'
10. weight =  15 'V suffix=ande tag=V.PTCP;PRS'

Top 10 Logistic Regression features (highest weights):
 1. weight =  5.97 'N suffix=s tag=GEN;SG;INDF'
 2. weight =  5.11 'ADJ suffix=e tag=POS;MASC;SG;DEF'
 3. weight =  5.06 'ADJ suffix=t tag=POS;NEUT;SG;INDF'
 4. weight =  4.96 'ADJ suffix=a tag=POS;PL;INDF'
 5. weight =  4.79 'V suffix=e tag=SBJV;ACT;PRS'
 6. weight =  4.74 'V suffix=a tag=IND;PL;ACT;PRS'
 7. weight =  4.48 'V suffix=a tag=NFIN;ACT'
 8. weight =  4.34 'N suffix=r tag=NOM;PL;INDF'
 9. weight =  4.31 'V suffix=ts tag=V.CVB;PASS'
10. weight =  4.28 'N suffix=n tag=NOM;SG;DEF'
```

---

## Results

| Model | Accuracy | Train (s) | Eval (s) |
|---|---|---|---|
| Perceptron | 65.44% | 86.1 | 0.3 |
| MLR (η=0.2, λ=0.0001) | 69.40% | 291.8 | 0.5 |
| Δ (MLR − Perceptron) | +3.96% | | |

The accuracy difference is statistically significant (p < 0.0001, paired bootstrap, R=10,000).

---

## How it works

Features are formed by combining a character prefix or suffix of the word (lengths 1–4) with the POS label and a candidate morphological tag, e.g. `ADJ suffix=a tag=POS;PL;INDF`. Each word–candidate-class pair activates at most eight binary features.

**Perceptron** — integer weights, online error-driven updates (increment gold class, decrement predicted class by 1 on each mistake).

**MLR** — real-valued weights, stochastic gradient ascent on log-likelihood, softmax over all 43 candidate classes, lazy L2 regularisation.

**Paired bootstrap test** — implemented from scratch using Python's `random` module; R=10,000 resampling iterations, one-sided test.

---

## Reference

Martijn Bernard Straatsburg (2026). *Comparative Evaluation of Morphological Feature Classification Methods*. Mini-paper, Natural Language Processing (LIS060), Stockholm University.
