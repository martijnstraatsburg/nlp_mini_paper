#!/usr/bin/env python3
"""
run with: python3 mini_paper_5_1.py --train swe-train --test swe-test
          --epochs 5 --lr 0.2 --l2 0.0001

output:
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
"""

import argparse
import math
import random
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

def read_data(filepath: str) -> List[Tuple[str, str, str]]:
    """Read morphological data from tab-separated file

    Args:
        filepath (str): Path to input data file

    Returns:
        List[Tuple[str, str, str]]: List of (word, pos, morph) tuples
    """
    # Initialize empty list to store parsed data entries
    data = []
    # Open file with utf-8 encoding and iterate over each line
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            # Strip leading and trailing whitespace from line
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Split line by tab character into parts
            parts = line.split("\t")
            # Skip lines that do not have exactly 3 tab-separated fields
            if len(parts) != 3:
                continue
            # Unpack parts into lemma, word, and full tag fields
            _lemma, word, full_tag = parts
            # Replace spaces in word with underscores for consistent formatting
            word = word.replace(" ", "_")
            # Split full tag into pos and morphological features
            # if semicolon present
            if ";" in full_tag:
                pos, morph = full_tag.split(";", 1)
            else:
                pos, morph = full_tag, ""
            # Append parsed tuple to data list
            data.append((word, pos, morph))
    return data

def generate_features(word: str, pos: str, morph: str) -> Set[str]:
    """Generate prefix and suffix features for given word, pos, and morph tag

    Args:
        word (str): Surface word form
        pos (str): Part-of-speech tag
        morph (str): Morphological feature string

    Returns:
        Set[str]: Set of feature strings combining pos,
            prefix/suffix, and morph
    """
    # Initialize empty set to collect feature strings
    features: Set[str] = set()
    # Compute word length for bounding prefix and suffix extraction
    word_len = len(word)
    # Iterate over prefix/suffix lengths from 1 up to min of 4 and word length
    for i in range(1, min(5, word_len) + 1):
        # Add prefix feature combining pos, prefix of length i, and morph tag
        features.add(f"{pos} prefix={word[:i]} tag={morph}")
        # Add suffix feature combining pos, suffix of length i, and morph tag
        features.add(f"{pos} suffix={word[-i:]} tag={morph}")
    return features

# Perceptron

def perceptron_predict(
    weights: Dict[Tuple[str, str], int],
    word: str,
    pos: str,
    all_morphs: List[str],
) -> str:
    """Predict morphological tag using perceptron weights via
    argmax over all candidate morphs

    Args:
        weights (Dict[Tuple[str, str], int]): Feature-morph weight dictionary
        word (str): Surface word form
        pos (str): Part-of-speech tag
        all_morphs (List[str]): List of all possible morphological tags

    Returns:
        str: Predicted morphological tag with highest score
    """
    # Initialize best morph to first candidate and best score
    # to negative infinity
    best_morph = all_morphs[0]
    best_score = -float("inf")
    # Iterate over all candidate morphological tags
    for morph in all_morphs:
        # Compute score as sum of weights for each feature
        # paired with current morph
        score = sum(weights.get((feat, morph), 0)
                    for feat in generate_features(word, pos, morph))
        # Update best morph if current score exceeds best seen so far
        if score > best_score:
            best_score = score
            best_morph = morph
    return best_morph

def train_perceptron(
    train_data: List[Tuple[str, str, str]],
    epochs: int = 3,
) -> Tuple[Dict[Tuple[str, str], int], List[str]]:
    """Train perceptron model on morphological tagging data
    using online updates

    Args:
        train_data (List[Tuple[str, str, str]]): List of
            (word, pos, morph) training examples
        epochs (int, optional): Number of passes over training
            data Defaults to 3

    Returns:
        Tuple[Dict[Tuple[str, str], int], List[str]]: Trained
            weight dictionary and sorted morph list
    """
    # Collect all unique morph tags from training data and sort for determinism
    all_morphs = sorted({morph for _, _, morph in train_data})
    # Initialize weight dictionary with default integer value of zero
    weights: Dict[Tuple[str, str], int] = defaultdict(int)

    # Loop over each epoch of training
    for epoch in range(epochs):
        print(f"[Perceptron] epoch {epoch + 1}/{epochs}")
        # Iterate over each training example
        for word, pos, true_morph in train_data:
            # Predict morph tag using current weights
            pred = perceptron_predict(weights, word, pos, all_morphs)
            # Update weights only when prediction does not match true label
            if pred != true_morph:
                # Increment weights for features associated with true morph
                for feat in generate_features(word, pos, true_morph):
                    weights[(feat, true_morph)] += 1
                # Decrement weights for features associated with
                # incorrectly predicted morph
                for feat in generate_features(word, pos, pred):
                    weights[(feat, pred)] -= 1

    return weights, all_morphs

def perceptron_get_predictions(
    data: List[Tuple[str, str, str]],
    weights: Dict[Tuple[str, str], int],
    all_morphs: List[str],
) -> List[bool]:
    """Generate list of correctness flags for each example
    using trained perceptron

    Args:
        data (List[Tuple[str, str, str]]): List of (word, pos,
            morph) examples to evaluate
        weights (Dict[Tuple[str, str], int]): Trained perceptron
            weight dictionary
        all_morphs (List[str]): List of all possible morphological tags

    Returns:
        List[bool]: Boolean list where True indicates correct prediction
    """
    # Return boolean list comparing predicted morph to true morph
    # for each example
    return [
        perceptron_predict(weights, word, pos, all_morphs) == true_morph
        for word, pos, true_morph in data
    ]

# Multinomial Logistic Regression

def logreg_score_all(
    weights: Dict[str, float],
    word: str,
    pos: str,
    all_morphs: List[str],
) -> Dict[str, float]:
    """Compute raw logistic regression scores for all candidate
    morphological tags

    Args:
        weights (Dict[str, float]): Feature weight dictionary
        word (str): Surface word form
        pos (str): Part-of-speech tag
        all_morphs (List[str]): List of all possible morphological tags

    Returns:
        Dict[str, float]: Mapping from each morph tag to its raw score
    """
    # Build dictionary mapping each morph to sum of weights over its features
    return {
        morph: sum(weights.get(feat, 0.0)
                   for feat in generate_features(word, pos, morph))
        for morph in all_morphs
    }

def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    """Convert raw scores to probability distribution via
    numerically stable softmax

    Args:
        scores (Dict[str, float]): Raw score dictionary keyed by class label

    Returns:
        Dict[str, float]: Probability distribution over class labels
    """
    # Subtract maximum score for numerical stability before exponentiation
    max_s = max(scores.values())
    # Compute exponentiated shifted scores for each class
    exp_s = {c: math.exp(s - max_s) for c, s in scores.items()}
    # Sum all exponentiated values to form normalizing constant
    total = sum(exp_s.values())
    # Normalize each exponentiated score by total to obtain probabilities
    return {c: v / total for c, v in exp_s.items()}

def logreg_predict(
    weights: Dict[str, float],
    word: str,
    pos: str,
    all_morphs: List[str],
) -> str:
    """Predict morphological tag using logistic regression
    weights via argmax over scores

    Args:
        weights (Dict[str, float]): Feature weight dictionary
        word (str): Surface word form
        pos (str): Part-of-speech tag
        all_morphs (List[str]): List of all possible morphological tags

    Returns:
        str: Predicted morphological tag with highest raw score
    """
    # Compute raw scores for all candidate morphs
    scores = logreg_score_all(weights, word, pos, all_morphs)
    # Return morph with highest score
    return max(scores, key=scores.__getitem__)

def train_logreg(
    train_data: List[Tuple[str, str, str]],
    epochs: int = 3,
    lr: float = 0.1,
    l2: float = 0.0,
) -> Tuple[Dict[str, float], List[str]]:
    """Train multinomial logistic regression with optional L2
    regularization via lazy updates

    Args:
        train_data (List[Tuple[str, str, str]]): List of
            (word, pos, morph) training examples
        epochs (int, optional): Number of passes over training
            data Defaults to 3
        lr (float, optional): Learning rate for gradient updates
            Defaults to 0.1
        l2 (float, optional): L2 regularization strength Defaults to 0.0

    Returns:
        Tuple[Dict[str, float], List[str]]: Trained weight
            dictionary and sorted morph list
    """
    # Collect all unique morph tags from training data and sort for determinism
    all_morphs = sorted({morph for _, _, morph in train_data})
    # Initialize weight dictionary with default float value of zero
    weights: Dict[str, float] = defaultdict(float)
    # Track last regularization step per feature for lazy L2 application
    last_reg: Dict[str, int] = defaultdict(int)
    # Precompute per-step decay factor from learning rate and L2 strength
    decay = 1.0 - lr * l2
    # Initialize global step counter for lazy regularization tracking
    t = 0

    # Loop over each epoch of training
    for epoch in range(epochs):
        print(f"[LogReg] epoch {epoch + 1}/{epochs}")
        # Iterate over each training example
        for word, pos, true_morph in train_data:
            # Increment global step counter
            t += 1
            # Compute raw scores for all candidate morphs using current weights
            scores = logreg_score_all(weights, word, pos, all_morphs)
            # Convert raw scores to probability distribution via softmax
            probs = _softmax(scores)

            # Compute gradient and update weights for each candidate morph
            for morph in all_morphs:
                # Gradient is 1 minus predicted probability for
                # true morph, else negative probability
                grad = (1.0 if morph == true_morph else 0.0) - probs[morph]
                # Skip update if gradient is effectively zero
                if abs(grad) < 1e-9:
                    continue
                # Update each feature weight with gradient and
                # optional lazy L2 decay
                for feat in generate_features(word, pos, morph):
                    if l2 > 0.0:
                        # Apply accumulated L2 decay for steps
                        # since last regularization of this feature
                        steps = t - last_reg[feat]
                        weights[feat] *= decay ** steps
                        last_reg[feat] = t
                    # Apply gradient ascent step scaled by learning rate
                    weights[feat] += lr * grad

    return weights, all_morphs

def logreg_get_predictions(
    data: List[Tuple[str, str, str]],
    weights: Dict[str, float],
    all_morphs: List[str],
) -> List[bool]:
    """Generate list of correctness flags for each example
    using trained logistic regression

    Args:
        data (List[Tuple[str, str, str]]): List of (word, pos,
            morph) examples to evaluate
        weights (Dict[str, float]): Trained logistic regression
            weight dictionary
        all_morphs (List[str]): List of all possible morphological tags

    Returns:
        List[bool]: Boolean list where True indicates correct prediction
    """
    # Return boolean list comparing predicted morph to true morph
    # for each example
    return [
        logreg_predict(weights, word, pos, all_morphs) == true_morph
        for word, pos, true_morph in data
    ]

def accuracy(correct_flags: List[bool]) -> float:
    """Compute accuracy as fraction of correct predictions over total examples

    Args:
        correct_flags (List[bool]): Boolean list of per-example correctness

    Returns:
        float: Accuracy value between 0 and 1
    """
    # Divide count of True values by total number of examples
    return sum(correct_flags) / len(correct_flags)

# Paired bootstrap significance test

def paired_bootstrap_test(
    correct_a: List[bool],
    correct_b: List[bool],
    R: int = 10_000,
    seed: int = 42,
) -> float:
    """Estimate one-sided p-value for model A being better than
    model B via paired bootstrap resampling

    Args:
        correct_a (List[bool]): Correctness flags for model A
        correct_b (List[bool]): Correctness flags for model B
        R (int, optional): Number of bootstrap resampling
            iterations Defaults to 10_000
        seed (int, optional): Random seed for reproducibility Defaults to 42

    Returns:
        float: Estimated p-value for H1 that model A outperforms model B
    """
    # Initialize seeded random generator for reproducible bootstrap sampling
    rng = random.Random(seed)
    # Store number of test examples
    n = len(correct_a)
    # Compute observed accuracy difference between model A and model B
    observed_delta = accuracy(correct_a) - accuracy(correct_b)

    # Initialize counter for bootstrap samples where shifted delta
    # meets significance threshold
    count = 0
    # Run R bootstrap resampling iterations
    for _ in range(R):
        # Sample n indices with replacement from test set
        indices = [rng.randrange(n) for _ in range(n)]
        # Build bootstrap sample for model A using sampled indices
        sample_a = [correct_a[i] for i in indices]
        # Build bootstrap sample for model B using sampled indices
        sample_b = [correct_b[i] for i in indices]
        # Compute accuracy difference on bootstrap sample
        delta_b = accuracy(sample_a) - accuracy(sample_b)
        # Count sample if bootstrap delta shifted by observed
        # delta is at least observed delta
        if delta_b - observed_delta >= observed_delta:
            count += 1

    # Return fraction of bootstrap samples satisfying condition
    # as p-value estimate
    return count / R

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mini-paper 5.1: Perceptron vs. Logistic Regression "
                    "on morphological feature classification"
    )
    parser.add_argument("--train", required=True, help="Path to training file")
    parser.add_argument("--test", required=True, help="Path to test file")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs for both models (default: 3)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate for logistic regression"
                        "(default: 0.1)")
    parser.add_argument("--l2", type=float, default=0.0,
                        help="L2 regularise strength for logistic regression"
                             "(default: 0.0, disabled)")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000,
                        help="Bootstrap resampling iters (default: 10000)")
    args = parser.parse_args()

    print("=" * 60)
    print("Loading data...")
    # Load training data from specified filepath
    train_data = read_data(args.train)
    # Load test data from specified filepath
    test_data = read_data(args.test)
    print(f"Training examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")
    # Count unique morphological tags present in training data
    num_classes = len({morph for _, _, morph in train_data})
    print(f"Unique morph tags: {num_classes}")

    print("\n" + "=" * 60)
    print("Training Perceptron...")
    # Record start time before perceptron training
    t0 = time.time()
    perc_weights, perc_morphs = train_perceptron(train_data,
                                                 epochs=args.epochs)
    # Compute elapsed training time for perceptron
    perc_time = time.time() - t0
    print(f"Training time: {perc_time:.1f}s")
    print("\nEvaluating Perceptron on test set...")
    # Record start time before perceptron evaluation
    t0 = time.time()
    perc_correct = perceptron_get_predictions(test_data,
                                              perc_weights, perc_morphs)
    # Compute elapsed evaluation time for perceptron
    perc_eval_time = time.time() - t0
    # Compute perceptron accuracy on test set
    perc_acc = accuracy(perc_correct)
    print(f"Accuracy: {perc_acc:.4f} ({perc_acc:.1%})")
    print(f"Evaluation time: {perc_eval_time:.1f}s")

    print("\n" + "=" * 60)
    print(f"Training Logistic Regression (lr={args.lr}, l2={args.l2})...")
    # Record start time before logistic regression training
    t0 = time.time()
    lr_weights, lr_morphs = train_logreg(
        train_data, epochs=args.epochs, lr=args.lr, l2=args.l2
    )
    # Compute elapsed training time for logistic regression
    lr_time = time.time() - t0
    print(f"Training time: {lr_time:.1f}s")
    print("\nEvaluating Logistic Regression on test set...")
    # Record start time before logistic regression evaluation
    t0 = time.time()
    lr_correct = logreg_get_predictions(test_data, lr_weights, lr_morphs)
    # Compute elapsed evaluation time for logistic regression
    lr_eval_time = time.time() - t0
    # Compute logistic regression accuracy on test set
    lr_acc = accuracy(lr_correct)
    print(f"Accuracy: {lr_acc:.4f} ({lr_acc:.1%})")
    print(f"Evaluation time: {lr_eval_time:.1f}s")

    print("\n" + "=" * 60)
    print("Results summary")
    print("-" * 40)
    print(f"Perceptron: {perc_acc:.4f} - {perc_time:.1f}s")
    print(f"Logistic Regression: {lr_acc:.4f} - {lr_time:.1f}s")
    # Compute accuracy difference with perceptron as reference
    delta = perc_acc - lr_acc
    # Determine which model achieved higher accuracy
    winner = "Perceptron" if delta > 0 else "Logistic Regression"
    print(f"\nDifference (Perc - LR): {delta:+.4f} -> {winner} is better")

    print("\n" + "=" * 60)
    print(f"Paired bootstrap test (R={args.bootstrap_samples:,})")
    print("H0: both models have the same accuracy")
    print("H1 (one-sided): Perceptron > Logistic Regression")
    # Estimate p-value for perceptron being better than logistic regression
    p_perc_better = paired_bootstrap_test(
        perc_correct, lr_correct, R=args.bootstrap_samples
    )
    # Estimate p-value for logistic regression being better than perceptron
    p_lr_better = paired_bootstrap_test(
        lr_correct, perc_correct, R=args.bootstrap_samples
    )
    print(f"\np(Perceptron > LR) = {p_perc_better:.4f}")
    print(f"p(LR > Perceptron) = {p_lr_better:.4f}")
    alpha = 0.05
    if p_perc_better < alpha:
        print(f"\n-> Perceptron is significantly better than LR (p < {alpha})")
    elif p_lr_better < alpha:
        print(f"\n-> LR is significantly better than Perceptron (p < {alpha})")
    else:
        print(f"\n-> No statistically significant difference (alpha ={alpha})")

    print("\n" + "=" * 60)
    print("Top 10 Perceptron features (highest weights):")
    # Sort perceptron feature-morph pairs by weight in
    # descending order and take top 10
    top_perc = sorted(perc_weights.items(),
                      key=lambda kv: kv[1], reverse=True)[:10]
    for rank, ((feat, morph), w) in enumerate(top_perc, 1):
        print(f"{rank:2d}. weight ={w:4d} '{feat}'")

    print("\nTop 10 Logistic Regression features (highest weights):")
    # Sort logistic regression features by weight in descending
    # order and take top 10
    top_lr = sorted(lr_weights.items(),
                    key=lambda kv: kv[1], reverse=True)[:10]
    for rank, (feat, w) in enumerate(top_lr, 1):
        print(f"{rank:2d}. weight ={w:6.2f} '{feat}'")


if __name__ == "__main__":
    main()
