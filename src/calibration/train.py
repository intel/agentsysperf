#!/usr/bin/env python3.12
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Train decision tree for CPU bottleneck classification.

Learns optimal thresholds from calibration dataset, extracts rules,
and validates accuracy.

Usage:
    poetry run python -m src.calibration.train --input calibration_data.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Input calibration CSV dataset.",
    )
    parser.add_argument(
        "--max-depth", type=int, default=4,
        help="Maximum tree depth (default: 4).",
    )
    args = parser.parse_args()

    print(f"=== Threshold Calibration Training ===\n")

    # Load data
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} samples")
    print(f"Classes: {df['true_bottleneck'].value_counts().to_dict()}\n")

    # Feature engineering
    features = [
        "ipc",
        "cache_miss_pct",
        "cpu_pct_mean",
        "llc_miss_per_s",
        "branch_miss_pct",
    ]

    # Add derived features
    df["cpu_utilization_ratio"] = df["cpu_time_s"] / df["duration_s"]
    features.append("cpu_utilization_ratio")

    X = df[features]
    y = df["true_bottleneck"]

    print(f"Features: {', '.join(features)}\n")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples\n")

    # Train decision tree
    clf = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=2,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    cv_scores = cross_val_score(clf, X, y, cv=3)

    print(f"Train accuracy: {train_score:.2%}")
    print(f"Test accuracy: {test_score:.2%}")
    print(f"CV accuracy: {cv_scores.mean():.2%} (±{cv_scores.std():.2%})\n")

    # Extract learned thresholds
    print("=== Learned Decision Rules ===\n")
    tree_rules = export_text(clf, feature_names=features, max_depth=args.max_depth)
    print(tree_rules)

    # Feature importances
    print("\n=== Feature Importances ===\n")
    importances = sorted(
        zip(features, clf.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    for feat, imp in importances:
        if imp > 0.01:
            print(f"  {feat:30s} {imp:.3f}")

    # Predict on test set and show confusion
    print("\n=== Test Set Predictions ===\n")
    from sklearn.metrics import confusion_matrix

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    print("Confusion matrix:")
    print(f"{'':20s}", end="")
    for c in clf.classes_:
        print(f"{c:20s}", end="")
    print()
    for i, c in enumerate(clf.classes_):
        print(f"{c:20s}", end="")
        for j in range(len(clf.classes_)):
            print(f"{cm[i, j]:<20d}", end="")
        print()

    print(f"\n✓ Training complete. Test accuracy: {test_score:.1%}")
    return 0 if test_score >= 0.7 else 1


if __name__ == "__main__":
    sys.exit(main())
