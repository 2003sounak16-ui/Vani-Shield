"""Build a leakage-resistant manifest for real-world voice-clone detection.

Input CSVs must contain path, label, source, and speaker_id when known.
The output keeps whole speakers and sources in one split, preventing the
overly optimistic random-clip validation used by the original 4,447 manifest.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


REQUIRED = {"path", "label", "source", "speaker_id"}


def make_split(df: pd.DataFrame, holdout: float, seed: int, split_name: str):
    group = df["speaker_id"].fillna("").astype(str)
    group = group.where(group.str.len() > 0, df["source"].astype(str))
    splitter = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=seed)
    train_idx, eval_idx = next(splitter.split(df, groups=group))
    result = df.copy()
    result["split"] = "train"
    result.iloc[eval_idx, result.columns.get_loc("split")] = split_name
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Labeled source CSV files")
    parser.add_argument("--output", default="robust_training_manifest.csv")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.inputs]
    df = pd.concat(frames, ignore_index=True)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["path"] = df["path"].map(str)
    unavailable = df[~df["path"].map(lambda p: Path(p).is_file())]
    if len(unavailable):
        raise FileNotFoundError(f"{len(unavailable)} manifest paths do not exist")

    # Keep the supplied external-test rows completely out of model selection.
    if "split" in df.columns:
        external_mask = df["split"].astype(str) == "external_test"
    else:
        external_mask = pd.Series(False, index=df.index)
    external = df[external_mask].copy()
    development = df.drop(external.index).reset_index(drop=True)
    development = make_split(development, args.test_size, args.seed, "val")
    result = pd.concat([development, external], ignore_index=True)
    result.to_csv(args.output, index=False)
    print(result.groupby(["split", "label"]).size())
    print(f"Wrote {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
