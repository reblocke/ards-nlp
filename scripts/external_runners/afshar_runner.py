import argparse
import gzip
import hashlib
import json
import os
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description="Isolated legacy Afshar SVC runner")
    parser.add_argument("--model", required=True)
    parser.add_argument("--vectorizer", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-vectorizer-sha256", required=True)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main():
    args = parse_args()
    verify_sha256(args.model, args.expected_model_sha256)
    verify_sha256(args.vectorizer, args.expected_vectorizer_sha256)
    with open(args.vectorizer, "rb") as handle:
        vectorizer = pickle.load(handle)  # noqa: S301
    with open(args.model, "rb") as handle:
        model = pickle.load(handle)  # noqa: S301
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    temp = args.output + ".partial"
    try:
        with open(temp, "w", encoding="utf-8") as output:
            batch = []
            with gzip.open(args.packet, "rt", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    batch.append(json.loads(line))
                    if len(batch) >= args.batch_size:
                        write_batch(batch, vectorizer, model, output)
                        batch = []
            if batch:
                write_batch(batch, vectorizer, model, output)
        os.replace(temp, args.output)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_batch(batch, vectorizer, model, output):
    texts = [str(record["report_text"]) for record in batch]
    features = vectorizer.transform(texts)
    labels = model.predict(features)
    scores = model.predict_proba(features)[:, 1]
    for record, label, score in zip(batch, labels, scores):  # noqa: B905
        output.write(
            json.dumps(
                {
                    "case_id": str(record["case_id"]),
                    "prediction_score": float(score),
                    "raw_predicted_class": int(label),
                },
                separators=(",", ":"),
            )
            + "\n"
        )


def verify_sha256(path, expected):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != expected:
        raise ValueError(
            f"Checksum mismatch for {os.path.basename(path)}: expected {expected}, found {observed}"
        )


if __name__ == "__main__":
    main()
