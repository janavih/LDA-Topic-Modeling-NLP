import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_documents(csv_path):
    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        raise ValueError("CSV must contain a column named 'text'.")
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)
    if df.empty:
        raise ValueError("No documents found in the CSV.")
    return df


def main():
    parser = argparse.ArgumentParser(description="Apply Latent Dirichlet Allocation (LDA) to a document collection.")
    parser.add_argument("--input", default="data/documents.csv", help="CSV file with a 'text' column.")
    parser.add_argument("--topics", type=int, default=4, help="Number of topics to discover (default: 4).")
    parser.add_argument("--words", type=int, default=10, help="Top words to display per topic.")
    parser.add_argument("--output", default="outputs", help="Directory for generated outputs.")
    args = parser.parse_args()

    if args.topics < 1:
        parser.error("--topics must be at least 1.")
    if args.words < 1:
        parser.error("--words must be at least 1.")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = load_documents(input_path)
    df["clean_text"] = df["text"].apply(clean_text)

    vectorizer = CountVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS),
        max_df=0.95,
        min_df=1,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
    )
    document_term_matrix = vectorizer.fit_transform(df["clean_text"])
    if document_term_matrix.shape[1] == 0:
        raise ValueError("No usable words found after preprocessing. Check the input documents.")

    topic_count = min(args.topics, max(1, document_term_matrix.shape[0]))
    model = LatentDirichletAllocation(
        n_components=topic_count,
        random_state=42,
        learning_method="batch",
        max_iter=20
    )
    document_topic_matrix = model.fit_transform(document_term_matrix)
    vocabulary = vectorizer.get_feature_names_out()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    topic_rows = []
    print("\n========== DISCOVERED TOPICS ==========")
    for topic_idx, weights in enumerate(model.components_):
        top_indices = weights.argsort()[-args.words:][::-1]
        top_words = [vocabulary[i] for i in top_indices]
        topic_rows.append({"topic_id": topic_idx + 1, "top_words": ", ".join(top_words)})
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

    topics_df = pd.DataFrame(topic_rows)
    topics_df.to_csv(out_dir / "topic_word_lists.csv", index=False)

    doc_topics = document_topic_matrix.argmax(axis=1) + 1
    df["dominant_topic"] = doc_topics
    df["topic_probability"] = document_topic_matrix.max(axis=1).round(4)
    for topic_idx in range(topic_count):
        df[f"topic_{topic_idx + 1}_probability"] = document_topic_matrix[:, topic_idx].round(4)
    df.to_csv(out_dir / "document_topic_assignments.csv", index=False)

    # A compact, readable report for the lab deliverable.
    report = [
        "LATENT DIRICHLET ALLOCATION (LDA) TOPIC INTERPRETATION REPORT",
        "=" * 62,
        f"Input file: {input_path}",
        f"Number of documents: {len(df)}",
        f"Number of discovered topics: {topic_count}",
        f"Vocabulary size: {len(vocabulary)}",
        "",
        "Topic-word distributions (top words are the strongest indicators):"
    ]
    for row in topic_rows:
        report.append(f"\nTopic {row['topic_id']}: {row['top_words']}")
    report += [
        "",
        "Interpretation guidance:",
        "- Read the top words for each topic and assign a human-readable label.",
        "- Documents can contain multiple topics; the CSV stores each topic probability.",
        "- The dominant topic is the topic with the highest probability for a document.",
        "- Topic labels are interpretive and should be chosen after reviewing the words and documents."
    ]
    (out_dir / "topic_interpretation_report.txt").write_text("\n".join(report), encoding="utf-8")

    # Optional visualization: topic-word weights as a bar chart.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(topic_count, 1, figsize=(10, max(3, 3 * topic_count)))
        if topic_count == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            weights = model.components_[i]
            idx = weights.argsort()[-min(args.words, len(vocabulary)):][::-1]
            words = vocabulary[idx]
            vals = weights[idx]
            ax.barh(words[::-1], vals[::-1])
            ax.set_title(f"Topic {i + 1}")
            ax.set_xlabel("Word weight")
        fig.tight_layout()
        fig.savefig(out_dir / "topic_word_visualization.png", dpi=150)
        plt.close(fig)
        print(f"\nVisualization saved to: {out_dir / 'topic_word_visualization.png'}")
    except ImportError:
        print("matplotlib not installed; skipping optional visualization.")

    print(f"\nOutputs saved in: {out_dir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
