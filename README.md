# LDA Topic Modeling — NLP Lab Project

This project implements **Latent Dirichlet Allocation (LDA)** on a collection of text documents, matching the lab requirements:
- Preprocess text data
- Build a document-term representation
- Train an LDA topic model
- Interpret topic-word distributions
- Save the preprocessed corpus, model outputs, topic-word lists, report, and optional visualization

## Requirements
- Python 3.9 or newer
- Internet connection only for installing packages (the example dataset runs locally)

## Run on Windows

1. Extract the ZIP file.
2. Open Command Prompt or the VS Code terminal in the extracted `LDA_Topic_Modeling_Project` folder.
3. (Recommended) Create and activate a virtual environment:

```bat
python -m venv .venv
.venv\Scripts\activate
```

4. Install libraries:

```bat
python -m pip install -r requirements.txt
```

5. Run with the included sample corpus:

```bat
python main.py
```

You should see topic-word lists in the terminal. Results will be saved in the `outputs` folder.

## Customize the input
Edit `data/documents.csv`. It must have a column named `text`, with one document per row. Then run:

```bat
python main.py --input data/documents.csv --topics 4 --words 10
```

You can change the number of topics, for example:

```bat
python main.py --topics 3 --words 8
```

For a different input CSV:

```bat
python main.py --input data/my_documents.csv --topics 5 --words 12 --output outputs
```

## Output files
- `outputs/topic_word_lists.csv` — top words for each topic
- `outputs/document_topic_assignments.csv` — each document's dominant topic and topic probabilities
- `outputs/topic_interpretation_report.txt` — concise interpretation report
- `outputs/topic_word_visualization.png` — optional bar-chart visualization
- The input corpus is retained in `data/documents.csv`; the script also writes cleaned text into the assignments CSV.

## How the method works
1. **Preprocessing:** lowercases text, removes URLs, punctuation, digits, and extra whitespace.
2. **Document-term matrix:** `CountVectorizer` converts documents into word-count vectors and removes common English stop words.
3. **LDA training:** estimates topic-word and document-topic distributions.
4. **Interpretation:** displays the most strongly weighted words for each topic. Topic names should be assigned by a person after reviewing the words.

## Common issues
- `python is not recognized`: install Python and select **Add Python to PATH**, then reopen the terminal.
- `No module named sklearn`: activate the environment and run `python -m pip install -r requirements.txt`.
- `No usable words found`: make sure the CSV's `text` column contains English text with words longer than one character.
- To rerun from scratch, delete the generated `outputs` folder and run the command again.

**Note:** LDA is an unsupervised topic-discovery method. The discovered topics and their labels depend on the dataset and number of topics; they are not guaranteed to match predefined categories.
