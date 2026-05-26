# NewsLens

An end-to-end NLP pipeline that ingests live news articles from multiple outlets on the same
event, classifies political bias using a fine-tuned transformer, clusters narratives to show
how different outlets frame the same story, and monitors bias drift over time.

The core idea comes from Entman's framing theory (1993): media outlets don't just report
events, they frame them by selecting which aspects to emphasise, who the implied villain and
victim are, and what the implied solution is. NewsLens makes that visible computationally.

---

## What it does

**1. Bias classification** - Every article is classified as left-leaning, centre, or
right-leaning using a fine-tuned RoBERTa-base model. Predictions below a confidence
threshold of 0.85 are flagged as uncertain rather than forced into a label.

**2. Narrative clustering** - Articles on the same topic are grouped by semantic similarity
using sentence embeddings. The clusters reveal how different outlets frame the same event -
and which outlets consistently end up in the same narrative group.

**3. Framing extraction** - For each article, a local LLM (Llama 3.2 3B via Ollama) extracts
three dimensions: the implied villain (who caused the problem), the implied victim (who is
being hurt), and the implied solution (what should be done). Aggregated across outlets, this
shows systematic editorial patterns that are invisible in any single article.

**4. Bias drift monitoring** - Using historical data from GDELT, the pipeline tracks whether
an outlet's political lean has shifted over months. Drift events are flagged when a bias
label's proportion deviates more than 20 percentage points from the outlet's baseline.

---

## Results

| Metric | Value |
|--------|-------|
| Total articles ingested | 472 |
| Outlets covered | BBC, Fox News, The Guardian, NPR, Al Jazeera, ABC News + NewsAPI sources |
| Topics analysed | Ukraine war, US economy, climate change |
| Bias classifier macro F1 | 0.613 (left: 0.71, right: 0.59, centre: 0.54) |
| Confidence threshold | 0.85 (empirically derived - yields ~70% accuracy on trusted subset) |
| Trusted predictions | 125 of 472 articles (26%) |
| Articles with framing extracted | 400 (85%) |
| Framing parse failures | 0 |
| Drift events detected | 6 (BBC Feb-Mar 2026, Guardian May 2026) |

**Bias distribution across corpus**: centre 46% · left 36% · right 18%

**Framing fill rates**: villain ~96% · victim ~89% · solution ~20%
Solution is structurally sparse - news rarely proposes explicit fixes.

**Clustering**: All three topics fell below the KMeans silhouette threshold of 0.3, meaning
the data does not form well-separated spherical clusters. DBSCAN was used for all topics.
US economy produced the clearest clusters: Cluster 0 = Trump-Xi trade summit coverage,
Cluster 1 = Federal Reserve/Warsh confirmation. Topic contamination is a known limitation
(see below).

---

## Tech stack

| Component | Choice | Why |
|-----------|--------|-----|
| Bias classifier | RoBERTa-base (fine-tuned) | Removes flawed NSP objective from BERT pre-training, trains on more data - consistently outperforms DistilBERT on bias classification tasks in the published literature |
| Training data | Qbias AllSides dataset (Haak & Schaer, 2023) | 21,747 full articles with article-level left/right/neutral labels from expert annotators - the correct dataset for this task |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) | Fast, small (80MB), 384-dim vectors that capture semantic meaning well for news text |
| Clustering | KMeans + DBSCAN fallback | Silhouette score used to choose k; DBSCAN used when no k produces score above 0.3 |
| Framing extraction | Ollama + `llama3.2:3b` | Faster than Mistral 7B, fits in 4GB VRAM, produces more reliably parseable JSON for constrained structured output |
| Vector store | ChromaDB | Purpose-built for embeddings - adds semantic search without a server, sits alongside SQLite |
| Structured storage | SQLite | Single-user, no setup overhead, standard SQL queries |
| Historical data | GDELT Project | Free, open, global news archive - provides months of historical data for drift monitoring without waiting to collect it live |
| Ingestion | NewsAPI + RSS feeds + newspaper3k | NewsAPI for topic search, RSS for continuous outlet coverage, newspaper3k to scrape full article text from truncated NewsAPI summaries |
| Dashboard | Streamlit | Rapid interactive dashboard for an ML pipeline - appropriate tool for the use case |

---

## Project structure

```
newslens/
├── src/
│   ├── ingestion.py        # NewsAPI + RSS fetching and deduplication
│   ├── scraper.py          # newspaper3k full-text scraping
│   ├── db.py               # SQLite schema, insert, and query functions
│   ├── chroma_store.py     # ChromaDB vector storage and retrieval
│   ├── preprocess.py       # Text cleaning, outlet name normalisation
│   ├── classifier.py       # Fine-tuned bias classifier - load and predict
│   ├── classify_articles.py # Run classifier over all unclassified articles
│   ├── embeddings.py       # Sentence embedding generation
│   ├── clustering.py       # Narrative clustering + evaluation metrics
│   ├── framing.py          # Ollama framing extraction with JSON validation
│   ├── drift.py            # Bias drift calculation and monthly aggregation
│   └── gdelt_pull.py       # GDELT historical data pull and SQLite ingestion
├── training/
│   └── fine_tune.py        # RoBERTa fine-tuning script
├── notebooks/
│   ├── 01_ingestion_validation.ipynb
│   ├── 02_classifier_eval.ipynb
│   ├── 03_narrative_clustering.ipynb
│   ├── 04_framing_extraction.ipynb
│   └── 05_drift_monitoring.ipynb
├── app/
│   ├── main.py             # Streamlit entry point + navigation
│   └── views/              # One file per dashboard page
├── data/
│   ├── allsides/           # Qbias fine-tuning dataset
│   └── gdelt/              # GDELT historical data CSVs
└── models/
    └── bias_classifier/    # Saved RoBERTa checkpoint
```

---

## Setup

**Prerequisites**: Python 3.11, Ollama installed (for framing extraction)

```bash
# Clone and install dependencies
git clone https://github.com/skanda-2003/news-lens.git
cd news-lens
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Pull the Llama model for framing extraction
ollama pull llama3.2:3b

# Add your NewsAPI key
echo "NEWSAPI_KEY=your_key_here" > .env
```

Get a free NewsAPI key at [newsapi.org](https://newsapi.org).

---

## Running the pipeline

Each step reads from and writes to the same SQLite database, so they can be run independently.

```bash
# 1. Ingest articles (NewsAPI + RSS feeds)
jupyter notebook notebooks/01_ingestion_validation.ipynb

# 2. Fine-tune the bias classifier (skip if using the saved checkpoint)
python training/fine_tune.py

# 3. Generate embeddings and run bias classification
python src/embeddings.py
python src/classify_articles.py

# 4. Run framing extraction (requires Ollama running)
python src/framing.py

# 5. Launch the dashboard
streamlit run app/main.py
```

---

## Known limitations

**Classifier trained on US political framing.** The Qbias AllSides dataset reflects the
US political spectrum. Applying these labels to non-US outlets (BBC, Al Jazeera) introduces
label noise - BBC "centre" in an AllSides context does not mean the same thing globally.
This is an acknowledged limitation, not a bug.

**Confidence threshold is empirically derived.** The threshold of 0.85 was derived from the
validation set of this specific training run. Only 26% of articles exceed it - the model is
overconfident overall (mean confidence 0.741 vs accuracy 0.643). Predictions below threshold
are shown in the dashboard as uncertain rather than dropped.

**Topic contamination in ingestion.** RSS feeds are not topic-filtered, so the climate change
query pulled in Eurovision and lifestyle articles. This carries through to framing and
clustering. Future work: add outlet-level keyword filtering at ingestion time.

**Framing extraction is LLM-generated and unverified.** The villain/victim/solution outputs
from Llama 3.2 3B are model interpretations, not ground-truth labels. No human validation
was performed at scale. Phrase fragmentation is a known issue ("Russia", "Russian government",
and "Vladimir Putin" are counted as separate villains).

**Drift monitoring uses sparse historical data.** GDELT averages 2-5 articles per outlet per
month for these outlets. Weekly rolling averages are meaningless at this density - monthly
aggregation is used instead. Drift events should be interpreted as directional signals, not
statistically rigorous findings.

**`newspaper3k` is not actively maintained.** Last PyPI release was 2019. It fails silently
on JS-rendered pages and some modern outlets. `trafilatura` is included as a fallback and is
the actively maintained alternative.

---

## References

Entman, R. M. (1993). Framing: Toward clarification of a fractured paradigm. *Journal of
Communication*, 43(4), 51-58.

Haak, F., & Schaer, P. (2023). Qbias: A dataset for media bias detection. *ACM Web Science
Conference 2023*. https://github.com/irgroup/Qbias

Spinde, T., et al. (2021). BABE - Bias annotations by experts. *EMNLP 2021 Findings*.
(Referenced as the rigorous academic benchmark for sentence-level bias annotation - not used
for training due to sentence-level granularity being poorly suited for full-article classification.)
