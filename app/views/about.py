import streamlit as st


def page_about():
    st.header("About NewsLens")

    st.markdown("""
Indian news outlets don't just report events - they frame them. Republic TV and The Wire can
cover the same protest and produce articles so different they read like separate realities.
NewsLens makes this visible computationally: it ingests articles from Indian English-language
outlets, classifies their political lean, clusters them into narrative groups, and extracts
the implied villain, victim, and solution from each piece.

The core idea comes from Robert Entman's 1993 framing theory: news frames define problems,
diagnose causes, make moral judgements, and suggest remedies. I built this project to see
whether those framing differences are detectable at scale without human annotation - and in
the Indian context, where ownership concentration and editorial alignment are unusually visible,
they are.
""")

    st.divider()

    st.subheader("Why Indian news specifically")
    st.markdown("""
The Indian media landscape has documented ownership concentration - Adani Group acquired NDTV
in 2022, and several large outlets have clear and publicly known editorial alignments. This
makes framing differences more measurable than in markets with stronger press independence:
the signal is large enough to detect with a relatively small dataset and a modest classifier.
Press freedom assessments by Reporters Without Borders (India ranked 159/180 in 2024) provide
external validation for the outlet-level bias assignments I use as training labels.
""")

    st.divider()

    st.subheader("Framing theory (Entman 1993)")
    st.markdown("""
Entman defines four framing functions: problem definition, causal interpretation, moral
evaluation, and treatment recommendation. NewsLens extracts three of them computationally:

- **Villain** - causal interpretation: who or what is blamed for the problem?
- **Victim** - moral evaluation: who is harmed and deserves sympathy?
- **Solution** - treatment recommendation: what should be done?

Extraction is done by a local Llama 3.2 3B model via Ollama, prompted per article. The
outputs are not human-verified at scale - treat them as indicative, not definitive.
""")

    st.divider()

    st.subheader("Pipeline")
    st.code("""
RSS / NewsAPI / GDELT
        |
        v
  Article ingestion
  (newspaper3k scraping)
        |
        v
  SQLite  +  ChromaDB
        |
        v
  RoBERTa classifier
  (bjp_aligned / opposition_aligned / neutral)
        |
        v
  Sentence-Transformers embeddings
  (all-MiniLM-L6-v2, 384-dim)
        |
        v
  KMeans / DBSCAN clustering
        |
        v
  Ollama framing extraction
  (llama3.2:3b - villain / victim / solution)
        |
        v
  Bias drift monitoring
  (GDELT historical, monthly aggregation)
        |
        v
  Streamlit dashboard
""", language="text")

    st.divider()

    st.subheader("Bias labels")
    st.markdown("""
Articles are classified into three labels:

| Label | Meaning | Example outlets |
|-------|---------|-----------------|
| `bjp_aligned` | Coverage that frames BJP/government actions positively or frames opposition negatively | Republic TV, Zee News |
| `opposition_aligned` | Coverage that frames opposition positively or government actions critically | The Wire, NDTV (historically) |
| `neutral` | Relatively balanced or factual reporting | The Hindu, Indian Express |

Labels are assigned using **distant supervision**: I use outlet identity as a weak label during
training. If Republic TV published it, it's marked `bjp_aligned`. If The Wire published it,
it's marked `opposition_aligned`. This is not expert annotation - outlet leanings are based on
documented press freedom assessments and editorial ownership records, not my own judgement.
""")

    st.divider()

    st.subheader("Limitations")
    st.markdown("""
1. **Distant supervision** - training labels are outlet-level, not article-level. A single
   article from Republic TV may be perfectly neutral; the model treats it as bjp_aligned anyway.
   This is a known limitation of the approach, not a bug.

2. **English only** - the classifier was trained on English Indian news and does not generalise
   to Hindi, Tamil, Marathi, or other regional language outlets. Outlets like Aaj Tak and
   Dainik Bhaskar are excluded for this reason.

3. **Framing extraction is LLM-generated** - Ollama outputs are not verified at scale. Single-word
   outputs are filtered out, but multi-word hallucinations are possible.

4. **Sparse GDELT data for Indian outlets** - drift monitoring requires historical monthly data.
   Most Indian outlets average 2-10 articles per month in GDELT, which is too sparse for
   statistically robust drift detection. Results should be interpreted cautiously.

5. **Topic contamination** - RSS feeds are not topic-filtered. An article from a political
   outlet's RSS feed may be about cricket. The bias label applies to the outlet, not the topic.
""")

    st.divider()

    st.subheader("References")
    st.markdown("""
- Entman, R. M. (1993). Framing: Toward clarification of a fractured paradigm.
  *Journal of Communication*, 43(4), 51-58.
- Haak, B. & Schaer, P. (2023). Automated news framing detection.
- Reporters Without Borders. (2024). *India - RSF Press Freedom Index*.
  [rsf.org/en/country/india](https://rsf.org/en/country/india)
""")
