import trafilatura
from newspaper import Article


def scrape_full_text(url: str) -> tuple[str, str]:
    """
    Fetch the full body text of an article from its URL.

    Tries newspaper3k first. Falls back to trafilatura if that fails.
    Returns a tuple of (body_text, body_source) where body_source is one of:
      "scraped"      - full text retrieved successfully
      "summary_only" - both scrapers failed, caller should use the summary instead
    """
    # ── Attempt 1: newspaper3k ────────────────────────────────────────────────
    try:
        article = Article(url)

        # download() fetches the raw HTML from the URL
        article.download()

        # parse() extracts the article body from the HTML
        article.parse()

        # article.text is empty string if parsing found nothing useful
        if article.text and len(article.text.strip()) > 100:
            return article.text.strip(), "scraped"

    except Exception:
        # newspaper3k raises various errors on paywalled or JS-rendered pages
        # - don't crash, just fall through to trafilatura
        pass

    # ── Attempt 2: trafilatura (fallback) ─────────────────────────────────────
    try:
        # fetch_url downloads the raw HTML
        html = trafilatura.fetch_url(url)

        if html:
            # extract() strips boilerplate (ads, nav, footers) and returns article text
            body = trafilatura.extract(html)

            if body and len(body.strip()) > 100:
                return body.strip(), "scraped"

    except Exception:
        pass

    # ── Both failed ───────────────────────────────────────────────────────────
    # Return empty string - the caller (ingestion.py) will use the summary instead
    # and record body_source = "summary_only" in the database
    return "", "summary_only"