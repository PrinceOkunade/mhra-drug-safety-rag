"""The cleaner must keep the article body and drop all gov.uk chrome."""
import pytest

from src.ingest.clean import _extract, _verify_clean

SAMPLE_HTML = """<html><head><title>x</title></head><body>
<div id="global-cookie-message">Cookies on GOV.UK. We use some essential cookies.</div>
<nav>Skip to main content</nav>
<h1 class="gem-c-title__text">Codeine linctus reclassification</h1>
<p class="gem-c-lead-paragraph">Codeine linctus is being reclassified to a \
prescription-only medicine.</p>
<div data-module="govspeak">
  <h2>Advice for healthcare professionals:</h2>
  <p>Codeine linctus will be reclassified to a POM from a P medicine.</p>
  <h2>Advice for patients:</h2>
  <p>Speak to your pharmacist if you currently use codeine linctus.</p>
</div>
<div class="footer">Is this page useful?</div>
</body></html>"""


def test_extract_keeps_body_drops_chrome():
    title, summary, body = _extract(SAMPLE_HTML)
    assert "Codeine linctus reclassification" in title
    assert "prescription-only" in summary
    assert "Advice for healthcare professionals" in body
    assert "Advice for patients" in body
    combined = "\n".join([title, summary, body]).lower()
    assert "cookies on gov.uk" not in combined
    assert "is this page useful" not in combined


def test_extract_output_passes_verification():
    title, summary, body = _extract(SAMPLE_HTML)
    # Should not raise.
    _verify_clean("\n".join([title, summary, body]), "codeine")


def test_verify_clean_raises_on_boilerplate():
    with pytest.raises(AssertionError):
        _verify_clean("some body text ... Cookies on GOV.UK ... more", "slug")
