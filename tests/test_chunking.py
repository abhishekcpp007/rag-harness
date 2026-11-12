from rag.chunking import chunk_markdown

DOC = """# Handbook

Opening paragraph about nothing in particular.

## Settlement

Captured charges are swept nightly.

```yaml
sweep:
  india: "23:40 IST"
  singapore: "23:40 SGT"
```

More prose after the fence.

### Recovery

One two three four five six seven eight nine ten eleven twelve.
"""


def test_heading_path_travels_with_the_text():
    chunks = chunk_markdown("doc", DOC, words=200, overlap=0)
    paths = {c.heading_path for c in chunks}
    assert "Handbook > Settlement" in paths
    assert "Handbook > Settlement > Recovery" in paths
    rec = next(c for c in chunks if c.heading_path.endswith("Recovery"))
    assert rec.embed_text.startswith("Handbook > Settlement > Recovery :: ")


def test_a_code_fence_is_never_split():
    chunks = chunk_markdown("doc", DOC, words=6, overlap=0)
    for c in chunks:
        assert c.text.count("```") % 2 == 0, c.text
    whole = next(c for c in chunks if "23:40 IST" in c.text)
    assert "23:40 SGT" in whole.text


def test_a_chunk_never_spans_two_sections():
    chunks = chunk_markdown("doc", DOC, words=500, overlap=0)
    assert len({c.heading_path for c in chunks}) == len(chunks)


def test_overlap_carries_the_tail_forward():
    body = "# T\n\n" + " ".join(f"w{i}" for i in range(60)) + "\n\nsecond paragraph here\n"
    chunks = chunk_markdown("doc", body, words=20, overlap=5)
    assert len(chunks) >= 2
    assert chunks[1].text.startswith("w55 w56 w57 w58 w59")


def test_no_empty_chunks():
    chunks = chunk_markdown("doc", "# A\n\n\n\n## B\n\n\n", words=50, overlap=5)
    assert all(c.text.strip() for c in chunks)
