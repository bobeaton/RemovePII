# RemovePII

Redacts PII from uploaded `.txt` or `.pdf` files using the
[openai/privacy-filter](https://huggingface.co/openai/privacy-filter) token-classification
model. Every detected item (names, emails, phone numbers, addresses, URLs, dates, account
numbers, credentials) is replaced with a fixed placeholder (`########`).

It is important that you double-check the redacted document before assuming that the model 
has successfully redacted all PII!

For PDFs, redaction is done directly on the PDF's content stream (via PyMuPDF), drawing a
black box with the placeholder over each detected item and truly removing the original text
-- not just covering it visually. You can request the redacted PDF back, or a plain-text
extraction of it.

PDF pages with no real text layer at all -- e.g. a scanned document, or a form that's been
flattened to a single image (common with DocuSign-processed applications) -- are OCR'd
first (via PyMuPDF's built-in Tesseract integration), so PII baked into an image can still
be found and redacted. Pages with real text plus a stamped image are handled too: OCR only
fills in the image regions, leaving existing real text alone.

## Usage: execute these commands in Powershell

```powershell
# (you can also run buildDocker.ps1, esp. if you have a HF_TOKEN and want to 
# speed up the download (though, it’s really small). 
# NOTE: either way, you must have Docker Desktop installed
PS> docker build -t removepii .
PS> docker run -p 8142:8142 removepii
```

Or, using docker-compose:

```powershell
PS> docker-compose up --build
```

Then in a web browser, navigate to http://localhost:8142/

## API

### `POST /api/v1/redact/`

`multipart/form-data` with:
- `file` -- a `.txt` or `.pdf` file (required)
- `outputFormat` -- `pdf` or `text` (optional; defaults to `pdf` for PDF input, `text` for
  text input; `.txt` input can only produce `text` output)

Returns JSON:
```json
{
  "jobId": "…",
  "downloadUrl": "/api/v1/download/…",
  "entityCounts": {"private_person": 2, "private_email": 1},
  "skippedPages": [],
  "ocrPages": [0]
}
```

`ocrPages` lists page numbers (0-indexed) where OCR contributed text (i.e. the page had at
least one image and OCR found more than the page's native text extraction alone).
`skippedPages` lists page numbers that still had no extractable text even after an OCR
attempt (e.g. a genuinely blank page, or Tesseract unavailable/erroring) and were therefore
left unredacted.

Example:
```powershell
curl -F "file=@sample.pdf" -F "outputFormat=text" http://localhost:8142/api/v1/redact/
```

### `GET /api/v1/download/<jobId>`

Streams the redacted file back with the correct filename and content type. **One-time-use**:
the file is deleted from the server immediately after being served, so a second request for
the same `jobId` returns `404`. Jobs that are never downloaded are automatically cleaned up
after `JOB_TTL_SECONDS` (see `settings.py`).

## Configuration

See `settings.py` for `API_KEY` (optional `Authorization` header auth), `PORT`, `DEVICE`
(CPU/GPU), `PII_PLACEHOLDER`, upload size limits, job cleanup timing, and `ENABLE_OCR` /
`OCR_LANGUAGE` / `OCR_DPI` (OCR fallback for image-only PDF pages -- can be turned off if
not needed; disabling it also skips the extra per-page image-detection check).

`HF_TOKEN` (optional) -- `openai/privacy-filter` is public, so this is never required. If
you have a Hugging Face token, set it in your shell environment before building/running and
it'll be picked up automatically (`buildDocker.ps1` and `docker-compose.yml` both forward
it); without one, everything still works, just with Hugging Face Hub's unauthenticated rate
limits and a one-line "unauthenticated requests" warning logged at startup.

## Notes / limitations

- OCR is slower and less accurate than a real text layer -- expect noticeably longer
  processing time on scanned/image-heavy PDFs, and treat OCR'd redactions with a bit more
  scrutiny than native-text ones.
- Multi-column layouts or tables may not reconstruct perfect reading order, which can affect
  redaction accuracy on complex PDFs (native text or OCR'd).
- The model has documented recall gaps -- it can miss uncommon names, abbreviations (e.g. a
  state code inside an otherwise-detected address), or PII in unfamiliar formatting. Per its
  own documentation, this is "a redaction and data minimization aid, not an anonymization,
  compliance, or a safety guarantee" -- review output for high-sensitivity use cases.
- Large `.txt` uploads (and, rarely, an unusually text-dense single PDF page) are split into
  ~`MAX_TEXT_CHUNK_CHARS`-sized chunks before being sent to the model, one chunk at a time --
  a single very large input (a real ~465KB/68k-word document) was found to crash the entire
  server process (out of memory), so this is a hard requirement, not a tuning knob to relax
  casually. One side effect: the same PII string repeated verbatim at different points in a
  long document is not guaranteed to be redacted at every occurrence, since each occurrence
  can land in a differently-shaped chunk with different surrounding context, and the model's
  confidence isn't perfectly stable across that variation. Expect noticeably longer
  processing time on large documents (roughly proportional to length: tens of minutes for a
  ~100-page document on CPU) -- this trades speed for not crashing.
- **The server is single-threaded and has no worker pool** (`gevent.pywsgi.WSGIServer`
  running everything cooperatively on one OS thread), and redaction is CPU-bound work that
  never yields control back. Confirmed directly: while one redaction is in flight, the
  server is completely unresponsive to *anything else* -- including its own homepage -- for
  the entire duration, which can now be many minutes for a large document (see above). One
  big upload effectively locks out every other user (and the uploader's own browser) until
  it finishes. Fine for genuinely single-user/local use; a real fix (a threaded/worker-pool
  server, with the existing job-registry lock re-verified for thread-safety under real OS
  threads) would be needed before this could serve more than one person at a time.
