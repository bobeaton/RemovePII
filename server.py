import os
import time
import uuid
import threading

import fitz  # PyMuPDF

from flask import Flask, request, jsonify, render_template, Response
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from settings import (MODEL_NAME, PORT, API_KEY, DEVICE, PII_PLACEHOLDER,
                       ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH, JOBS_DIR, JOB_TTL_SECONDS,
                       ENABLE_OCR, OCR_LANGUAGE, OCR_DPI, MAX_TEXT_CHUNK_CHARS)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(JOBS_DIR, exist_ok=True)

checkpoint = MODEL_NAME
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
# trust_remote_code=True: transformers doesn't (yet) recognize this model's architecture
# natively, so it ships its own modeling code -- see the note in import_model.py.
# HF_TOKEN (optional, see Dockerfile/import_model.py) is read again here because this
# from_pretrained() call runs fresh at every container startup -- unlike import_model.py,
# which only runs once at build time -- so it needs its own token pass-through to avoid
# the same "unauthenticated requests" warning/rate limit at runtime.
hf_token = os.environ.get("HF_TOKEN")
model_kwargs = {"token": hf_token} if hf_token else {}
model = AutoModelForTokenClassification.from_pretrained(checkpoint, trust_remote_code=True, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, **model_kwargs)

# Built once at startup (unlike a naive per-request pipeline() call, which would reload
# a fresh pipeline object on every single request).
redactor = pipeline('token-classification', model=model, tokenizer=tokenizer,
                     aggregation_strategy='simple', device=DEVICE)

# In-memory job registry: job_id -> {file_path, filename, mimetype, created_at}
JOBS = {}
JOBS_LOCK = threading.Lock()


def IsNullOrEmpty(s):
    return s is None or s == ''


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def sweep_stale_jobs():
    """Lazily delete jobs that were created but never downloaded, once they're older
    than JOB_TTL_SECONDS. Runs on every request instead of via a background
    thread/scheduler -- cheap (a dict scan) and avoids an extra dependency."""
    now = time.time()
    with JOBS_LOCK:
        stale = [jid for jid, j in JOBS.items() if now - j['created_at'] > JOB_TTL_SECONDS]
        for jid in stale:
            job = JOBS.pop(jid)
            try:
                os.remove(job['file_path'])
            except FileNotFoundError:
                pass


def merge_adjacent_entities(entities, text=None):
    """The model's raw token-classification output frequently splits one PII span into
    several word/sub-word-level fragments that share the same entity_group and are
    exactly contiguous or overlapping (e.g. ' John' + ' Smith', '000123456' + '789')
    instead of one coherent span -- aggregation_strategy='simple' doesn't merge these
    for this model's label scheme. Merge same-label touching/overlapping entities into
    a single span before counting or redacting, so downstream code sees one entity per
    real PII item rather than several fragments of it.

    If `text` is given, also bridge a whitespace-only gap between two same-label
    entities (e.g. the '\\n' our own build_page_text_and_wordmap() inserts between a
    PDF address's wrapped lines) -- that's still one logical PII item, not two."""
    entities = sorted(entities, key=lambda e: e['start'])
    merged = []
    for e in entities:
        if merged and merged[-1]['entity_group'] == e['entity_group'] and e['start'] <= merged[-1]['end']:
            merged[-1]['end'] = max(merged[-1]['end'], e['end'])
        elif (merged and text is not None and merged[-1]['entity_group'] == e['entity_group']
              and e['start'] > merged[-1]['end']
              and text[merged[-1]['end']:e['start']].strip() == ''):
            merged[-1]['end'] = max(merged[-1]['end'], e['end'])
        else:
            merged.append({'entity_group': e['entity_group'], 'start': e['start'], 'end': e['end']})
    return merged


def splice_placeholder(text, entities):
    """Splice PII_PLACEHOLDER into text at each (already merged, start-sorted) entity
    span. Shared by plain-text redaction and by the PDF path's text-output derivation,
    so both produce output the same way regardless of whether the PDF page's text came
    from its native text layer or from OCR."""
    parts = []
    cursor = 0
    for e in entities:
        start, end = e['start'], e['end']
        if start < cursor:
            if end <= cursor:
                continue  # fully inside an already-redacted span
            start = cursor  # clamp so we don't re-splice into already-consumed text
        parts.append(text[cursor:start])
        parts.append(PII_PLACEHOLDER)
        cursor = end
    parts.append(text[cursor:])
    return ''.join(parts)


def chunk_text(text, max_chars=MAX_TEXT_CHUNK_CHARS):
    """Split text into (chunk, offset) pairs of at most max_chars, breaking at the last
    newline within the window where possible (so a PII item is rarely split exactly at
    a chunk boundary) rather than mid-word. offset is the chunk's starting character
    position in the original text, for mapping entity spans back."""
    if len(text) <= max_chars:
        return [(text, 0)]
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            break_at = text.rfind('\n', start, end)
            end = break_at + 1 if break_at > start else end
        chunks.append((text[start:end], start))
        start = end
    return chunks


def run_redactor(text):
    """Run the PII classifier on `text`, returning a flat list of raw (unmerged)
    entities in `text`'s own coordinate space. Chunks first if `text` is long: a single
    ~465KB/68k-word document was observed to crash the whole server process (SIGKILL,
    out of memory) when sent to the classifier in one call -- likely attention memory
    scaling with sequence length -- so anything over MAX_TEXT_CHUNK_CHARS is split into
    page-sized pieces first, batched in one classifier call, with each chunk's entity
    offsets corrected back into `text`'s coordinates."""
    chunks = [(c, off) for c, off in chunk_text(text) if c.strip()]
    if not chunks:
        return []
    batch = redactor([c for c, off in chunks])
    entities = []
    for (chunk, offset), result in zip(chunks, batch):
        for e in result:
            entities.append({
                'entity_group': e['entity_group'],
                'start': e['start'] + offset,
                'end': e['end'] + offset,
            })
    return entities


def redact_text(text):
    """Run the PII classifier on a plain text string and splice in PII_PLACEHOLDER for
    every detected entity span. Returns (redacted_text, entity_counts) where
    entity_counts maps entity_group -> number of occurrences (for logging/feedback only
    -- never includes the actual PII values)."""
    entities = merge_adjacent_entities(run_redactor(text), text)
    counts = {}
    for e in entities:
        counts[e['entity_group']] = counts.get(e['entity_group'], 0) + 1
    return splice_placeholder(text, entities), counts


def cluster_visual_lines(words):
    """Group words into visual lines, starting from PyMuPDF/Tesseract's own
    (block_no, line_no) groups and then merging ADJACENT (in reading order) groups
    that are clearly a continuation of the same line -- high vertical (y0, y1) bbox
    overlap AND a small horizontal gap consistent with normal word spacing.

    This targeted merge exists because that (block_no, line_no) numbering can
    incorrectly split one visual line into two internal "lines" -- observed on a real
    scanned form where OCR assigned a word at the *identical* y-coordinate as its
    line-mate a different line_no (a grid/form-layout quirk), which broke a street
    address's continuity and left part of it undetected/unredacted.

    Merging is deliberately conservative (small horizontal gap only) rather than a
    blanket "same y-range = same line": a form row often has several unrelated fields
    at the same y-coordinate but in far-apart columns (e.g. date of birth, phone
    number, and a bank of checkboxes all on one visual row) -- merging those into one
    reconstructed line would scramble unrelated context together and was observed to
    make the model miss things it otherwise found. A large horizontal gap is treated
    as a genuinely different field, not a line-numbering artifact, and stays separate.

    Returns a list of lines (reading order), each a list of words (left-to-right) in
    the original (x0, y0, x1, y1, word, block_no, line_no, word_no) tuple shape."""
    groups = {}
    order = []
    for w in words:
        key = (w[5], w[6])
        groups.setdefault(key, []).append(w)
        if key not in order:
            order.append(key)

    raw_lines = []
    for key in order:
        ws = sorted(groups[key], key=lambda w: w[0])
        raw_lines.append({
            'y0': min(w[1] for w in ws), 'y1': max(w[3] for w in ws),
            'x0': ws[0][0], 'x1': max(w[2] for w in ws),
            'words': ws,
        })
    raw_lines.sort(key=lambda l: (l['y0'], l['x0']))

    merged = []
    for line in raw_lines:
        if merged:
            prev = merged[-1]
            overlap = min(prev['y1'], line['y1']) - max(prev['y0'], line['y0'])
            height = min(line['y1'] - line['y0'], prev['y1'] - prev['y0'])
            gap = line['x0'] - prev['x1']
            if height > 0 and overlap / height > 0.5 and 0 <= gap < height * 3:
                prev['words'].extend(line['words'])
                prev['y0'] = min(prev['y0'], line['y0'])
                prev['y1'] = max(prev['y1'], line['y1'])
                prev['x1'] = max(prev['x1'], line['x1'])
                continue
        merged.append(dict(line))

    for line in merged:
        line['words'].sort(key=lambda w: w[0])
    return [line['words'] for line in merged]


def build_page_text_and_wordmap(page):
    """Reconstruct a page's plain text from PyMuPDF word-level extraction, along with an
    exact map of each word's [start, end) character offset in that reconstructed string
    to its bounding box and a synthetic (line_idx, word_idx) position from
    cluster_visual_lines(). This lets us send plain text (with known character offsets)
    to the token-classification model, then map its answers back to page coordinates.

    Some PDFs have PII baked into an image rather than real text -- e.g. a scanned or
    DocuSign-flattened form where the whole page (or a stamped block on it) is one
    embedded picture. If the page contains any image, we OCR it (via PyMuPDF's
    Tesseract integration) and merge that with whatever real text is already on the
    page ("full=False" merge mode -- it only OCRs the image regions, leaving existing
    real text as-is), so PII hidden in an image can still be found. If OCR finds
    nothing beyond what native extraction already had (or Tesseract isn't available/
    errors out), we fall back to the native-only result rather than losing data.
    Returns (text, word_spans, used_ocr)."""
    words = page.get_text("words")
    used_ocr = False
    if ENABLE_OCR and page.get_images():
        try:
            ocr_textpage = page.get_textpage_ocr(language=OCR_LANGUAGE, dpi=OCR_DPI, full=False)
            ocr_words = page.get_text("words", textpage=ocr_textpage)
            if len(ocr_words) > len(words):
                words = ocr_words
                used_ocr = True
        except Exception:
            app.logger.exception(f"OCR failed for page {page.number}; continuing without it")

    parts = []
    spans = []
    cursor = 0
    for line_idx, line in enumerate(cluster_visual_lines(words)):
        if line_idx > 0:
            parts.append('\n')
            cursor += 1
        for word_idx, (x0, y0, x1, y1, word, _block_no, _line_no, _word_no) in enumerate(line):
            if word_idx > 0:
                parts.append(' ')
                cursor += 1
            start = cursor
            parts.append(word)
            cursor += len(word)
            spans.append((start, cursor, x0, y0, x1, y1, line_idx, word_idx))
    return ''.join(parts), spans, used_ocr


def map_entity_to_rects(entity, word_spans):
    """Map one entity's [start, end) character span back to one or more page rects.
    Words are matched by overlap (not strict containment) so a boundary-clipped entity
    span never leaves a PII fragment un-redacted. Overlapping words are grouped by
    visual line index (word_spans[*][6], from cluster_visual_lines()) so an entity that
    wraps across a line (e.g. a long address) produces one rect per line instead of one
    rect spanning unrelated text in between."""
    e_start, e_end = entity['start'], entity['end']
    overlapping = [w for w in word_spans if w[0] < e_end and w[1] > e_start]
    if not overlapping:
        return []
    groups = {}
    order = []
    for w in overlapping:
        key = w[6]  # visual line index
        groups.setdefault(key, []).append(w)
        if key not in order:
            order.append(key)
    rects = []
    for key in order:
        g = groups[key]
        rect = fitz.Rect(g[0][2], g[0][3], g[0][4], g[0][5])
        for w in g[1:]:
            rect |= fitz.Rect(w[2], w[3], w[4], w[5])
        rects.append(rect)
    return rects


def redact_pdf(doc):
    """Redact every page of an open PyMuPDF document in place. Pages with no
    extractable text at all -- even after an OCR attempt -- are skipped entirely: not
    sent to the model, never redacted -- and reported back in skipped_pages. Pages
    where OCR contributed text are reported in ocr_pages. Returns
    (entity_counts, skipped_pages, ocr_pages, page_redacted_texts) where
    page_redacted_texts is a list of one redacted plain-text string per page (used for
    outputFormat=text, so that output is derived the same way regardless of whether a
    page's text came from its native text layer or from OCR)."""
    counts = {}
    skipped_pages = []
    ocr_pages = []
    page_texts = []
    page_wordmaps = []

    for page in doc:
        text, wordmap, used_ocr = build_page_text_and_wordmap(page)
        page_texts.append(text)
        page_wordmaps.append(wordmap)
        if used_ocr:
            ocr_pages.append(page.number)
        if not text.strip():
            skipped_pages.append(page.number)

    # run_redactor() chunks internally if a page's text is unusually long (see
    # MAX_TEXT_CHUNK_CHARS) -- defense in depth against the same crash .txt uploads hit,
    # in case a single PDF page ever has an enormous amount of text on it.
    results_by_page = {i: [] for i in range(len(page_texts))}
    for idx, text in enumerate(page_texts):
        if text.strip():
            results_by_page[idx] = merge_adjacent_entities(run_redactor(text), text)

    page_redacted_texts = list(page_texts)

    for page in doc:
        entities = results_by_page.get(page.number, [])
        page_redacted_texts[page.number] = splice_placeholder(page_texts[page.number], entities)
        if not entities:
            continue
        wordmap = page_wordmaps[page.number]
        seen = set()
        for entity in entities:
            counts[entity['entity_group']] = counts.get(entity['entity_group'], 0) + 1
            for rect in map_entity_to_rects(entity, wordmap):
                key = (round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1))
                if key in seen:
                    continue
                seen.add(key)
                page.add_redact_annot(rect, text=PII_PLACEHOLDER, fill=(0, 0, 0), text_color=(1, 1, 1))
        page.apply_redactions()

    return counts, skipped_pages, ocr_pages, page_redacted_texts


@app.before_request
def before_request_hook():
    sweep_stale_jobs()

    # Check if the API key is present in the request headers
    if not IsNullOrEmpty(API_KEY):
        if 'Authorization' not in request.headers or request.headers['Authorization'] != API_KEY:
            return jsonify({'error': 'Unauthorized'}), 401


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'Uploaded file exceeds the maximum allowed size'}), 413


@app.route('/')
def index():
    return render_template('index.html', MODEL_NAME=MODEL_NAME)


@app.route('/api/v1/redact/', methods=['POST'])
def redact():
    try:
        if 'file' not in request.files:
            return jsonify({'error': "Missing 'file' part in form-data"}), 400

        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = secure_filename(f.filename)
        if not allowed_file(filename):
            return jsonify({'error': 'Unsupported file type; allowed: txt, pdf'}), 400
        ext = filename.rsplit('.', 1)[1].lower()

        output_format = request.form.get('outputFormat', 'pdf' if ext == 'pdf' else 'text')
        if output_format not in ('pdf', 'text'):
            return jsonify({'error': "outputFormat must be 'pdf' or 'text'"}), 400
        if ext == 'txt' and output_format == 'pdf':
            return jsonify({'error': 'outputFormat=pdf is not supported for .txt input'}), 400

        data = f.read()
        if len(data) == 0:
            return jsonify({'error': 'Uploaded file is empty'}), 400

        job_id = uuid.uuid4().hex
        skipped_pages = []
        ocr_pages = []

        if ext == 'txt':
            text = data.decode('utf-8', errors='replace')
            redacted_text, counts = redact_text(text)
            out_path = os.path.join(JOBS_DIR, f'{job_id}.txt')
            with open(out_path, 'w', encoding='utf-8') as fh:
                fh.write(redacted_text)
            mimetype, out_name = 'text/plain', f'redacted_{filename}'
        else:
            try:
                doc = fitz.open(stream=data, filetype='pdf')
            except Exception:
                return jsonify({'error': 'Could not read PDF file (it may be corrupt or unsupported)'}), 400

            if doc.needs_pass:
                doc.close()
                return jsonify({'error': 'PDF is password protected'}), 400
            if doc.page_count == 0:
                doc.close()
                return jsonify({'error': 'PDF has no pages'}), 400

            counts, skipped_pages, ocr_pages, page_redacted_texts = redact_pdf(doc)

            if output_format == 'pdf':
                out_path = os.path.join(JOBS_DIR, f'{job_id}.pdf')
                doc.save(out_path, garbage=4, deflate=True)
                mimetype, out_name = 'application/pdf', f'redacted_{filename}'
            else:
                # Built from page_redacted_texts (not a post-redaction page.get_text()
                # re-extraction) so OCR'd pages' non-PII text is included too -- that
                # text only ever existed inside the image, never written back as real
                # page content, so re-extracting from the page itself would lose it.
                redacted_text = '\n\n'.join(page_redacted_texts)
                out_path = os.path.join(JOBS_DIR, f'{job_id}.txt')
                with open(out_path, 'w', encoding='utf-8') as fh:
                    fh.write(redacted_text)
                mimetype, out_name = 'text/plain', f'redacted_{os.path.splitext(filename)[0]}.txt'
            doc.close()

        with JOBS_LOCK:
            JOBS[job_id] = {
                'file_path': out_path,
                'filename': out_name,
                'mimetype': mimetype,
                'created_at': time.time(),
            }

        # Only log category counts and structural info -- never the actual PII values.
        app.logger.info(f"job {job_id}: type={ext} entity_counts={counts} "
                         f"skipped_pages={skipped_pages} ocr_pages={ocr_pages}")

        return jsonify({
            'jobId': job_id,
            'downloadUrl': f'/api/v1/download/{job_id}',
            'entityCounts': counts,
            'skippedPages': skipped_pages,
            'ocrPages': ocr_pages,
        })

    except HTTPException:
        # Let Flask/Werkzeug HTTP exceptions (e.g. RequestEntityTooLarge/413 from
        # MAX_CONTENT_LENGTH) reach their registered error handler instead of being
        # swallowed into a generic 500 below.
        raise
    except Exception:
        app.logger.exception('Unhandled error while processing redaction request')
        # Deliberately generic -- not str(e) -- since an exception message could
        # inadvertently echo a fragment of file content.
        return jsonify({'error': 'Internal server error while processing file'}), 500


@app.route('/api/v1/download/<job_id>', methods=['GET'])
def download(job_id):
    with JOBS_LOCK:
        job = JOBS.pop(job_id, None)  # pop = one-time claim; a concurrent second request 404s

    if job is None:
        return jsonify({'error': 'Job not found or expired'}), 404

    try:
        with open(job['file_path'], 'rb') as fh:
            data = fh.read()
    except FileNotFoundError:
        return jsonify({'error': 'Job file no longer available'}), 404
    finally:
        try:
            os.remove(job['file_path'])
        except FileNotFoundError:
            pass

    resp = Response(data, mimetype=job['mimetype'])
    resp.headers['Content-Disposition'] = f'attachment; filename="{job["filename"]}"'
    return resp


if __name__ == '__main__':
    # Debug/Development:
    # app.run(host="0.0.0.0", port=PORT, debug=True)
    # Production:
    http_server = WSGIServer(('', PORT), app)
    http_server.serve_forever()
