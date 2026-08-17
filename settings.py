import os
import tempfile

# Define your API key: it can only be alphanumeric characters + '-'; nothing else (i.e. [a-zA-Z-])
# Auth is only enforced if this is non-empty. Every request must then send a header:
#   Authorization: <this value>
API_KEY = ''    # e.g. 'RemovePII-Auth-Key your-api-key-here'

# You can either access the server locally via http://localhost:8142/ or from another machine
# on the network by using the IP address of the machine hosting the docker container
# (e.g. http://192.168.69.156:8142/) AND you can use a different port if you want, but you
# might need to open the port in your Firewall.
# Deliberately not 8000 (the port used by the sibling NLLB translator container), so both
# containers can run side by side on the same machine without a port clash.
PORT = 8142        # e.g. 8142

# The Hugging Face Hub repo id for the PII token-classification model.
MODEL_NAME = 'openai/privacy-filter'

# You can have the model use a GPU if your computer has one.
# one of  -1    # use cpu
#         0     # use gpu (you can replace 0, if you have multiple GPUs and want to use a different one)
# The model is small (1.5B total / 50M active params), so CPU is a reasonable default.
# Also, if you change this to use a GPU, then the docker run command will need to be changed
# to include the --gpus "device=0" option (or whatever GPU index you want to use).
DEVICE = -1

# Fixed placeholder used for EVERY redacted span, regardless of PII category
# (private_person, private_email, private_phone, private_address, private_url,
# private_date, account_number, secret). No category name is shown in the output.
# ASCII on purpose: PDF redaction boxes are drawn with PyMuPDF's default Base-14 font,
# which has no glyph for block-element characters (e.g. '█') -- those render as
# '?' tofu in a PDF viewer. '#' is guaranteed to render correctly in both .txt and .pdf
# output.
PII_PLACEHOLDER = '########'

# Only these file extensions are accepted for upload.
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# Maximum accepted upload size, in bytes.
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB

# Where processed job output files are staged before being downloaded.
JOBS_DIR = os.path.join(tempfile.gettempdir(), 'removepii_jobs')

# Jobs that are created but never downloaded get swept (deleted) once they are this old.
# Downloaded jobs are deleted immediately after being served, regardless of this setting.
JOB_TTL_SECONDS = 60 * 60  # 1 hour

# Some PDFs have no real text layer at all -- e.g. a scanned/flattened form where the
# entire page is one embedded image (common with DocuSign-processed or scanned-then-
# re-saved documents). Any page that contains at least one image gets an OCR pass
# (via PyMuPDF's built-in Tesseract integration) merged with whatever real text is
# already on the page, so PII baked into an image can still be found and redacted.
# Requires the tesseract-ocr system package (installed in the Dockerfile).
ENABLE_OCR = True
OCR_LANGUAGE = 'eng'
OCR_DPI = 300

# The classifier is never run on more than this many characters at once -- a single
# large input (observed: a ~465KB/68k-word document, e.g. a multi-page contract
# exported to .txt) can crash the entire server process (SIGKILL, out of memory),
# almost certainly from attention memory scaling with sequence length. Long .txt
# uploads (and, defensively, any single PDF page whose text is this long) are split
# into chunks around this size before being sent to the classifier, each chunk's
# entity offsets corrected back to the original text afterward. Deliberately well
# under the model's advertised 128,000-token context length -- that figure is not a
# safe practical bound, per the crash above -- and roughly page-sized, matching what
# the PDF path already handles reliably one page at a time.
MAX_TEXT_CHUNK_CHARS = 3000

# Optional caller-supplied list of exact strings to redact in addition to whatever the
# model detects -- a guarantee, not a suggestion: useful for known PII values the model
# missed (e.g. a short state abbreviation) or anything you specifically want redacted
# regardless of model confidence. Matched case-insensitively as a literal substring
# (not a regex). Capped to keep worst-case cost (terms x document length) bounded.
MAX_CUSTOM_TERMS = 200
MAX_CUSTOM_TERM_LENGTH = 200
