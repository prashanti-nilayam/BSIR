"""Answer parsing for GSM8K.

Two parsers:
- parse_gold: extracts the canonical answer from GSM8K's `#### N` format.
- parse_model_output: extracts the answer from a model response, preferring \\boxed{N}
  (Du et al.'s prompt format) with last-number fallback.

Numbers are returned as normalized strings; equality checks via parsed_floats_equal.
"""

from __future__ import annotations

import math
import re

# Gold answer in GSM8K is "<reasoning>\n#### <number>"
_GOLD_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")

# Model output: prefer \boxed{...} (last occurrence). Capture digits/decimal/comma/sign inside.
_BOXED_RE = re.compile(r"\\boxed\s*\{\s*([^{}]*?)\s*\}")

# Fallback: any trailing number in the output.
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _normalize_number_string(s: str) -> str | None:
    """Strip commas, currency, whitespace; return cleaned numeric string or None."""
    if s is None:
        return None
    cleaned = re.sub(r"[^\d.\-]", "", s)
    if not cleaned or cleaned in {"-", ".", "-."}:
        return None
    try:
        float(cleaned)
    except ValueError:
        return None
    return cleaned


def parse_gold(gsm8k_answer: str) -> str | None:
    """Extract the canonical answer from a GSM8K gold answer string."""
    match = _GOLD_RE.search(gsm8k_answer)
    if not match:
        return None
    return _normalize_number_string(match.group(1))


def parse_model_output(output: str) -> str | None:
    """Extract the model's final answer.

    Strategy: take the last `\\boxed{...}` and pull a number out of it.
    If no boxed expression, fall back to the last number in the output.
    Returns a normalized numeric string, or None if nothing parseable.
    """
    if not output:
        return None
    boxed = _BOXED_RE.findall(output)
    for candidate in reversed(boxed):
        normalized = _normalize_number_string(candidate)
        if normalized is not None:
            return normalized
    numbers = _NUMBER_RE.findall(output)
    if numbers:
        return _normalize_number_string(numbers[-1])
    return None


def answers_match(a: str | None, b: str | None) -> bool:
    """Compare two parsed numeric strings as floats."""
    if a is None or b is None:
        return False
    try:
        return float(a) == float(b)
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# MATH benchmark parsing and equality
# ---------------------------------------------------------------------------

# LaTeX spacing and decoration to strip during normalization.
_LATEX_SPACING = (r"\,", r"\!", r"\;", r"\:", r"\quad", r"\qquad", r"~")
_LATEX_WRAPPERS = (r"\left", r"\right", r"\displaystyle")
_BOXED_OPEN_RE = re.compile(r"\\boxed\s*\{")


def _extract_boxed_balanced(text: str) -> str | None:
    """Extract the content of the last \\boxed{...}, respecting brace nesting."""
    if not text:
        return None
    starts = [m.end() for m in _BOXED_OPEN_RE.finditer(text)]
    if not starts:
        return None
    start = starts[-1]
    depth = 1
    i = start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None


def parse_math_answer(text: str | None) -> str | None:
    """Extract the answer from a model output for MATH.

    Prefers the last \\boxed{...} (with balanced brace handling).
    Falls back to the trailing line as a guess.
    """
    if not text:
        return None
    boxed = _extract_boxed_balanced(text)
    if boxed is not None:
        return boxed.strip()
    return None


def _normalize_math_answer(s: str | None) -> str | None:
    """Normalize a MATH answer string for equality comparison.

    Strips spacing macros, wrappers (\\left/\\right/\\displaystyle), \\text{}/\\mathrm{},
    whitespace, surrounding $ signs, and trailing periods. Converts \\dfrac/\\tfrac to
    \\frac. Lowercases ASCII letters. This is intentionally permissive: it under-counts
    correctness slightly (no symbolic equivalence), which is fine for smoke signal where
    we mainly need to detect answer changes round-to-round.
    """
    if s is None:
        return None
    s = s.strip()
    s = s.replace("\\$", "")  # LaTeX-escaped dollar
    s = s.replace("$", "")
    s = s.replace("\\%", "")  # percent sign in LaTeX
    for sp in _LATEX_SPACING:
        s = s.replace(sp, "")
    for w in _LATEX_WRAPPERS:
        s = s.replace(w, "")
    # \dfrac, \tfrac -> \frac
    s = re.sub(r"\\(d|t)frac", r"\\frac", s)
    # Strip \text{...} and \mathrm{...} wrappers (one level).
    s = re.sub(r"\\(text|mathrm|mathbf|mathit)\s*\{([^{}]*)\}", r"\2", s)
    # Strip outer braces if they wrap the whole thing.
    while len(s) >= 2 and s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    s = re.sub(r"\s+", "", s)
    # Trailing period or comma sometimes left from prose.
    s = s.rstrip(".,")
    return s.lower() or None


# ---------------------------------------------------------------------------
# Multiple-choice parsing for benchmarks like GPQA-Diamond
# ---------------------------------------------------------------------------

# Prefer \boxed{X} where X is a single A/B/C/D letter. Fall back to "the answer
# is X" / "the correct answer is X" / "answer: X" patterns in the last 300 chars.
_MCQ_BOXED_RE = re.compile(r"\\boxed\s*\{\s*([A-D])\s*\}", re.IGNORECASE)
_MCQ_ANSWER_PHRASE_RE = re.compile(
    r"(?i)(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:|=)\s*\*?\*?\(?([A-D])\)?",
)
_MCQ_TRAILING_LETTER_RE = re.compile(
    r"(?i)\bfinal\s+answer\s*[:=]?\s*\*?\*?\(?([A-D])\)?",
)


def parse_mcq_answer(output: str | None) -> str | None:
    """Extract A/B/C/D from a model response.

    Order of preference:
      1. Last \boxed{X} containing a single letter
      2. "final answer: X" in last 400 chars
      3. "the answer is X" / "answer: X" in last 400 chars
      4. None
    """
    if not output:
        return None
    boxed = _MCQ_BOXED_RE.findall(output)
    if boxed:
        return boxed[-1].upper()
    tail = output[-400:]
    m = _MCQ_TRAILING_LETTER_RE.search(tail)
    if m:
        return m.group(1).upper()
    m = _MCQ_ANSWER_PHRASE_RE.search(tail)
    if m:
        return m.group(1).upper()
    return None


def mcq_answers_match(pred: str | None, gold: str | None) -> bool:
    """Compare A/B/C/D answers case-insensitively."""
    if pred is None or gold is None:
        return False
    return pred.strip().upper() == gold.strip().upper()


def _to_float_scientific(s: str | None) -> float | None:
    """Best-effort parse of a SciBench-style numeric answer to a float.

    Handles plain decimals, scientific notation (1.2e3, 1.2E3), LaTeX
    \\times 10^{n} / x10^n forms, leading +/-, and surrounding units/text by
    extracting the first signed number and an optional power-of-ten.
    Returns None if no number can be recovered.
    """
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    # Normalize common LaTeX/text scientific-notation forms to Python 'e'.
    t = t.replace("\\times", "x").replace("\\cdot", "x").replace("^", "")
    t = re.sub(r"\s+", "", t)
    t = t.replace(",", "")  # strip thousands separators (e.g. 6,760,000)
    # "x10{3}" / "x10{-3}" / "x103" -> "e3" / "e-3"
    t = re.sub(r"[xX]10\{?(-?\d+)\}?", r"e\1", t)
    # Find a mantissa and optional exponent.
    m = re.search(r"(-?\d+(?:\.\d+)?)(?:[eE](-?\d+))?", t)
    if not m:
        return None
    try:
        val = float(m.group(1))
        if m.group(2) is not None:
            val *= 10 ** int(m.group(2))
        return val
    except (ValueError, OverflowError):
        return None


def _sympy_eval(s: str) -> float | None:
    """Evaluate an expression-like answer (LaTeX fractions, products, powers,
    scientific notation) to a float via sympy. Returns None on failure."""
    try:
        import sympy
    except ImportError:
        return None
    e = s
    e = re.sub(r"\\(?:text|mathrm|mathbf|operatorname)\s*\{[^{}]*\}", "", e)
    e = e.replace("\\boxed", "")  # keep the boxed content; braces -> parens below
    for tok in ("\\left", "\\right", "\\!", "\\,", "\\ ", "$"):
        e = e.replace(tok, "")
    e = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", e)
    e = e.replace("\\times", "*").replace("\\cdot", "*")
    e = re.sub(r"\^\s*\{([^{}]*)\}", r"**(\1)", e)
    e = e.replace("^", "**").replace("{", "(").replace("}", ")").replace("\\", "")
    try:
        return float(sympy.sympify(e, evaluate=True).evalf())
    except Exception:
        return None


def _parse_sci_number(s: str | None) -> float | None:
    """Recover a float from a SciBench-style answer. Routes expression-like
    strings (fractions, products, LaTeX powers) through sympy first so e.g.
    \\frac{1}{2} -> 0.5 rather than being misread as 1."""
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    # Only route to sympy for genuine fractions/ratios. The fast path already
    # handles sci-notation (\times 10^n, e-notation) and strips unit exponents
    # (dm^3), so routing on '^' would mis-evaluate "1.20 dm^3" as 1.2 cubed.
    if re.search(r"\\frac", t) or re.search(r"\d\s*/\s*\d", t):
        v = _sympy_eval(t)
        if v is not None:
            return v
    return _to_float_scientific(t)


def scibench_answers_match(
    pred: str | None, gold: str | None, rel_tol: float = 1e-2, abs_tol: float = 1e-9
) -> bool:
    """Tolerance-based numeric match for SciBench science answers.

    Textbook answers are rounded decimals (often 3 sig figs) with units, so
    exact comparison is wrong. Grade numerically within a relative tolerance
    (default 1%), with a small absolute tolerance for answers near zero.
    Model answers may be fractions/expressions, so parsing routes through sympy.
    """
    if pred is None or gold is None:
        return False
    p = _parse_sci_number(pred)
    g = _parse_sci_number(gold)
    if p is None or g is None:
        return False
    return math.isclose(p, g, rel_tol=rel_tol, abs_tol=abs_tol)


def math_answers_match(pred: str | None, gold: str | None) -> bool:
    """Compare MATH predicted vs gold answers.

    Strategy:
      1. Normalize both strings (LaTeX spacing, wrappers, braces).
      2. If both parse as floats, compare numerically.
      3. Else string-compare the normalized forms.
    """
    if pred is None or gold is None:
        return False
    np_ = _normalize_math_answer(pred)
    ng_ = _normalize_math_answer(gold)
    if np_ is None or ng_ is None:
        return False
    try:
        return float(np_) == float(ng_)
    except ValueError:
        pass
    return np_ == ng_


# ---------------------------------------------------------------------------
# Free-text QA parsing / matching for MuSiQue-Ans (multi-hop reading comp).
#
# MuSiQue answers are short spans (1-2 tokens) with aliases. gold is stored as
# the primary answer plus aliases joined by ALIAS_DELIM (see musique_data.py).
# `musique_answers_match` is normalized exact-match (max over answer+aliases) and
# becomes the record's `correct` bool; `qa_token_f1` is used by the post-hoc scorer.
# ---------------------------------------------------------------------------

_ALIAS_DELIM = " ||| "  # must match musique_data.ALIAS_DELIM
_QA_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_QA_ANSWER_PHRASE_RE = re.compile(r"(?i)(?:final\s+)?answer\s*(?:is|:|=)\s*(.+)")


def _normalize_qa(s: str | None) -> str:
    """SQuAD/MuSiQue answer normalization: lowercase, drop punctuation, drop
    articles (a/an/the), collapse whitespace."""
    if not s:
        return ""
    s = s.lower()
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    s = _QA_ARTICLES_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_qa_answer(text: str | None) -> str | None:
    """Extract a free-text answer from a model output.

    Prefers the last \\boxed{...} (balanced), stripping any \\text{}/\\mathrm{}
    wrapper. Falls back to an "answer is/:" phrase, then the last non-empty line.
    """
    if not text:
        return None
    boxed = _extract_boxed_balanced(text)
    if boxed is not None:
        b = re.sub(r"\\(text|mathrm|mathbf|mathit)\s*\{([^{}]*)\}", r"\2", boxed).strip()
        b = b.strip("$ ").strip()
        if b:
            return b
    tail = text[-400:]
    m = None
    for m in _QA_ANSWER_PHRASE_RE.finditer(tail):
        pass
    if m is not None:
        cand = m.group(1).strip().splitlines()[0]
        cand = cand.lstrip(":=-").strip().strip("*").strip().strip('"').strip(".").strip()
        if cand:
            return cand
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line:
            return line
    return None


def musique_answers_match(pred: str | None, gold: str | None) -> bool:
    """Normalized exact-match against the primary answer OR any alias."""
    if pred is None or gold is None:
        return False
    pn = _normalize_qa(pred)
    if not pn:
        return False
    return any(pn == _normalize_qa(alias) for alias in gold.split(_ALIAS_DELIM))


def qa_token_f1(pred: str | None, gold: str | None) -> float:
    """Max token-level F1 of pred against the primary answer + aliases."""
    from collections import Counter

    pn = _normalize_qa(pred).split()
    best = 0.0
    for alias in (gold or "").split(_ALIAS_DELIM):
        gn = _normalize_qa(alias).split()
        if not pn and not gn:
            best = max(best, 1.0)
            continue
        if not pn or not gn:
            continue
        common = Counter(pn) & Counter(gn)
        n_same = sum(common.values())
        if n_same == 0:
            continue
        prec = n_same / len(pn)
        rec = n_same / len(gn)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best
