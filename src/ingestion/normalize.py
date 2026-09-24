"""Text normalization for mixed Uzbek / Russian / English documents."""
import re
import unicodedata


# Uzbek Cyrillic → Latin transliteration table (official Uzbek Latin alphabet)
# Order matters: multi-char sequences MUST come before single chars.
UZ_CYRL_TO_LATN = [
    # Multi-character first
    ("Ш", "Sh"), ("ш", "sh"),
    ("Ч", "Ch"), ("ч", "ch"),
    ("Ю", "Yu"), ("ю", "yu"),
    ("Я", "Ya"), ("я", "ya"),
    ("Ё", "Yo"), ("ё", "yo"),
    ("Ў", "Oʻ"), ("ў", "oʻ"),
    ("Ғ", "Gʻ"), ("ғ", "gʻ"),
    ("Ҳ", "H"),  ("ҳ", "h"),
    ("Қ", "Q"),  ("қ", "q"),
    # Single-char mappings
    ("А", "A"), ("а", "a"), ("Б", "B"), ("б", "b"),
    ("В", "V"), ("в", "v"), ("Г", "G"), ("г", "g"),
    ("Д", "D"), ("д", "d"), ("Е", "E"), ("е", "e"),
    ("Ж", "J"), ("ж", "j"), ("З", "Z"), ("з", "z"),
    ("И", "I"), ("и", "i"), ("Й", "Y"), ("й", "y"),
    ("К", "K"), ("к", "k"), ("Л", "L"), ("л", "l"),
    ("М", "M"), ("м", "m"), ("Н", "N"), ("н", "n"),
    ("О", "O"), ("о", "o"), ("П", "P"), ("п", "p"),
    ("Р", "R"), ("р", "r"), ("С", "S"), ("с", "s"),
    ("Т", "T"), ("т", "t"), ("У", "U"), ("у", "u"),
    ("Ф", "F"), ("ф", "f"), ("Х", "X"), ("х", "x"),
    ("Ц", "S"), ("ц", "s"), ("Ъ", "ʼ"), ("ъ", "ʼ"),
    ("Ы", "I"), ("ы", "i"), ("Ь", ""),  ("ь", ""),
    ("Э", "E"), ("э", "e"),
]

# Cyrillic characters that belong to Russian (not present in Uzbek Latin table map above)
RU_ONLY_CYRL = set("ЩщЫыЪъЭэ")


def detect_script(text: str, sample_size: int = 500) -> str:
    """Cheap heuristic: which script dominates?
    Returns one of: 'uz-latn', 'uz-cyrl', 'ru', 'en', 'mixed'.
    """
    sample = text[:sample_size]
    cyrl = sum(1 for c in sample if "\u0400" <= c <= "\u04FF")
    latn = sum(1 for c in sample if "A" <= c <= "z")
    ru_markers = sum(1 for c in sample if c in RU_ONLY_CYRL)

    if cyrl == 0 and latn > 0:
        return "en" if not _has_uzbek_latin_markers(sample) else "uz-latn"
    if latn == 0 and cyrl > 0:
        # Cyrillic-only: distinguish RU vs UZ-CYRL by RU-only letter frequency
        return "ru" if ru_markers > cyrl * 0.02 else "uz-cyrl"
    if cyrl > 0 and latn > 0:
        return "mixed"
    return "mixed"


def _has_uzbek_latin_markers(text: str) -> bool:
    """Uzbek Latin uses distinctive digraphs and apostrophe-letters."""
    markers = ("oʻ", "gʻ", "sh", "ch", "Oʻ", "Gʻ", "Sh", "Ch")
    return any(m in text for m in markers)


def transliterate_uz_cyrl_to_latn(text: str) -> str:
    """Convert Uzbek Cyrillic to official Uzbek Latin."""
    for cyrl, latn in UZ_CYRL_TO_LATN:
        text = text.replace(cyrl, latn)
    return text


def clean_whitespace(text: str) -> str:
    """Collapse runs of whitespace, fix common PDF extraction artifacts."""
    # Normalize unicode (NFC = canonical composition; keeps Uzbek Latin ʻ intact)
    text = unicodedata.normalize("NFC", text)
    # Kill zero-width and BOM chars
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    # Fix broken hyphenation at line ends (common in PDF): "regu-\nlation" → "regulation"
    text = re.sub(r"-\s*\n\s*", "", text)
    # Collapse multiple spaces/tabs (but preserve newlines for structure)
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines to 2 (keep paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize(text: str, target_script: str = "latin") -> tuple[str, str]:
    """Full normalization pipeline.

    Args:
        text: Raw extracted text.
        target_script: 'latin' (default) transliterates Uzbek Cyrillic to Latin
                       for consistent indexing. 'preserve' keeps original scripts.

    Returns:
        (normalized_text, detected_language_hint)
    """
    text = clean_whitespace(text)
    lang = detect_script(text)

    if target_script == "latin" and lang == "uz-cyrl":
        text = transliterate_uz_cyrl_to_latn(text)
        lang = "uz-latn"

    return text, lang