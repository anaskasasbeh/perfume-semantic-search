"""A small catalog of common fragrance notes.

Used for two things:
* the note filter in the UI (include / exclude), with English and Arabic labels;
* naming the clusters on the home page ("Vanilla · Tonka · Amber").

Matching uses whole-word regular expressions, so "rose" does not match
"rosemary" or "rosewood", and "orange" does not match "orange blossom".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache


@dataclass(frozen=True)
class Note:
    id: str
    en: str
    ar: str
    pattern: str
    family: str = ""


NOTE_CATALOG: tuple[Note, ...] = (
    # Sweet / amber
    Note("vanilla", "Vanilla", "فانيلا", r"\bvanill?as?\b|\bvanille\b", "amber"),
    Note("tonka", "Tonka bean", "حبوب التونكا", r"\btonka\b", "amber"),
    Note("amber", "Amber", "عنبر", r"\bamber\b", "amber"),
    Note("benzoin", "Benzoin", "جاوي", r"\bbenzoin\b", "amber"),
    Note("labdanum", "Labdanum", "لادن", r"\blabdanum\b|\bcistus\b", "amber"),
    Note("musk", "Musk", "مسك", r"\bmusks?\b|\bmusky\b", "fresh"),
    Note("honey", "Honey", "عسل", r"\bhoney\b", "amber"),
    Note("caramel", "Caramel", "كراميل", r"\bcaramel(?:ized)?\b", "amber"),
    Note("praline", "Praline", "برالين", r"\bpralines?\b", "amber"),
    Note("chocolate", "Chocolate", "شوكولاتة", r"\bchocolate\b|\bcacao\b|\bcocoa\b", "amber"),
    Note("coffee", "Coffee", "قهوة", r"\bcoffee\b|\bespresso\b", "amber"),
    Note("almond", "Almond", "لوز", r"\balmonds?\b", "amber"),
    Note("coconut", "Coconut", "جوز الهند", r"\bcoconut\b", "amber"),
    # Woods / resins / smoke
    Note("oud", "Oud", "عود", r"\b(?:oud|oudh|aoud|agarwood)\b", "woods"),
    Note("sandalwood", "Sandalwood", "صندل", r"\bsandalwood\b|\bsantal\b", "woods"),
    Note("cedar", "Cedar", "أرز", r"\bcedar(?:wood)?\b", "woods"),
    Note("vetiver", "Vetiver", "نجيل الهند (فيتيفر)", r"\bvetiver\b", "woods"),
    Note("patchouli", "Patchouli", "باتشولي", r"\bpatchouli\b", "woods"),
    Note("oakmoss", "Oakmoss", "طحلب البلوط", r"\boak ?moss\b", "woods"),
    Note("pine", "Pine & fir", "صنوبر", r"\bpine\b|\bfir\b", "woods"),
    Note("cypress", "Cypress", "سرو", r"\bcypress\b", "woods"),
    Note("incense", "Incense", "بخور", r"\bincense\b", "smoke"),
    Note("frankincense", "Frankincense", "لبان", r"\bfrankincense\b|\bolibanum\b", "smoke"),
    Note("myrrh", "Myrrh", "مُر", r"\bmyrrh\b", "smoke"),
    Note("leather", "Leather", "جلد", r"\bleather\b|\bsuede\b", "smoke"),
    Note("tobacco", "Tobacco", "تبغ", r"\btobacco\b", "smoke"),
    Note("birch", "Birch", "بتولا", r"\bbirch\b", "smoke"),
    # Spices
    Note("saffron", "Saffron", "زعفران", r"\bsaffron\b", "spice"),
    Note("cardamom", "Cardamom", "هيل", r"\bcardamom\b", "spice"),
    Note("cinnamon", "Cinnamon", "قرفة", r"\bcinnamon\b|\bcassia\b", "spice"),
    Note("clove", "Clove", "قرنفل", r"\bcloves?\b", "spice"),
    Note("nutmeg", "Nutmeg", "جوزة الطيب", r"\bnutmeg\b", "spice"),
    Note("pepper", "Pepper", "فلفل", r"\bpeppers?\b|\bpeppercorns?\b", "spice"),
    Note("ginger", "Ginger", "زنجبيل", r"\bginger\b", "spice"),
    Note("cumin", "Cumin", "كمون", r"\bcumin\b", "spice"),
    # Florals
    Note("rose", "Rose", "ورد", r"\broses?\b", "floral"),
    Note("jasmine", "Jasmine", "ياسمين", r"\bjasmin(?:e)?\b|\bsambac\b", "floral"),
    Note("tuberose", "Tuberose", "مسك الروم", r"\btuberoses?\b", "floral"),
    Note("orange_blossom", "Orange blossom", "زهر البرتقال", r"\borange (?:blossoms?|flowers?)\b", "floral"),
    Note("neroli", "Neroli", "نيرولي", r"\bneroli\b", "floral"),
    Note("iris", "Iris", "سوسن", r"\biris\b|\borris\b", "floral"),
    Note("violet", "Violet", "بنفسج", r"\bviolets?\b", "floral"),
    Note("ylang", "Ylang-ylang", "يلانغ يلانغ", r"\bylang\b", "floral"),
    Note("gardenia", "Gardenia", "غاردينيا", r"\bgardenias?\b", "floral"),
    Note("magnolia", "Magnolia", "ماغنوليا", r"\bmagnolias?\b", "floral"),
    Note("lily", "Lily", "زنبق", r"\blily\b|\blilies\b", "floral"),
    Note("peony", "Peony", "فاوانيا", r"\bpeon(?:y|ies)\b", "floral"),
    Note("geranium", "Geranium", "عطرة (جيرانيوم)", r"\bgeraniums?\b", "floral"),
    Note("heliotrope", "Heliotrope", "هليوتروب", r"\bheliotrope\b", "floral"),
    Note("lavender", "Lavender", "لافندر", r"\blavender\b|\blavandin\b", "floral"),
    # Citrus
    Note("bergamot", "Bergamot", "برغموت", r"\bbergamot\b", "citrus"),
    Note("lemon", "Lemon", "ليمون", r"\blemons?\b", "citrus"),
    Note("lime", "Lime", "ليمون أخضر (لايم)", r"\blimes?\b", "citrus"),
    Note("orange", "Orange", "برتقال", r"\boranges?\b(?! (?:blossoms?|flowers?|tree|leaf|leaves))", "citrus"),
    Note("mandarin", "Mandarin", "يوسفي", r"\bmandarins?\b|\btangerines?\b|\bclementines?\b", "citrus"),
    Note("grapefruit", "Grapefruit", "جريب فروت", r"\bgrapefruit\b", "citrus"),
    Note("yuzu", "Yuzu", "يوزو", r"\byuzu\b", "citrus"),
    # Fruits
    Note("peach", "Peach", "دراق", r"\bpeach(?:es)?\b", "fruit"),
    Note("plum", "Plum", "برقوق", r"\bplums?\b", "fruit"),
    Note("apple", "Apple", "تفاح", r"\bapples?\b", "fruit"),
    Note("pear", "Pear", "إجاص", r"\bpears?\b", "fruit"),
    Note("fig", "Fig", "تين", r"\bfigs?\b", "fruit"),
    Note("raspberry", "Raspberry", "توت العليق", r"\braspberr(?:y|ies)\b", "fruit"),
    Note("blackcurrant", "Blackcurrant", "كشمش أسود", r"\bblack ?currant\b|\bcassis\b", "fruit"),
    # Fresh / green
    Note("mint", "Mint", "نعنع", r"\b(?:pepper|spear)?mint\b", "fresh"),
    Note("sage", "Sage", "ميرمية", r"\bsage\b", "fresh"),
    Note("rosemary", "Rosemary", "إكليل الجبل", r"\brosemary\b", "fresh"),
    Note("tea", "Tea", "شاي", r"\btea\b", "fresh"),
    Note("marine", "Sea notes", "نفحات بحرية", r"\bmarine\b|\bsea\b|\bocean\b|\baquatic\b|\bsalt\b", "fresh"),
    Note("aldehydes", "Aldehydes", "ألدهيدات", r"\baldehyd(?:e|es|ic)\b", "fresh"),
)

NOTES_BY_ID: dict[str, Note] = {note.id: note for note in NOTE_CATALOG}

# Scent families group the notes in the UI and colour-code them.
FAMILIES: dict[str, tuple[str, str]] = {
    "amber": ("Amber & sweet", "عنبري وحلو"),
    "woods": ("Woods", "خشبي"),
    "smoke": ("Smoke & leather", "دخاني وجلدي"),
    "spice": ("Spices", "توابل"),
    "floral": ("Floral", "زهري"),
    "citrus": ("Citrus", "حمضيات"),
    "fruit": ("Fruits", "فواكه"),
    "fresh": ("Fresh & clean", "منعش ونظيف"),
}


@cache
def _compiled(note_id: str) -> re.Pattern[str]:
    return re.compile(NOTES_BY_ID[note_id].pattern, re.IGNORECASE)


def match_notes(text: str) -> list[str]:
    """Return the ids of catalog notes that appear in ``text`` (whole words)."""
    if not text:
        return []
    return [note.id for note in NOTE_CATALOG if _compiled(note.id).search(text)]
