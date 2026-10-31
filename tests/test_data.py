from perfume_search.data import build_embedding_text, clean_name, clean_text, load_catalog, split_notes
from perfume_search.notes import match_notes


def test_clean_text_repairs_mojibake_and_spaces():
    assert clean_text("Created by AmÃ©lie  Bourgeois\r\n") == "Created by Amélie Bourgeois"
    assert clean_text(None) == ""
    assert clean_text('Rose L"˜Orange') == "Rose L‘Orange"


def test_clean_name_drops_concentration():
    assert clean_name("Tihota Eau de Parfum") == "Tihota"
    assert clean_name("Oud Night Extrait") == "Oud Night"
    assert clean_name("Parfum") == "Parfum"  # never returns an empty name


def test_split_notes_deduplicates_and_capitalises():
    assert split_notes("vanilla bean, musks; Vanilla bean,") == ["Vanilla bean", "Musks"]


def test_embedding_text_uses_notes_or_description_snippet():
    assert build_embedding_text("Tihota", ["Vanilla", "Musk"], "long story") == "Tihota. Notes: Vanilla, Musk."
    assert build_embedding_text("Mystery", [], "A soft musky veil.").startswith("Mystery. A soft")


def test_note_matching_uses_whole_words():
    assert "rose" not in match_notes("Rosemary, rosewood")  # no false 'rose'
    assert "rose" in match_notes("Turkish rose, oud")
    assert match_notes("Orange blossom") == ["orange_blossom"]  # not plain 'orange'
    assert "pepper" in match_notes("Pink pepper") and "mint" not in match_notes("Pink pepper")


def test_load_catalog_keeps_rows_without_notes(raw_csv):
    catalog = load_catalog(raw_csv)
    assert len(catalog) == 8
    no_notes = catalog[-1]
    assert no_notes.notes == [] and "Amélie" in no_notes.embedding_text
