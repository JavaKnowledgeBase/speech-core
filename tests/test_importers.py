from pathlib import Path

from app.importers import load_reference_vectors, load_target_profiles


class TestImporters:
    def test_load_target_profiles_from_json(self):
        items = load_target_profiles(Path("seed_data/target_profiles.json"))
        assert len(items) == 20

    def test_load_reference_vectors_from_json(self):
        items = load_reference_vectors(Path("seed_data/reference_vectors.json"))
        assert len(items) >= 80
        assert items[0].embedding

    def test_load_reference_vectors_from_csv(self):
        items = load_reference_vectors(Path("seed_data/reference_vectors.csv"))
        assert len(items) >= 80
        assert items[0].embedding
        assert items[0].reference_id.startswith("ref-")
