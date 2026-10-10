import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from dadmatools import normalizer


class NormalizerResourcesTests(unittest.TestCase):
    def load_normalizer(self, directory):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        package = Path(temporary.name) / directory / "dadmatools"
        (package / "utils").mkdir(parents=True)
        source = Path(normalizer.__file__)
        module_path = package / "normalizer.py"
        shutil.copyfile(source, module_path)
        shutil.copyfile(
            source.parent / "utils" / "stopwords-fa.py",
            package / "utils" / "stopwords-fa.py",
        )
        spec = importlib.util.spec_from_file_location("relocated_normalizer", module_path)
        relocated = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(relocated)
        return relocated.Normalizer()

    def test_normalizer_loads_from_paths_containing_models(self):
        for directory in ("models", "saved_models", "models_workspace"):
            with self.subTest(directory=directory):
                instance = self.load_normalizer(directory)
                self.assertIn("از", instance.STOPWORDS)
                self.assertEqual(instance.normalize("سلام   دنیا"), "سلام دنیا")

    def test_normalizer_loads_from_an_ordinary_path(self):
        instance = self.load_normalizer("library")
        self.assertIn("از", instance.STOPWORDS)
        self.assertEqual(instance.normalize("سلام   دنیا"), "سلام دنیا")
