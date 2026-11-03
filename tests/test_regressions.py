"""Offline regression tests for aircraft_damage_classifier.

TensorFlow, transformers, PIL and matplotlib are replaced by stubs while the
module is imported, so these tests need no heavy dependencies, no dataset and
no model weights. Only pytest is required.
"""
import ast
import importlib
import io
import math
import os
import re
import sys
import tarfile
import types
from unittest import mock

import numpy as np  # imported before the stubs so patch.dict never unloads it
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEAVY_MODULES = [
    "tensorflow", "tensorflow.keras", "tensorflow.keras.applications",
    "tensorflow.keras.models", "tensorflow.keras.layers",
    "tensorflow.keras.optimizers", "tensorflow.keras.preprocessing",
    "tensorflow.keras.preprocessing.image",
    "transformers", "PIL", "matplotlib", "matplotlib.pyplot",
]


@pytest.fixture(scope="module")
def mod():
    stubs = {name: mock.MagicMock() for name in HEAVY_MODULES}
    sys.path.insert(0, ROOT)
    try:
        with mock.patch.dict(sys.modules, stubs):
            sys.modules.pop("aircraft_damage_classifier", None)
            module = importlib.import_module("aircraft_damage_classifier")
    finally:
        sys.path.remove(ROOT)
        sys.modules.pop("aircraft_damage_classifier", None)
    return module


# ---------------------------------------------------------------- evaluation

class FakeBatchIterator:
    """Mimics a Keras directory iterator: yields batches of sample ids and wraps around."""

    def __init__(self, samples, batch_size):
        self.samples = samples
        self.batch_size = batch_size
        self._pos = 0

    def __len__(self):
        return math.ceil(self.samples / self.batch_size)

    def __next__(self):
        if self.samples == 0:
            raise StopIteration
        batch = [(self._pos + i) % self.samples for i in range(min(self.batch_size, self.samples - self._pos))]
        self._pos = (self._pos + len(batch)) % self.samples
        return batch


class FakeModel:
    """evaluate() consumes `steps` batches (len(x) when steps is None) and records the ids."""

    def __init__(self):
        self.seen = []

    def evaluate(self, x, steps=None):
        for _ in range(len(x) if steps is None else steps):
            self.seen.extend(next(x))
        return 0.5, 0.75


@pytest.mark.parametrize("samples", [70, 64, 33, 5, 1])
def test_evaluate_model_uses_every_test_sample_exactly_once(mod, samples):
    clf = mod.AircraftDamageClassifier()
    clf.model = FakeModel()
    clf.test_generator = FakeBatchIterator(samples, batch_size=32)
    assert clf.evaluate_model() == (0.5, 0.75)
    assert sorted(clf.model.seen) == list(range(samples))


def test_evaluate_model_rejects_empty_test_set(mod):
    clf = mod.AircraftDamageClassifier()
    clf.model = FakeModel()
    clf.test_generator = FakeBatchIterator(0, batch_size=32)
    with pytest.raises(ValueError):
        clf.evaluate_model()


# ------------------------------------------------------------ label decoding

KERAS_MAPPING = {"crack": 0, "dent": 1}  # Keras assigns indices from sorted folder names
REVERSED_MAPPING = {"dent": 0, "crack": 1}


@pytest.mark.parametrize("probability, mapping, label, confidence", [
    (0.9, KERAS_MAPPING, "dent", 0.9),
    (0.1, KERAS_MAPPING, "crack", 0.9),
    (0.9, REVERSED_MAPPING, "crack", 0.9),
    (0.1, REVERSED_MAPPING, "dent", 0.9),
    (0.5, KERAS_MAPPING, "crack", 0.5),  # Keras scores only > 0.5 as class 1
])
def test_decode_prediction_follows_training_class_mapping(mod, probability, mapping, label, confidence):
    got_label, got_confidence = mod.decode_prediction(probability, mapping)
    assert got_label == label
    assert got_confidence == pytest.approx(confidence)


@pytest.mark.parametrize("mapping", [None, {}, {"crack": 0}, {"a": 0, "b": 2}])
def test_decode_prediction_requires_binary_mapping(mod, mapping):
    with pytest.raises(ValueError):
        mod.decode_prediction(0.9, mapping)


class FakeGenerator:
    def __init__(self, class_indices):
        self.class_indices = class_indices
        self.samples = 10


def _install_fake_generators(mod, monkeypatch, valid_mapping):
    mappings = {"train": KERAS_MAPPING, "valid": valid_mapping, "test": KERAS_MAPPING}

    class FakeDatagen:
        def __init__(self, **kwargs):
            pass

        def flow_from_directory(self, directory, **kwargs):
            return FakeGenerator(dict(mappings[os.path.basename(directory)]))

    monkeypatch.setattr(mod, "ImageDataGenerator", FakeDatagen)


def test_setup_data_generators_keeps_training_class_mapping(mod, monkeypatch):
    _install_fake_generators(mod, monkeypatch, KERAS_MAPPING)
    clf = mod.AircraftDamageClassifier()
    clf.setup_data_generators("data")
    assert clf.class_indices == KERAS_MAPPING


def test_setup_data_generators_rejects_inconsistent_mapping(mod, monkeypatch):
    _install_fake_generators(mod, monkeypatch, REVERSED_MAPPING)
    with pytest.raises(ValueError):
        mod.AircraftDamageClassifier().setup_data_generators("data")


class FakePredictModel:
    def __init__(self, probability):
        self.probability = probability

    def predict(self, array):
        return [[self.probability]]


@pytest.fixture
def classifier_with_fakes(mod, monkeypatch):
    fake_tf = types.SimpleNamespace(keras=types.SimpleNamespace(utils=types.SimpleNamespace(
        load_img=lambda path, target_size: np.zeros(target_size + (3,)),
        img_to_array=lambda img: np.asarray(img, dtype="float32"),
    )))
    monkeypatch.setattr(mod, "tf", fake_tf)

    def make(probability, mapping=KERAS_MAPPING):
        clf = mod.AircraftDamageClassifier()
        clf.model = FakePredictModel(probability)
        clf.class_indices = dict(mapping)
        clf.generate_caption = lambda path, task: f"{task} text"
        return clf

    return make


def test_predict_and_caption_labels_with_actual_mapping(classifier_with_fakes):
    # sigmoid output 0.9 is class index 1, which is "dent" under Keras' sorted-folder mapping
    result = classifier_with_fakes(0.9).predict_and_caption("img.jpg")
    assert result["damage_type"] == "dent"
    assert result["confidence"] == pytest.approx(0.9)
    assert classifier_with_fakes(0.1).predict_and_caption("img.jpg")["damage_type"] == "crack"


# ------------------------------------------------------------ dataset set-up

def _write_tar(path, members):
    """members: list of (name, bytes or None for directory)."""
    with tarfile.open(path, "w") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            if data is None:
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))


def _patch_download(mod, monkeypatch, members):
    calls = []

    def fake_urlretrieve(url, filename):
        calls.append(url)
        _write_tar(filename, members)

    monkeypatch.setattr(mod.urllib.request, "urlretrieve", fake_urlretrieve)
    return calls


# the real archive has one top-level folder with explicit folder entries and plain files
VALID_MEMBERS = [
    ("aircraft_damage_dataset_v1", None),
    ("aircraft_damage_dataset_v1/train", None),
    ("aircraft_damage_dataset_v1/train/dent", None),
    ("aircraft_damage_dataset_v1/train/dent/a.jpg", b"x"),
    ("aircraft_damage_dataset_v1/valid/dent/b.jpg", b"x"),
    ("aircraft_damage_dataset_v1/test/dent/c.jpg", b"x"),
]


def test_download_dataset_extracts_into_data_dir_and_cleans_up(mod, monkeypatch, tmp_path):
    _patch_download(mod, monkeypatch, VALID_MEMBERS)
    path = mod.AircraftDamageClassifier().download_dataset(str(tmp_path))
    assert path == str(tmp_path / "aircraft_damage_dataset_v1")
    assert (tmp_path / "aircraft_damage_dataset_v1" / "test" / "dent" / "c.jpg").exists()
    assert os.listdir(tmp_path) == ["aircraft_damage_dataset_v1"]  # no tar or staging left behind


def test_download_dataset_reuses_valid_existing_copy(mod, monkeypatch, tmp_path):
    calls = _patch_download(mod, monkeypatch, VALID_MEMBERS)
    for split in ("train", "valid", "test"):
        (tmp_path / "aircraft_damage_dataset_v1" / split).mkdir(parents=True)
    mod.AircraftDamageClassifier().download_dataset(str(tmp_path))
    assert calls == []


def test_download_dataset_does_not_delete_unrelated_existing_folder(mod, monkeypatch, tmp_path):
    calls = _patch_download(mod, monkeypatch, VALID_MEMBERS)
    target = tmp_path / "aircraft_damage_dataset_v1"
    target.mkdir()
    (target / "my_notes.txt").write_text("keep me")
    with pytest.raises(FileExistsError):
        mod.AircraftDamageClassifier().download_dataset(str(tmp_path))
    assert (target / "my_notes.txt").read_text() == "keep me"
    assert calls == []


def test_download_dataset_refuses_path_traversal(mod, monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    _patch_download(mod, monkeypatch, [("../evil.txt", b"x")] + VALID_MEMBERS)
    with pytest.raises(ValueError):
        mod.AircraftDamageClassifier().download_dataset(str(data_dir))
    assert not (tmp_path / "evil.txt").exists()
    assert os.listdir(data_dir) == []  # no half-extracted dataset, no staging left


def test_download_dataset_rejects_unexpected_archive_layout(mod, monkeypatch, tmp_path):
    _patch_download(mod, monkeypatch, [("something_else/readme.txt", b"x")])
    with pytest.raises(RuntimeError):
        mod.AircraftDamageClassifier().download_dataset(str(tmp_path))
    assert os.listdir(tmp_path) == []


# ------------------------------------------------------------ README example

def _readme_example():
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        blocks = re.findall(r"```python\n(.*?)```", f.read(), re.S)
    assert blocks, "README has no python example"
    return ast.parse(blocks[0])


def test_readme_example_matches_implementation(mod, classifier_with_fakes):
    tree = _readme_example()

    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert [(i.module, a.name) for i in imports for a in i.names] == [
        ("aircraft_damage_classifier", "AircraftDamageClassifier")
    ]
    assert os.path.exists(os.path.join(ROOT, "aircraft_damage_classifier.py"))

    instance_names, methods, result_names, keys = set(), [], set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id == "AircraftDamageClassifier":
                instance_names |= {t.id for t in node.targets}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and func.attr == "predict_and_caption":
                result_names |= {t.id for t in node.targets}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id in instance_names:
                methods.append(node.func.attr)
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in result_names:
            keys.append(node.slice.value)

    assert instance_names and methods and keys
    for method in methods:
        assert callable(getattr(mod.AircraftDamageClassifier, method)), method
    returned = classifier_with_fakes(0.9).predict_and_caption("img.jpg")
    assert set(keys) <= set(returned), (keys, sorted(returned))
