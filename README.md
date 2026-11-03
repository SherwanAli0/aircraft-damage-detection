# Aircraft Damage Classification

A transfer-learning classifier that labels aircraft-surface images as **dent** or **crack** (VGG16 backbone, frozen ImageNet weights, small dense head), plus a BLIP image-captioning step that produces a short text description of the same image.

This project was completed by Sherwan Ali as part of the course **Introduction to Deep Learning and Neural Networks with Keras** in the **IBM AI Engineering Professional Certificate**. The course supplied the dataset (downloaded from IBM's course storage, URL in `aircraft_damage_classifier.py`) and the task structure. VGG16 (ImageNet weights) and BLIP (`Salesforce/blip-image-captioning-base`) are third-party pre-trained models that are used as they are, not trained here.

It is a learning project, not a maintenance tool: no inspection-workflow integration, API, real-time capability or safety effectiveness has been built or measured.

## What is in this repository

```
aircraft-damage-detection/
├── aircraft_damage_classifier.py   # AircraftDamageClassifier class and main() pipeline
├── requirements.txt                # Dependencies
├── tests/test_regressions.py       # Offline tests (no dataset, weights or TensorFlow needed)
├── Result/
│   └── Model loss and accuracy.png # Training curves from an earlier run (see Results)
├── Sample images/                  # Two example images from the dataset
├── LICENSE
└── README.md
```

## Requirements

- Python 3.11 (TensorFlow does not support Python 3.14).
- `pip install -r requirements.txt` installs TensorFlow, transformers, PyTorch (needed by the BLIP captioning code), Pillow, matplotlib and numpy. The BLIP classes in transformers need PyTorch and fail with an `ImportError` without it.
- Environment used for the checks in this repository: Python 3.11.9, TensorFlow 2.21.0 (Keras 3.15.1), transformers 5.22.1, torch 2.15.0 (CPU), numpy 2.4.6, Pillow 13.0.0, matplotlib 3.11.2. In it the module imports, the offline tests pass, and the classification pipeline (dataset download and extraction, VGG16 training, evaluation, label decoding) was run on the real dataset (see Results). BLIP captioning was not run.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python aircraft_damage_classifier.py
```

This downloads the dataset into the current directory (about 17 MB; see Data), downloads the VGG16 ImageNet weights (about 59 MB) and the BLIP model on first use, trains for 5 epochs, evaluates on the test split, shows the training curves, and classifies and captions one sample image. Training and BLIP are slow without a GPU.

## Usage

```python
from aircraft_damage_classifier import AircraftDamageClassifier

classifier = AircraftDamageClassifier()
data_path = classifier.download_dataset()          # extracts ./aircraft_damage_dataset_v1
classifier.setup_data_generators(data_path)
classifier.build_model()
classifier.train_model()
classifier.evaluate_model()                        # returns (test_loss, test_accuracy)

result = classifier.predict_and_caption(
    "Sample images/149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg"
)
print(f"Damage type: {result['damage_type']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Caption: {result['caption']}")
print(f"Summary: {result['summary']}")
```

`predict_and_caption` returns a dictionary with the keys `damage_type`, `confidence`, `caption` and `summary`. The model must be trained in the same session; trained weights are not saved to disk.

## Data

- Source: `https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar` (about 17.2 MB).
- `download_dataset(data_dir=".")` unpacks it to `<data_dir>/aircraft_damage_dataset_v1/`. If a valid copy is already there it is reused. If a different folder with that name exists, it is left untouched and an error is raised. Nothing is moved into place unless the archive contains `train`, `valid` and `test` folders.
- The dataset ships with fixed train (300 images), validation (96) and test (50) splits, each balanced between the two classes (the code does not create splits and does not use cross-validation):

```
aircraft_damage_dataset_v1/
├── train/{dent,crack}/
├── valid/{dent,crack}/
└── test/{dent,crack}/
```

- Keras assigns class indices from the sorted folder names, so with these folder names `crack` is 0 and `dent` is 1. The model's single sigmoid output is the probability of class 1 (`dent`). The code prints the mapping when the generators are created and uses that same mapping to turn predictions into labels (a prediction above 0.5 is class 1, anything else class 0).

## Model

```
Backbone:  VGG16, ImageNet weights, no top, layers frozen
Head:      GlobalAveragePooling2D -> Dense(512) -> Dropout(0.3) -> Dense(512) -> Dropout(0.3) -> Dense(1, sigmoid)
Input:     RGB 224x224, pixels rescaled to [0, 1]
Training:  Adam (lr 0.0001), binary cross-entropy, batch size 32, 5 epochs, seed 42
```

## Results

### Historical result (earlier run)

`Result/Model loss and accuracy.png` is the training-curve plot from the original run, kept as it was. Read off the plot, after 5 epochs training accuracy is roughly 0.57 and validation accuracy roughly 0.59, with loss around 0.67. No log, saved model or test-set metric came with it.

### Re-run on 2026-11-03

The pipeline in this repository (after the repairs below) was re-run with unchanged preprocessing, architecture and hyperparameters, on CPU, in a scratch copy so that no data or weights were added to the repository.

- Environment: Python 3.11.9, TensorFlow 2.21.0 (Keras 3.15.1), numpy 2.4.6, Pillow 13.0.0; CPU only. VGG16 weights: the standard Keras ImageNet `notop` file.
- Data: the real archive (17,182,208 bytes) was used. It contains one top-level folder with `train` (300 images, 150 `crack` + 150 `dent`), `valid` (96, 48 + 48) and `test` (50, 25 + 25), regular files and folders only. Keras gave the mapping `{'crack': 0, 'dent': 1}`.
- Seeds: `random`, numpy and TensorFlow are seeded with 42 and the generators use seed 42. The run was done twice from a fresh extraction and gave identical per-epoch numbers and test results (training took 202 s in one run and 475 s in the other on the same machine, so runtime is only indicative). Other hardware or library versions may give different numbers.
- Per-epoch results (epochs 1 to 5):

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---|---|---|---|---|
| 1 | 0.7235 | 0.4867 | 0.7034 | 0.5000 |
| 2 | 0.7060 | 0.5433 | 0.6811 | 0.5938 |
| 3 | 0.6942 | 0.5167 | 0.6752 | 0.6146 |
| 4 | 0.6890 | 0.5467 | 0.6706 | 0.6458 |
| 5 | 0.6645 | 0.5700 | 0.6683 | 0.5938 |

  These match the historical plot, so the plot is reproduced.
- Full test set (all 50 images, `evaluate_model()`): loss 0.6452, accuracy 0.62.
- Check of `decode_prediction` on the real predictions (all 50 test images, each predicted once): the decoded labels give accuracy 0.62, the same as Keras. Confusion matrix (rows true, columns predicted):

| | predicted crack | predicted dent |
|---|---|---|
| true crack (25) | 7 | 18 |
| true dent (25) | 1 | 24 |

  The model predicts `dent` for 84% of images (sigmoid outputs range from 0.43 to 0.65), so it separates the classes only weakly: it finds most dents but only 7 of 25 cracks. Decoding with the old hardcoded mapping would have given 0.38.
- The earlier evaluation code (`samples // batch_size` steps) would have scored only the first 32 of the 50 test images, giving accuracy 0.4375 instead of 0.62.
- Not measured: BLIP captions (the BLIP model was not downloaded or run, so `predict_and_caption` captions are unverified), other metrics, and any run on a GPU or other library versions.

## Notes on earlier code

Earlier versions of the script evaluated only complete batches of the test set and mapped sigmoid outputs to labels in the wrong direction (`crack` for values above 0.5, although with these folders index 1 is `dent`). Both are fixed; any numbers or labels produced by the old code are not comparable.

## Tests

```bash
pip install pytest
python -m pytest tests
```

The tests use stand-ins for TensorFlow, BLIP and the download, so they need no dataset, model weights or GPU. They cover evaluation of the final partial test batch, label decoding against the real and a reversed class mapping, the dataset download/extract safeguards, and that the Usage example above matches the code.

## Limitations

- Small binary dataset (dent vs crack); no other damage types or severity levels.
- Accuracy is low (0.62 on the 50-image test set; weak on cracks); the model is not suitable for any real inspection decision.
- Captions come from a general-purpose BLIP model and have not been evaluated for aviation accuracy.
- The BLIP prompts and 0.5 decision threshold are fixed in the code.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The dataset, the VGG16 weights and the BLIP model are not covered by it and remain under their own terms.
