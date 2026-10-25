# Rock-Paper-Scissors — Custom PyTorch CNN + W&B + Webcam GUI

This project trains a CNN **from scratch** (no pretrained weights) using PyTorch, logs training/validation/test metrics to Weights & Biases, and provides a live webcam GUI.

## 1. Dataset format

Place your dataset under `rps/` with exactly these class folders:

```text
rps/
├── paper/
│   ├── person_a/
│   │   ├── image001.jpg
│   │   └── ...
│   └── ...
├── rock/
│   └── ...
└── scissor/
    └── ...
```

`torchvision.datasets.ImageFolder` recursively discovers images within each class directory. The current `rps-1024/paper`, `rps-1024/rock`, and `rps-1024/scissor` dataset can be used directly because nested source folders are supported.

Run from this project directory and point `--data` at your dataset root, e.g. `--data rps-1024`.

## 2. Dataset requirements

The assignment asks for:
- At least 500 genuine images in each class.
- At least 2,000 images overall.
- An imbalanced class distribution.

`train.py` prints class counts and warns if these requirements are not met. Collect images from your own hands and/or consenting friends/family. Do not inflate counts by duplicating images. Try to vary lighting, background, distance, hand orientation, and people.

## 3. Setup (Ubuntu/Linux)

Python 3.10 or 3.11 is a practical choice. In a terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your system has an incompatible Python version or the PyTorch install fails, use a supported Python 3.10/3.11 virtual environment. For CPU-only installation, follow the official PyTorch installation selector at https://pytorch.org/get-started/locally/.

On Ubuntu, Tkinter may need to be installed separately:

```bash
sudo apt update
sudo apt install python3-tk
```

## 4. Weights & Biases

Create/log into your W&B account:

```bash
wandb login
```

Copy the API key from your W&B account when prompted. Then train:

```bash
python train.py --data_path rps-1024 --epochs 25 --batch-size 16
```

The default run logs train/validation loss, accuracy, learning rate, held-out test accuracy, and a confusion matrix to the W&B project `rps-custom-cnn`.

For a quick local debug run only, you can disable W&B:

```bash
python train.py --data_path rps-1024 --epochs 2 --no-wandb
```

For submission, run without `--no-wandb` and make sure the W&B run is visible in your account.

## 5. Outputs

- `checkpoints/rps_custom_cnn.pth` — best model selected by validation loss.
- `results/training_curves.png` — train/validation loss and accuracy.
- `results/confusion_matrix.png` — held-out test confusion matrix.
- `results/test_report.json` — test accuracy and per-class metrics.

The test set is held out from model selection and is evaluated after training. Split is 70% train / 15% validation / 15% test using stratified random image splitting.

**Evaluation caveat:** if nested source folders correspond to people, random image splitting may put images from the same person in both train and test. For a stronger generalization claim, use a source/person-separated split where there are enough distinct sources for every class. Do not claim source-separated evaluation unless you actually performed it.

## 6. Webcam GUI

After training:

```bash
python webcam_gui.py
```

If the default camera does not work:

```bash
python webcam_gui.py --camera 1
```

The app shows the live camera, prediction, confidence, and approximate FPS. Press **Quit** or close the window to release the camera.

## 7. Accuracy

The code is designed as a CPU-friendly baseline, but accuracy above 85% cannot be guaranteed without running experiments on your collected dataset. If validation accuracy stalls, inspect class labels and examples, collect more diverse real images, and check whether the model is learning the background rather than the hand gesture. Report the actual held-out test accuracy.

## 8. Assignment checklist

- [ ] ImageFolder directory structure used.
- [ ] 500+ genuine images per class.
- [ ] 2,000+ total images.
- [ ] Class imbalance present and documented.
- [ ] Custom CNN trained from scratch (no pretrained weights).
- [ ] Train/validation/test split documented.
- [ ] W&B logs show losses and metrics.
- [ ] Test report and confusion matrix saved.
- [ ] Webcam GUI displays live classification.
- [ ] Report actual measured test accuracy; do not fabricate results.