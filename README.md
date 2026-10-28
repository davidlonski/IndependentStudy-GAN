# IndependentStudy-GAN

**A Low-Cost Approach for Fake Medical Image Synthesis.** PyTorch DC-GAN and W-GAN models trained to generate synthetic retinal OCT scans on a single lab GPU.

David Lonski, with faculty advisor Dr. Ning Yu, SUNY Brockport. Research conducted February to July 2025: an independent study in the spring and paid summer research.

**Full write-up:** [Research Report 01 (PDF)](https://www.davidlonski.com/research/medical-image-synthesis/report.pdf) · **More research:** [davidlonski.com/research](https://www.davidlonski.com/research)

## About

Medical imaging models need large, balanced, labeled datasets, but clinical scans are scarce, unevenly spread across conditions, and restricted by patient privacy. This project tested whether one lab workstation and an open dataset could train generative adversarial networks (GANs) to produce realistic synthetic retinal optical coherence tomography (OCT) scans.

The code trains two GAN families on the same data pipeline:

- **DC-GAN:** binary cross-entropy with logits, Adam, one discriminator update per generator update.
- **W-GAN:** Wasserstein loss, RMSprop, five critic updates per generator update, critic weights clipped to ±0.01.

Both families share the same convolutional generator and discriminator, sized for 64, 128 or 256 px images, train with mixed precision (`torch.amp`), and log the generator and discriminator loss of every batch.

The data is the [UCSD 3-class labeled retinal OCT images](https://www.kaggle.com/datasets/mmazizi/ucsd-3-class-labeled-retinal-oct-images) dataset on Kaggle (mmazizi), a three-class version (NORMAL, CNV, DRUSEN) of the Kermany et al. (2018) collection.

## Results

From [Research Report 01](https://www.davidlonski.com/research/medical-image-synthesis/report.pdf):

- **12 instrumented runs** (4 W-GAN, 8 DC-GAN) at 64 and 256 px on one NVIDIA RTX 5000 Ada GPU: **123.5 GPU-hours** and 2,494,057 logged batches.
- **Best images: DC-GAN runs 256_4 and 256_6** (batch 32, 128 generator feature maps). Their 256 × 256 samples show continuous retinal layers and foveal pits.
- **Four DC-GAN runs stalled** (64_1, 64_2, 256_1, 256_2): the generator loss stayed at exactly ln 2 = 0.6931 and the discriminator loss at 1.0064. Run 256_2 produced uniform gray output.
- **W-GAN images improved with longer training** (5 to 100 epochs), while the losses barely moved.
- **Read the per-batch logs, not the summary.** `training_summary.csv` records the losses of the last batch only, and GAN losses swing widely from batch to batch: run 256_4's generator loss ranged from 0 to 293.8. The last batch of 256_4 (G 0.052, D 0.443) suggested balance, but averaged over the final epoch, runs 256_4, 256_5 and 256_6 all ended with the discriminator dominating (epoch-mean D loss 0.014 to 0.022). Section 6.3 of the report has the details.
- No quantitative image-quality metric such as FID was computed; image quality was judged from the saved sample grids.

## Setup

You need Python 3.11 and an NVIDIA GPU with CUDA. The training loops run under `torch.amp` autocast and gradient scaling for `cuda`.

```bash
git clone https://github.com/davidlonski/IndependentStudy-GAN.git
cd IndependentStudy-GAN
python -m venv venv
```

Activate the environment (`venv\Scripts\activate` on Windows, `source venv/bin/activate` on Linux or macOS). On Windows, install the CUDA build of PyTorch first, because the `torch==2.5.1` wheel on PyPI is CPU-only there:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Then install the pinned dependencies, plus the packages the trainers and DC-GAN scripts use that the list leaves out:

```bash
pip install -r src/wgan/requirements.txt pandas python-dotenv psutil tqdm
```

The root `requirements.txt` is a placeholder; the real pins are in `src/wgan/requirements.txt`. To confirm PyTorch can see the GPU, run `python doc/testcuda.py`.

## Dataset

Download the dataset (a 6.3 GB zip) from [Kaggle](https://www.kaggle.com/datasets/mmazizi/ucsd-3-class-labeled-retinal-oct-images), or with the Kaggle CLI:

```bash
kaggle datasets download -d mmazizi/ucsd-3-class-labeled-retinal-oct-images
```

The archive holds `OCTUCSD-3class/OCTUCSD-3class/OCT/{train,test}/{CNV,DRUSEN,NORMAL}/` and three split manifests (`ucsd_train.csv`, `ucsd_val.csv`, `ucsd_test.csv`). The trainers read `content/OCT/train/<CLASS>/` at the repository root, so unzip it and move the `OCT` folder there:

```bash
unzip ucsd-3-class-labeled-retinal-oct-images.zip -d content
mv content/OCTUCSD-3class/OCTUCSD-3class/OCT content/OCT
```

The `train` folder holds 96,961 images, the dataset's train and validation splits together, and this project trained on all of them. The 750-image `test` split was held out. Both `content/` and the zip are git-ignored.

## Repository structure

```
IndependentStudy-GAN/
├── src/
│   ├── dcgan/                         DC-GAN
│   │   ├── train.py                   DCGANTrainer: preprocessing, training loop, saving results
│   │   ├── generator.py               Generator for 64, 128 or 256 px images
│   │   ├── discriminator.py           Discriminator for 64, 128 or 256 px images
│   │   ├── data_loader.py             OCT preprocessing and the processed-image cache
│   │   ├── test.py                    Runs the configuration of DC-GAN run 256_6, then sends an email alert
│   │   ├── analyze_results.py         Compares the DCGAN_Test_256_* runs
│   │   ├── analyze_training_results.py  Loss curves, epoch times and statistics for one run's CSVs
│   │   ├── test_suite.py              A four-configuration sweep (June 19, 2025)
│   │   ├── improved_test.py           A batch-8 follow-up with resource monitoring (June 30, 2025)
│   │   ├── run_improved_test.py       Checks system resources, then runs that follow-up
│   │   ├── enhanced_trainer.py        A standalone trainer variant written for the follow-up
│   │   ├── utils.py                   Email alerts on run completion or failure
│   │   └── test_email.py              Checks the email settings and sends test alerts
│   ├── wgan/                          W-GAN: same core files as dcgan/ (WGANTrainer), plus
│   │   ├── test.py, test2.py, test3.py  256 × 256 configurations; test3.py is W-GAN Test 3
│   │   ├── test_suite.py              Five-configuration follow-up sweep
│   │   └── requirements.txt           Pinned dependencies
│   ├── idcgan/                        Separate ID-CGAN image de-raining experiment, not part of the OCT work
│   └── common/                        Empty placeholder package
├── generated_images/                  Telemetry CSVs from the runs finished by June 24, 2025
├── models/WGAN/gan_model.pth          A 64 × 64 W-GAN checkpoint (generator, critic and optimizer states)
├── doc/
│   ├── Test01.md                      Notes on the May 2025 modular rewrite and switch to W-GAN
│   ├── Email_Setup_Guide.md           How to set up the email alerts
│   ├── testcuda.py                    Prints CUDA availability and the GPU name
│   ├── DCGAN_Notes.md                 Empty
│   └── WGAN_Notes.md                  Empty
├── GAN_IndependentStudy.ipynb         Spring 2025 notebook, adapted from the PyTorch DC-GAN tutorial
└── requirements.txt                   Placeholder (see Setup)
```

The DC-GAN follow-up scripts (`test_suite.py`, `improved_test.py`, `run_improved_test.py`, `enhanced_trainer.py`) were prepared in late June 2025. The report records no results from them.

## Running the trainers

The trainers take no command-line flags. Each is a class whose constructor arguments define the run, and a run has three steps: `preprocess_data()` cleans and caches the raw images, `train()` trains, and `save_results()` writes everything to disk.

Run each script from inside its own folder. The scripts import their neighbors (`from train import DCGANTrainer`) and resolve paths from there: raw images from `../../content/OCT/train/`, results to `../../generated_images/`.

### DC-GAN

```bash
cd src/dcgan
python train.py            # constructor defaults: 64 × 64, batch 256, 5 epochs
python test.py             # run 256_6's settings: 256 × 256, batch 32, ngf 128, ndf 64, 150 epochs, lr 0.0002
python analyze_results.py  # table and plots comparing generated_images/DCGAN_Test_256_*
```

`test.py` sends an email when training ends or fails. Without the email settings (see [Email alerts](#email-alerts)) it prints a warning, skips the alert, and training is unaffected.

### W-GAN

```bash
cd src/wgan
python train.py            # constructor defaults: 64 × 64, batch 256, 5 epochs
python test_suite.py       # five 256 × 256 configurations, one after another
python analyze_results.py  # table and plots comparing generated_images/WGAN_Test_256_*
```

`test_suite.py` is a follow-up sweep planned in June 2025 that left no results: a base of batch 256, latent size 128, 100 epochs and lr 0.0001, then one change per test (128 feature maps, batch 64, lr 0.0002, latent size 256, 200 epochs). Each configuration is a long run; W-GAN Test 3, at 100 epochs, took 14.2 hours. Don't use `src/wgan/test.py`, `test2.py` or `test3.py` as committed (see [Known issues](#known-issues)).

### Your own configuration

Write a short script next to `train.py`, the way `test.py` does. The `if __name__ == '__main__'` guard and `freeze_support()` matter on Windows, where the data loader starts worker processes. For example, with the settings the report lists for DC-GAN run 256_4 (which took 31.1 hours):

```python
# src/dcgan/run_256_4.py
from multiprocessing import freeze_support
from train import DCGANTrainer

if __name__ == '__main__':
    freeze_support()
    trainer = DCGANTrainer(workers=24, batch_size=32, image_size=256,
                           ngf=128, ndf=128, num_epochs=150, lr=0.0002)
    trainer.preprocess_data()
    trainer.train()
    trainer.save_results()
```

Both trainers take the same constructor arguments:

| Argument | Default | Meaning |
|---|---|---|
| `manual_seed` | `999` | Seed for Python and PyTorch; deterministic algorithms are switched on |
| `workers` | `4` | DataLoader worker processes (the project's runs used 24) |
| `batch_size` | `256` | Images per batch |
| `image_size` | `64` | `64`, `128` or `256` |
| `nc` | `3` | Image channels |
| `nz` | `100` | Latent vector size |
| `ngf` | `64` | Generator base feature maps |
| `ndf` | `64` | Discriminator (critic) base feature maps |
| `num_epochs` | `5` | Training epochs |
| `lr` | `0.0002` | Learning rate (Adam for DC-GAN, RMSprop for W-GAN) |
| `beta1` | `0.5` | Adam β1 (DC-GAN only) |
| `ngpu` | `1` | GPUs; more than 1 wraps both networks in `DataParallel` |
| `input_dirs` | `None` | Raw image folders; `None` means `../../content/OCT/train/{DRUSEN,CNV,NORMAL}` |
| `output_dir` | `"content/TRAIN"` | Processed-image cache, relative to the working directory |

The W-GAN's five critic updates and ±0.01 clip are fixed inside `train()` (`n_critic`, `clip_value`).

### Preprocessing

The first time it runs, `preprocess_data()` writes a cleaned copy of every raw image to `output_dir`, one folder per class. Later runs reuse that cache and skip images already in it. Each image is:

1. loaded as grayscale, with pure-white pixels set to black, which removes burned-in scan text;
2. min-max normalized, with any new white pixels cleared again.

At load time, images are resized and center-cropped to `image_size`, read as 3 channels, and scaled to the range −1 to 1.

The original pipeline also rotated each image by a random angle between 0 and 360 degrees. That step is commented out in both loaders: in the W-GAN loader since June 5, 2025, and in the DC-GAN loader since June 25, 2025, just before run 256_4. DC-GAN runs before 256_4 trained on rotated scans. If you have a cache built by an older version, delete `src/dcgan/content/TRAIN` so it is rebuilt from upright scans.

### Output

Each `save_results()` call writes a new folder, `generated_images/DCGAN_Test_<size>_<n>/` or `generated_images/WGAN_Test_<size>_<n>/`, using the next free number `n`:

| File | Contents |
|---|---|
| `batch_results.csv` | Generator and discriminator loss for every batch |
| `epoch_results.csv` | Wall-clock time of every epoch |
| `training_summary.csv` | Total time, epochs and batches, and the **last batch's** losses |
| `training_results.png` | Loss plot |
| `final_grid.png` | 64 samples from a fixed noise batch |
| `real_vs_fake_comparison.png` | Real training images beside generated ones |
| `training_progress.gif` | Fixed-noise samples saved every 500 batches |
| `individual_images/` | 256 generated images as PNGs |
| `gan_model_<size>.pth` | Generator, discriminator and both optimizer states |

Images and `.pth` files are git-ignored, so the repository keeps only the CSVs.

### What's in `generated_images/`

This folder holds telemetry for the eight runs finished by June 24, 2025: W-GAN 64_3, 256_1, 256_2 and 256_3, and DC-GAN 64_1, 64_2, 256_1 and 256_2. It also holds `WGAN_Test_256_results_comparison.csv` from `src/wgan/analyze_results.py`, and `WGAN_Test_256_1/individual_images.zip` with that run's 256 samples. DC-GAN runs 256_3 to 256_6, including the two best, are not in the repository; their results are in the report.

## Email alerts

Runs of up to 31 hours needed unattended monitoring, so in late June 2025 the DC-GAN scripts gained an email alert on completion or failure. `src/dcgan/utils.py` sends it through Gmail's SMTP server, reading the settings from environment variables or from a `.env` file in `src/dcgan/` (git-ignored):

```
EMAIL_SENDER=sender@gmail.com
EMAIL_PASSWORD=<a Gmail app password>
EMAIL_RECIPIENT=recipient@example.com
```

`doc/Email_Setup_Guide.md` explains how to create the app password, and `python test_email.py` (from `src/dcgan/`) checks the settings.

## Known issues

- `src/wgan/test.py`, `test2.py` and `test3.py` call `trainer.plot_training_results()`, which is defined only inside `save_results()`. They raise `AttributeError` after training finishes, before any results are saved.
- The root `requirements.txt` is a placeholder, and `src/wgan/requirements.txt` leaves out pandas, python-dotenv, psutil and tqdm.

## Report and acknowledgments

[Research Report 01, "A Low-Cost Approach for Fake Medical Image Synthesis" (PDF)](https://www.davidlonski.com/research/medical-image-synthesis/report.pdf) covers the methods, every run's configuration, the figures, and a re-analysis of the per-batch telemetry. The work was presented as poster #35 at SUNY Brockport Scholars Day on April 9, 2025 (Lonski, Yu). More projects are at [davidlonski.com/research](https://www.davidlonski.com/research).

Thanks to Dr. Ning Yu for advising this work, and to SUNY Brockport for lab GPU access and summer research support.

## References

- A. Radford, L. Metz, and S. Chintala. "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." ICLR, 2016. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
- M. Arjovsky, S. Chintala, and L. Bottou. "Wasserstein Generative Adversarial Networks." ICML, 2017. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
- N. Inkawhich. ["DCGAN Tutorial."](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) PyTorch Tutorials. The starting point for this code.
- D. S. Kermany et al. "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." Cell 172(5):1122–1131, 2018.
- ["UCSD 3-class labeled retinal OCT images."](https://www.kaggle.com/datasets/mmazizi/ucsd-3-class-labeled-retinal-oct-images) Kaggle dataset (mmazizi).
