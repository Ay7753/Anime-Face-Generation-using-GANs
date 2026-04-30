# 🎨 Anime Face Generation using GANs

## 📌 Objective

The goal of this project is to build a Generative Adversarial Network (GAN) using PyTorch to generate realistic anime face images by learning from a dataset.

---

## 📂 Dataset

* **Name:** Anime Face Dataset
* **Source:** Kaggle
* **Description:** A collection of anime face images used to train the GAN model.

### 📁 Dataset Structure

```
data/
   Anime_Face_Dataset/
      faces/
         image1.jpg
         image2.jpg
         ...
```

---

## ⚙️ Preprocessing

* Resize images to **64×64**
* Center crop images
* Convert to tensor
* Normalize pixel values to **[-1, 1]**

---

## 🧠 Model Architecture

### 🔹 Generator

* Input: Random noise vector (size = 100)
* Layers:

  * ConvTranspose2D
  * BatchNorm
  * ReLU
* Output: 64×64 RGB image
* Final activation: **Tanh**

---

### 🔹 Discriminator

* Input: 64×64 image
* Layers:

  * Conv2D
  * BatchNorm
  * LeakyReLU
* Output: Probability (real/fake)
* Final activation: **Sigmoid**

---

## 🔁 Training Process

1. Train Discriminator on real images
2. Train Discriminator on fake images
3. Train Generator to fool Discriminator
4. Repeat for multiple epochs

---

## ⚙️ Training Configuration

* **Framework:** PyTorch
* **Epochs:** 5
* **Batch Size:** 32
* **Learning Rate:** 0.0002
* **Optimizer:** Adam
* **Loss Function:** Binary Cross Entropy Loss

---

## 📊 Results

### 🖼 Generated Images

Images generated at each epoch:

```
output_epoch_0.png
output_epoch_1.png
output_epoch_2.png
output_epoch_3.png
output_epoch_4.png
```

---

### 📈 Loss Curve

Loss graph generated using matplotlib:

```
loss_curve.png
```

* Shows Generator vs Discriminator loss
* Helps analyze training stability

---

## 📈 Evaluation

The model is evaluated visually based on generated images.

### Observations:

* Faces are recognizable after training
* Some images appear slightly blurry
* Diversity improves with more epochs

⚠️ Advanced metrics like **FID Score** were not implemented due to computational limitations.

---

## ⚠️ Challenges Faced

* GAN training instability
* Mode collapse (limited diversity)
* Initial blurry outputs

---

## 🚀 Future Improvements

* Increase number of epochs
* Use advanced GAN architectures (StyleGAN, WGAN)
* Implement evaluation metrics (FID Score)
* Tune hyperparameters for better quality

---

## ▶️ How to Run

### 1. Install dependencies

```
pip install torch torchvision matplotlib
```

---

### 2. Run the model

```
python gan_anime_faces.py
```

---

### 3. Output Files

After execution, the following files will be generated:

```
output_epoch_*.png
loss_curve.png
generator.pth
discriminator.pth
```

---

## 📁 Project Structure

```
project/
│
├── gan_anime_faces.py
├── README.md
├── loss_curve.png
├── output_epoch_0.png
├── output_epoch_1.png
├── output_epoch_2.png
├── output_epoch_3.png
├── output_epoch_4.png
├── generator.pth
├── discriminator.pth
│
└── data/
    └── Anime_Face_Dataset/
        └── faces/
```

---

## ✅ Conclusion

This project successfully demonstrates the use of GANs for generating anime face images. While the results are promising, further improvements can enhance realism and diversity.

---
