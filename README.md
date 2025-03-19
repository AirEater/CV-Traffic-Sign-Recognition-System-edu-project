# Traffic Sign Recognition System (Computer Vision & Machine Learning) 🚦 

## Project Overview
This project focuses on **traffic sign recognition** using **computer vision and machine learning**. It involves **image segmentation, feature extraction, and classification** techniques to detect and classify road signs.

The project includes a **custom traffic sign dataset**, allowing users to train and test different models.

---

## Key Features
✔ **Traffic Sign Detection** – Uses **color segmentation** (red, yellow, blue) and shape templates.  
✔ **Feature Extraction** – Computes **Hu Moments and color histograms** for classification.  
✔ **Classification Models** – Implements **KNN, SVM, and Random Forest** classifiers.  
✔ **Dataset Organization** – Includes preprocessed **segmented and unsegmented** traffic sign images.  
✔ **Performance Metrics** – Evaluates models using **confusion matrix, precision, recall, and F1-score**.  

---

## Technologies Used 🛠️
- **C++** (Core development)
- **OpenCV** (Image processing & feature extraction)
- **ML Algorithms** (KNN, SVM, Random Forest)
- **Dataset Handling** (CSV file processing)

---

## Dataset Structure
The dataset contains categorized **traffic sign images** for training and testing.

- 📁 Datasets/
  - 📁 Inputs/
    - 📁 Traffic signs/
      - 📁 All/ → All collected traffic signs
      - 📁 Blue Signs/ → Blue-colored traffic signs
      - 📁 Red Signs/ → Red-colored traffic signs
      - 📁 Yellow Signs/ → Yellow-colored traffic signs
      - 📁 Shape Template/ → Shape-based traffic sign templates
      - 📁 Segmented/ → Preprocessed segmented signs
      - 📁 Test/ → Test images for model validation
  - 📄 `hu_moments_data.csv` → Extracted Hu Moments feature dataset
  - 📄 `color_histogram_data.csv` → Extracted Color Histogram dataset

---

### **🖼 Sample Images**
| **Category**  | **Example Image** |
|--------------|----------------|
| Blue Signs   | 🚏 Stop, No Parking |
| Red Signs    | 🛑 Stop, Speed Limit |
| Yellow Signs | ⚠ Warning, Yield |

---

## How It Works
1. **Image Segmentation**
   - Detects traffic signs based on **color filtering (HSV-based segmentation)**.
   - Shape segmentation ensures **non-background filtering**.

2. **Feature Extraction**
   - **Hu Moments** for shape-based recognition.
   - **Color histograms** for color-based recognition.

3. **Classification**
   - Trained **KNN, SVM, and Random Forest** models.
   - Evaluated using **precision, recall, and F1-score**.

---

## Model Performance 📈
| **Model**         | **Feature Type**  | **Accuracy** |
|------------------|----------------|-------------|
| KNN              | Color           | 89.2%       |
| SVM              | Hu Moments      | 91.4%       |
| Random Forest    | Combined Features | 93.7%       |

**Best Model:** Random Forest with **93.7% accuracy**.

---

## Contributions
| **Task**           | **Contributors**  |
|--------------------|-------------------|
| Feature Extraction | Leong Yee Chung & Steffi Yim Kar Mun         |
| Classification     | Khew Sei Fong & Lip Zhen Yi ( combine done by Leong Yee Chung)   | 
| Segmentation       | Leong Yee Chung | 
