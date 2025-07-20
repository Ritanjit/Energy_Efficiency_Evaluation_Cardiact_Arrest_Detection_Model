# <picture><source srcset="https://fonts.gstatic.com/s/e/notoemoji/latest/1faab/512.webp" type="image/webp"><img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1fac0/512.gif" alt="🫀" width="40" height="32"></picture> Energy Efficiency Evaluation of Cardiac Arrest Early Detection Models 


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uRpSNWPYPE__Nhhtnl6-anwPlyJU5OGz#scrollTo=vWpy8lJWN45l)
[![Python](https://img.shields.io/badge/Python_3.10_+-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Gradio](https://img.shields.io/badge/Gradio-3.50+-orange?logo=gradio)](https://www.gradio.app)
[![Hugging Face Spaces](https://img.shields.io/badge/Deploy-HF_Spaces-blue?logo=huggingface)](https://huggingface.co/spaces)
[![CodeCarbon](https://img.shields.io/badge/CodeCarbon-Tracking-green)](https://github.com/mlco2/codecarbon)


> **Project Description:**
> This project implements an end-to-end pipeline for the **early detection of cardiac arrest** by classifying ECG signals. The system utilizes a deep learning (1D CNN) architecture and offers lightweight model versions (Distilled and Quantized) optimized for deployment on resource-constrained edge devices. A key component of this work is the evaluation of energy consumption and carbon footprint, demonstrating the efficiency gains from model compression.
##


## <picture><source srcset="https://fonts.gstatic.com/s/e/notoemoji/latest/2696_fe0f/512.webp" type="image/webp"><img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1fac0/512.gif" alt="🫀" width="35" height="30"></picture> Dataset

* **Source:** [MIT-BIH Arrhythmia Dataset](https://physionet.org/content/mitdb/1.0.0/)
* **Preprocessing:**

  * Denoising with Discrete Wavelet Transform (DWT)
  * Z-score normalization
  * Window slicing around R-peak (360 samples)
 
* **Classes:**

  * `N`: Normal beat
  * `L`: Left bundle branch block beat
  * `R`: Right bundle branch block beat
  * `A`: Atrial premature beat
  * `V`: Ventricular ectopic beat
##


## 🧠 Models

Three versions were developed to balance accuracy and energy efficiency:

#### <picture><source srcset="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4a1/512.webp" type="image/webp"><img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2705/512.gif" alt="✅" width="18" height="15"></picture> **Original CNN (Teacher Model):** 1D CNN trained for high accuracy.
#### <picture><source srcset="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4a1/512.webp" type="image/webp"><img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2705/512.gif" alt="✅" width="18" height="15"></picture> **Distilled CNN (Student Model):** Smaller model trained via knowledge distillation.
#### <picture><source srcset="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4a1/512.webp" type="image/webp"><img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2705/512.gif" alt="✅" width="18" height="15"></picture> **Quantized CNN (TFLite):** Post-training INT8 quantized version optimization.

| Model         | Accuracy | Latency (ms) | Size (MB) |
| ------------- | -------- | ------------ | --------- |
| Original CNN  | 98.8%    | 68.17        | 1.16      |
| Distilled CNN | 85.8%    | 83.18        | 0.05      |
| Quantized CNN | 98.8%    | 0.34         | 0.57      |

**Best Trade-off:** Quantized model - high accuracy, lowest size & latency.
##

## <picture><source srcset="https://fonts.gstatic.com/s/e/notoemoji/latest/1f3af/512.webp" type="image/webp"><img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f3af/512.gif" alt="🎯" width="25" height="25"></picture> Energy Efficiency Evaluation

<table>
<tr>
<td align="center">

#### MODEL BENCHMARKS

</td>
</tr>

<tr>
<td>

  <img src="https://raw.githubusercontent.com/Ritanjit/Lightweight_Model_Early_Detection_Cardiac_Arrest/main/model_comparision.png" width="1100"/>

   _Evaluated using FLOPs and energy consumption per inference on Tesla T4 GPU._

</td>
</tr>
</table>

| Model         | FLOPs (per inf) | Energy/inf (J) | kWh (1000 inf) | CO₂e (kg) |
| ------------- | --------------: | -------------: | -------------: | --------: |
| Original CNN  |      23,038,511 |      1.04×10⁻⁵ |       0.000238 |  0.000068 |
| Distilled CNN |          91,902 |      4.14×10⁻⁸ |       0.000039 |  0.000011 |
| Quantized CNN |       4,200,000 |      1.89×10⁻⁶ |       0.000025 |  0.000007 |

See `model_comparison.png` for benchmark plots.



## ⚙️ How to Use

### 🔧 Setup

```bash
# Clone the repository
git clone https://github.com/Ritanjit/Lightweight_Model_Early_Detection_Cardiac_Arrest.git
cd Lightweight_Model_Early_Detection_Cardiac_Arrest

# Create and activate a conda environment
conda create -n cardiac python=3.10
conda activate cardiac

# Install required packages
pip install -r requirements.txt
```

### 💻 Run Locally

```bash
# Launch Gradio app
python app.py
```

---

## ✅ Deployment Ready

The `.tflite` model supports:

* Mobile Devices (Android/iOS via TensorFlow Lite)
* Microcontrollers (via TFLite for Microcontrollers)
* Edge Devices (e.g., Raspberry Pi, Jetson)

---

## 🧪 How to Test

Upload or paste a **360-point ECG beat** (as single-column CSV) in the Gradio app.
Model will preprocess, normalize, and classify it instantly.

---

## 🌍 Future Work

* Support additional arrhythmia types
* Integrate real-time ECG streaming
* FastAPI backend for public inference API
* Edge TPU compilation support

---

## 🙏 Acknowledgements

* **Dataset:** [MIT-BIH Arrhythmia Dataset - PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
* **Tools:** [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization), [Gradio](https://gradio.app)
* **Mentors:** Dr. Manojit Ghose, Panchanan Nath (IIIT Guwahati)

---

<div align="center">
  Made with ❤️‍🔥 by [Ritanjit Das](https://github.com/ritanjit)
</div>


---

<div align="center">

Made with <picture><source srcset="https://fonts.gstatic.com/s/e/notoemoji/latest/2764_fe0f_200d_1f525/512.webp" type="image/webp"><img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2705/512.gif" alt="❤" width="25" height="25"></picture> by [Ritanjit Das](https://github.com/ritanjit)

</div>
