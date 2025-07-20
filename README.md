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

 _Evaluated using FLOPs (energy consumption per inference on Tesla T4 GPU) and CodeCarbon._

<table>

<tr>
<td>

  <img src="https://raw.githubusercontent.com/Ritanjit/Energy_Efficiency_Evaluation_Cardiact_Arrest_Detection_Model/main/Flops1.png" width="1000"/>

</td>
<td>

  <img src="https://raw.githubusercontent.com/Ritanjit/Energy_Efficiency_Evaluation_Cardiact_Arrest_Detection_Model/main/Flops2.png" width="1100"/>

</td>
</tr>

<tr>
<td colspan="2" align="center">

 <img src="https://raw.githubusercontent.com/Ritanjit/Energy_Efficiency_Evaluation_Cardiact_Arrest_Detection_Model/main/CodeCarbon.png" width="1100"/>

</td>
</tr>

</table>

| Model         | FLOPs (per inf) | Energy/inf (J) | kWh (1000 inf) | CO₂e (kg) |
| ------------- | --------------: | -------------: | -------------: | --------: |
| Original CNN  |      23,038,511 |      1.04×10⁻⁵ |       0.000238 |  0.000068 |
| Distilled CNN |          91,902 |      4.14×10⁻⁸ |       0.000039 |  0.000011 |
| Quantized CNN |       4,200,000 |      1.89×10⁻⁶ |       0.000025 |  0.000007 |
##


## <picture><source srcset="https://fonts.gstatic.com/s/e/notoemoji/latest/2699_fe0f/512.webp" type="image/webp"><img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2699_fe0f/512.gif" alt="⚙" width="20" height="20"></picture> How to Use

### 🔧 Setup

```bash
conda create -n cardiac python=3.10
conda activate cardiac
pip install -r requirements.txt
```

### 💻 Run Locally

```bash
python app.py     # Launch Gradio interface locally
```

### ☁️ Deploy Permanently

```bash
gradio deploy     # Uploads to Hugging Face Spaces
```
##


## <picture><source srcset="https://fonts.gstatic.com/s/e/notoemoji/latest/1f3c1/512.webp" type="image/webp"><img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2705/512.gif" alt="✅" width="20" height="20"></picture> Deployment Ready

The quantized `.tflite` model can be deployed to:

* Mobile (via TensorFlow Lite)
* Microcontrollers (with Edge Impulse / TinyML)
* Raspberry Pi / NVIDIA Jetson
##


## 🧪 How to Test

Simply paste or upload a **360-point ECG beat** into the Gradio app. The model will preprocess, normalize, and classify the beat in real-time.
##


## 🔭 Future Work

* Add support for more heartbeat types
* Convert to TFLite with Edge TPU compatibility
* Real-time ECG streaming integration
* API deployment via FastAPI / Streamlit
##


## <picture><source srcset="https://fonts.gstatic.com/s/e/notoemoji/latest/1faf1_1f3fc_200d_1faf2_1f3fe/512.webp" type="image/webp"><img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2705/512.gif" alt="✅" width="20" height="20"></picture> Acknowledgements

* [MIT-BIH Dataset - PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
* [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)
* [Gradio](https://gradio.app/)

---


<div align="center">

Made with <picture><source srcset="https://fonts.gstatic.com/s/e/notoemoji/latest/2764_fe0f_200d_1f525/512.webp" type="image/webp"><img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2705/512.gif" alt="❤" width="25" height="25"></picture> by [Ritanjit Das](https://github.com/ritanjit)

</div>
