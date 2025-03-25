# 🎨 ComicCrafter AI

**ComicCrafter AI** is an innovative AI-powered tool that transforms your text prompts into vibrant 4-panel comic strips. Powered by advanced machine learning models like DistilGPT-2, Stable Diffusion, and ControlNet, this project combines storytelling and visual art to bring your ideas to life. Whether you're a comic enthusiast, a storyteller, or just exploring AI creativity, ComicCrafter AI makes comic creation effortless and fun!

---

## ✨ Features
- **📝 Automated Story Generation**: Generates a 4-part comic storyline (Introduction, Development, Climax, Conclusion) using DistilGPT-2.
- **🎨 Stunning Visuals**: Creates detailed, high-quality comic panels with Stable Diffusion, enhanced by ControlNet for sketch-based control.
- **🔧 User-Friendly Interface**: Built with Gradio, offering an interactive, customizable web UI with adjustable settings.
- **🖌️ Customizable Content**: Supports user-uploaded sketches and a JSON-based character library for recurring characters.
- **⚡ Optimized Performance**: Supports both GPU and CPU environments with memory-efficient model handling.

---

## 🛠️ Tech Stack
- **🤖 DistilGPT-2**: For generating dynamic comic storylines.
- **🖼️ Stable Diffusion 2.1**: For rendering high-quality comic images.
- **✏️ ControlNet (Canny)**: Integrates user sketches into image generation for enhanced creative control.
- **🖥️ Gradio**: Provides an intuitive, web-based user interface.
- **🔥 PyTorch**: Backend framework for model handling and GPU/CPU optimization.
- **📸 OpenCV, Pillow, NumPy**: For image manipulation and array processing.
- **🚀 Accelerate**: Ensures optimized model loading and performance.

---

## 📋 Prerequisites
- **🐍 Python**: 3.10 or higher
- **💻 Hardware**: CUDA-enabled GPU (recommended for faster processing) or CPU
- **💾 Disk Space**: ~10-15 GB for pre-trained models
- **📌 Pre-trained Models** (stored locally):
  - **DistilGPT-2**: `models/distilgpt2`
  - **Stable Diffusion 2.1**: `models/stable-diffusion-2-1-base`
  - **ControlNet (Canny)**: `models/controlnet-canny`

---

## 🚀 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/him4n-shu/comic_crafter_ai_using_LLM
cd comic-crafter-ai
```

### 2️⃣ Create and Activate Virtual Environment
```bash
python -m venv comic_env
source comic_env/bin/activate   # For Mac/Linux
# or
.\comic_env\Scripts\Activate.ps1   # For Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up Models
Ensure the pre-trained models are correctly placed in the following directories:
- **DistilGPT-2**: `models/distilgpt2`
- **Stable Diffusion 2.1**: `models/stable-diffusion-2-1-base`
- **ControlNet (Canny)**: `models/controlnet-canny`

### 5️⃣ Run the Application
```bash
python app.py
```

---

## 🎉 Get Creative!
Once running, open the web interface in your browser, input your prompts, and watch ComicCrafter AI bring your ideas to life in comic form. Happy comic crafting!

## 📜 License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).  
- You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of this software, as long as you include the original copyright notice and this permission notice in all copies or substantial portions of the software.