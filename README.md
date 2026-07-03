<div align="center">
  <img src="static/character-new.png" alt="SIMEMEK Logo" width="200"/>
  
  # SIMEMEK
  **Smart Image Metadata Extraction Mapping Engine Kit**

  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.103.1-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

  *A collaborative, retro 8-bit themed web UI engine for building, mapping, and downloading AI-ready geodata datasets.*
</div>

---

## 📖 Overview

**SIMEMEK** is a powerful yet beautifully designed dataset compilation tool aimed at simplifying the creation of segmentation datasets (specifically for models like **SegFormer**). It provides an interactive map interface for users to define bounding boxes, automatically fetches high-resolution ESRI satellite imagery, and processes vector data into training-ready masks.

Built with a unique **1-bit / retro RPG aesthetic**, SIMEMEK supports multi-user collaboration in real-time, making dataset collection both efficient and incredibly fun.

## ✨ Features

- **🕹️ Retro Collaborative UI**: A unique 8-bit interface where multiple users can work simultaneously. Each user gets a unique color-coded bounding box identifier.
- **🗺️ Interactive Mapping**: Integrated Leaflet.js allows precise area selection directly on the map.
- **⚡ Non-Blocking Architecture**: Features an asynchronous background ZIP packager. Generate massive dataset exports without freezing the server or interrupting other users.
- **🛰️ High-Res Satellite Integration**: Automatically fetches sharp ESRI World Imagery tiles.
- **🤖 AI-Ready**: Generates pixel-perfect 3-class masks (background, building interior, building border) optimized for semantic segmentation models.

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8 or higher
- The `IDN.fgb` (FlatGeobuf) file containing your vector data (due to its massive 2.5GB size, this is not included in the repository).

### 2. Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/akbarnurrahim/SIMEMEK.git
cd SIMEMEK
pip install -r requirements.txt
```

### 3. Setup Data
Place your `IDN.fgb` file directly in the root of the project directory.

### 4. Run the Server
Launch the FastAPI server using Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```
Navigate to `http://localhost:8080` in your web browser.

## 💻 Usage Guide
1. **Identify Yourself**: Enter your username on the landing page to receive a uniquely hashed collaboration color.
2. **Select Area**: Use the map's rectangle tool to draw a bounding box over your area of interest.
3. **Start Scraping**: Click `START`. The engine will stream real-time logs to the retro terminal in the UI.
4. **Download**: Once the data is processed, click `GET ZIP`. The backend will package the dataset asynchronously. You can continue working, and once the button lights up, your dataset is ready to download!

## 📸 Screenshots
*(Your screenshots go here! You can easily embed them later by putting images in your repository and linking them like `![Dashboard](link-gambar-anda.png)`)*

## 🛠️ Built With
- [FastAPI](https://fastapi.tiangolo.com/) - High performance backend
- [Leaflet.js](https://leafletjs.com/) - Interactive maps
- [Fiona](https://fiona.readthedocs.io/) & [Geopandas](https://geopandas.org/) - Geospatial data handling

## 📄 License
This project is licensed under the MIT License.
