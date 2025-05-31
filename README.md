# Slime Chunk Finder

**Slime Chunk Finder** is an advanced tool designed to locate and visualize the best **slime chunks** in **Minecraft**. It uses optimized algorithms and GPU acceleration to perform fast and precise calculations, helping players identify the most efficient areas to build slime farms.

## Features

- **User-Friendly Interface**:
  - Simple configuration for world seed and scan radius.
  - Interactive visualization of the slime chunk map.
  - Options to save and load coordinates.

- **Advanced Optimization**:
  - **GPU Acceleration**: Uses CUDA and Numba for fast calculations.
  - **Multithreaded Processing**: Optimized alternative for systems without a GPU.

- **Interactive Visualization**:
  - Detailed map with zoom, grid toggling, and options to save as PNG.
  - Dynamic coordinates based on mouse position on the map.

- **Resource Management**:
  - Cancel ongoing searches.
  - Automatic resource cleanup when closing the application.

## Requirements

### Software
- **Python 3.8 or higher**.
- External libraries:
  - `numpy`
  - `matplotlib`
  - `numba`

### Hardware
- **CUDA-compatible GPU** (optional for GPU acceleration).

## Installation

1. Clone this repository or download the exe.
2. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt

