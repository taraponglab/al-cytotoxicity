# Molecular Active Learning for Predicting Skin Cytotoxicity

This repository contains the source code and dataset for the paper "Molecular Active Learning Approaches for Predicting Skin Cytotoxicity", published in *Computers in Biology and Medicine*.

## Authors

-   **Sastiya Kampaengsri**¹
-   **Darlene Nabila Zetta**²
-   **Andi Endang Kusuma Intan**³
-   **Huynh Anh Duy**³,⁴
-   **Tarapong Srisongkram**¹

### Affiliations

¹ Division of Pharmaceutical Chemistry, Faculty of Pharmaceutical Sciences, Khon Kaen University, Khon Kaen, 40002, Thailand  
² Graduate School in the Program of Pharmaceutical Sciences, Faculty of Pharmaceutical Sciences, Khon Kaen University, Khon Kaen, 40002, Thailand  
³ Graduate School in the Program of Research and Development in Pharmaceuticals, Faculty of Pharmaceutical Sciences, Khon Kaen University, Khon Kaen, 40002, Thailand  
⁴ Department of Health Sciences, College of Natural Sciences, Can Tho University, Vietnam

### Corresponding Author

-   **Tarapong Srisongkram**: `tarasri@kku.ac.th`

---

## Graphical Abstract

![Graphical Abstract](./assets/Abstract.png)

---

## Installation

It is recommended to create a dedicated Conda environment to run this project using the provided environment file.

```bash
# Create and activate the conda environment
conda env create -f environment_cpu.yml
conda activate ai-drugdiscovery
```

## Requirements

The project dependencies are listed in the `environment_cpu.yml` file. The main packages include:

-   Python 3.11
-   PyTorch
-   TensorFlow / Keras
-   RDKit
-   Scikit-learn
-   modAL (for Active Learning)
-   PyTorch Geometric (for GNNs)
-   NumPy
-   Pandas
-   Matplotlib / Seaborn
-   XGBoost
-   UMAP

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Dr. Tarapong Srisongkram

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.