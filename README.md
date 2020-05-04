# LEt-SNE: A Hybrid Approach To Data Embedding And Visualization of Hyperspectral Bands In Satellite Imagery
Published in the 45th IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2020.<br>
DOI (ICASSP Publication):  https://doi.org/10.1109/ICASSP40776.2020.9053924<br>
DOI (Code Ocean): https://doi.org/10.24433/CO.7476989.v1<br>


Authors:
- [Megh Shukla](https://linkedin.com/in/megh-shukla), Machine Learning Engineer, MBUX Intelligent Interior, Mercedes-Benz Research and Development India
- [Biplab Banerjee](https://biplab-banerjee.github.io/index.html), Assistant Professor, Centre of Studies in Resources Engineering, Indian Institute of Technology Bombay
- [Krishna Mohan Buddhiraju](http://www.csre.iitb.ac.in/bkmohan/), Professor, Centre of Studies in Resources Engineering, Indian Institute of Technology Bombay

This repository contains the code needed to reproduce the experiment results in the published paper.
In this paper, we attempt to address the **curse of dimensionality**, a phenomenon plaguing high dimensional datasets.
We also identify two subproblems within dimensionality reduction: Data Visualization and Clustering.
To address the Curse of Dimensionality, we introduce a new term, that we call the **Compression Factor**.

## Organization

The repository contains files that can be used to reproduce the results in the paper. A short description of the contents of the repository follows:
- _Thesis_LEt-SNE.pdf_: Master's Thesis submitted to IIT Bombay, 2019. Contains a list of various experiments and detailed discussion on the algorithm.
- _LEt-SNE_arXiv.pdf_: Accepted version of the paper to be presented in ICASSP 2020.
- *LEt-SNE_requirements.txt*: Contains list of packages needed to be installed to run the experiment.
- _Dataset.rar_: Contains the datasets for which results have been reported: 1) Salinas 2) Indian Pines 3) Pavia University
- *Results.xlsx*: Contains the results reported in the paper.
- *Compression_Factor.ipynb*: Colab notebook for experimental validation of the Compression Factor.
- *LEt_SNE_ICASSP2020.ipynb*: Colab notebook for experimenting with LEt-SNE.

## Experimentation

To experiment with the algorithm, we recommend opening the notebook with Google Colab. 
Upload the files: Dataset.rar as well as LEt-SNE_requirements.txt in Colab, and run the notebook cells.

To choose a dataset for experimenting, a small change needs to be made in two function calls, where ```<dataset>``` is {salinas, indian_pines, pavia}:
```
load_data(<dataset> = True)
segment_image(<dataset> = True)
```

To switch between different modes of operation, assign them to be True. Eg: MANIFOLD = True, SEGMENTATION = True or LABEL = True.
Ensure to assign exactly one of them to be True.
The training / testing split ratio can be changed with the global constant ```TRAIN_SPLIT = 0.5```.
Other settings/hyperparameter choices are described in the notebook.

## Contact

The author, Megh Shukla can be contacted via:
```
- e-mail: work.meghshukla@gmail.com
- linkedin: www.linkedin.com/in/megh-shukla/
```
