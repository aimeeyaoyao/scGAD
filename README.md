# scGAD: a new task and end-to-end framework for generalized cell type annotation and discovery.
# Introduction
Here, we propose a new, realistic, and challenging task called generalized cell type annotation and discovery in the single-cell annotation field. To effectively tackle this task, we propose a novel end-to-end algorithmic framework called scGAD. Moreover, we design the comprehensive comparison baselines and evaluation benchmarks to validate the practicality of scGAD, and the experimental results on simulation and real datasets demonstrate that scGAD outperforms other state-of-the-art annotation and clustering methods under the three kinds of accuracy. 

The input of the scGAD model is the mixed reference data and target data, the rows represent the cells and the columns represent the genes. We supply the data preprocessing, network architecture, algorithm running in the corresponding python files. With scGAD, you can transfer the cell type labels from reference data to target data and generate the clustering labels within the target data. 
# Requirement
The version of python environment and packages we used can be summarized as follows,

python environment >=3.6

torch >=1.10.2

scanpy 1.4.4

scikit-learn 0.20.4

scipy 1.1.0

jgraph 0.2.1

tqdm 4.64.1

...

Please build the corresponding operation environment before runing our codes.
# Example and Quickstart
We provide some explanatory descriptions for the codes, please see the specific code files. We supply three kinds of training codes for simulation data, cross data and single data, respectively. Specifically, for simulation experiment, we use the balanced setting as an example. These datasets have two batches and each batch has 8 cell types. You just need to focus on the train_simu.py file. Defaultly, we remove 4 cell types from one batch by parameter "removal". You can run the following code in your command lines:

python train_simu.py

Then you will obtain the ten results, and the average seen (annotation) accuracy is 98.8, the average novel (clustering) accuracy is 96.9, the overall (clustering) accuracy is 97.9. Considering the running time of ten datasets, you can also train one dataset by modifying the main file. 

For cross data example, we use the placenta tissue as an example and you can download the data from <a href="https://cblast.gao-lab.org/ALIGNED_Homo_sapiens_Placenta/ALIGNED_Homo_sapiens_Placenta.h5">Placenta</a>. Then you put the data file into the "data/real_data/ALIGNED_Homo_sapiens_Placenta" folder. We first use the Vento-Tormo 10x as reference data and the Vento-Tormo Smart-seq2 as target data. You just need to focus on the train_cross.py file. And you can run the following code in your command lines:

python train_cross.py

Then you will get the seen accuracy 98.8, novel accuracy 80.5 and overall accuracy 92.4. Besides, you can also use the Vento-Tormo Smart-seq2 as reference data and the Vento-Tormo 10x as target data. For it, you just need to set the "num" parameter as 5. 

For single data example, we use the Chen dataset as an example and you can download the data from <a href="https://cblast.gao-lab.org/Chen/Chen.h5">Chen</a>. Then you put the data file into the "data/real_data/Chen" folder. We set the labeled ratio as 0.5 defaultly and you can modify it by changing the "ratio" parameter. You just need to focus on the train_single.py file. And you can run the following code in your command lines:

python train_single.py

Then you will get the seen accuracy 98.3, novel accuracy 93.0 and overall accuracy 94.1. For other case studies, you can input different parameters to achieve them.

# Reference
Our paper is submitted to Briefings in Bioinformatics and the specific details will come soon. Please consider citing it.
# Contributing
Author email: zhaiyuyao@stu.pku.edu.cn. If you have any questions, please contact with me. 
