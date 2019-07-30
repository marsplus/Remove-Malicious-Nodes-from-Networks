# Remove-Malicious-Nodes-from-Networks
Code to replicate the experimental results in paper "Remove Malicious Nodes from Networks" @ AAMAS 2019

### Installation
You need **Python3.7** and conda to install required packages. 
1. First, clone the project folder to your computer.
2. Then, create a conda environment and activate it:
  ```
  conda create -n MINT python=3.7
  conda activate MINT
  ```
3. After the environment is activated, install the following required packages:
   ```
   conda install numpy scipy pandas scikit-learn seaborn matplotlib networkx 
   pip install cvxpy
   pip install cvxopt
   ```
   
### Generate results
1. Inside the project folder, create a folder to store experimental outputs:
```
mkdir result/
```

2.  Under the root directory of the project folder, run the following script to generate experimental results in the paper:
```
./exp.sh
```

3. After generating all the results, an IPython notebook named plot.ipynb in src/ is provided to draw the box plots in the paper.

### Reference
```
@inproceedings{tong2018adversarial,
  title={Adversarial Regression with Multiple Learners},
  author={Tong, Liang and Yu, Sixie and Alfeld, Scott and others},
  booktitle={International Conference on Machine Learning},
  pages={4953--4961},
  year={2018}
}
```
