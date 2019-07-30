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
@inproceedings{yu2019removing,
  title={Removing malicious nodes from networks},
  author={Yu, Sixie and Vorobeychik, Yevgeniy},
  booktitle={Proceedings of the 18th International Conference on Autonomous Agents and MultiAgent Systems},
  pages={314--322},
  year={2019},
  organization={International Foundation for Autonomous Agents and Multiagent Systems}
}
```
