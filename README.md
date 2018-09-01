# Reinforcement Learning Project

Use reinforcement learning to teach a quadcopter how to fly.

## Getting Started

Clone the repository:

``` batch
git clone https://github.com/craig-martinson/quadcopter-project.git
cd quadcopter-project
```

Create a Conda environment:

``` batch
conda create -n quadcopter python=3.6 matplotlib numpy pandas
conda activate quadcopter
 ```

Create an IPython kernel for the quadcopter environment:

``` batch
python -m ipykernel install --user --name quadcopter --display-name "quadcopter"
 ```

Open the notebook:

``` batch
 jupyter notebook Quadcopter_Project.ipynb
```

Before running code, change the kernel to match the quadcopter environment by using the drop-down menu (Kernel > Change kernel > quadcopter)
