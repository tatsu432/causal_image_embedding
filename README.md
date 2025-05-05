This repository contains the code for the final project of CPSC 452/552 - Spring Semester 2025 

1. How to run the code

You can run the main experiment by running 

```
src/main_experiment.ipynb
```
You can see the resulting table by running

```
src/main_analysis.ipynb
```


2. Required Depndencies

We run the experiment on Google Colab. You can reproduce the result by using the package version specified in 

```
requirements_colab.txt
```

3. Description of dataset

We use the modified version of [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist). We modify each image to simulate the post-treatment effect in each image. Specifically, we add an icon to each image where the following parameters are determined by the post-treatment variable.
- the type of icon
- the transparancy of the added icon
- the position of the icon
- the size of the icon are 

In addition to the modified Fashion-MNIST, we also synthetically generated the following variables.
- covariates X_i
- covariates X_i^V in images (causal inference estimators cannot access this variable)
- treatment D_i
- post-treatment variable P_i^V in images (causal inference estimators cannot access this variable)
- modified Fashion-MNIST V_i
- outcome Y_i