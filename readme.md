# LAG-Pytorch
Official pytorch codes and models for paper:

Jie Yang, Yang Luo, Chunbo Luo. FedTC: An Effective Federated Learning to Alleviate Model Forgetting and Model Drifting for Gastritis Detection. Submission to IEEE Journal of Biomedical and Health Informatics.

# Pretrained Model & Datasets
To view a sample of the data: [data-samples](https://github.com/fengcherenxi/LAG/tree/main/data)

To download data please contact: juanliao@scu.edu.cn && c.luo@uestc.edu.cn
# Method
![](https://github.com/fengcherenxi/FedTC-pytorch/blob/main/resources/TopN.png)
The process of updating $Top_N$.

![](https://github.com/fengcherenxi/FedTC-pytorch/blob/main/resources/CLD.png)
# Experiments Results
![](https://github.com/fengcherenxi/FedTC-pytorch/blob/main/resources/Results.png)
A schematic diagram of the difference in feature distributions between the training and testing sets for models updated with and without CLD. Mean difference represents the magnitude of the difference in feature distribution between the training and testing sets. Variance difference represents the magnitude of the difference in feature distribution variances between the training and testing sets.
# Requirements
```
pytorch==1.7+cuda10.1
torchvision==0.6.0
numpy==1.19.5
```
# Citation
```
Jie Yang, Yang Luo, Chunbo Luo. FedTC: An Effective Federated Learning to Alleviate Model Forgetting and Model Drifting for Gastritis Detection [J].
```
