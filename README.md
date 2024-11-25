# UDT 
UDT: Uncertainty-Informed Dynamic Threshold

A key challenge in anomaly detection lies in setting an optimal threshold for anomaly scores to distinguish anomalies from normal data. In this paper, we propose UDT, which dynamically adjusts the thresholds using uncertainty of each sample.
The main contributions of this paper are summarized as follows:
- We improve existing methods for uncertainty quantification in anomaly detection by addressing potential scale differences between aleatoric and epistemic uncertainties. Our approach adjusts the scale of these uncertainties and performs a weighted sum, ensuring accurate representation of both types of uncertainties. 
- We present the first attempt to construct a dynamic thresholding method using uncertainty quantification, applicable to a wide range of domains. Our method demonstrates performance improvements in time series anomaly detection by more effectively identifying misclassified samples near the fixed threshold, using uncertainty as a measure of the ambiguity of each sample.

![Structure](https://github.com/user-attachments/assets/9e418c70-e62d-4859-9adb-ec01993b5843)


Especially, we use anomaly transformer proposed by [Xu et al, 2021](https://arxiv.org/abs/2110.02642) as backbone model with benchmark datasets. Therefore, we used [their repository](https://github.com/thuml/Anomaly-Transformer) as the base framework for this work.

## Contact
If you have any questions, please contact jungmin9195@korea.ac.kr
