Myocardial infarction (MI), also identifed as heart attack [1], is severe heart disease a life threat caused by blood retention as a result of the blockage of one of the coronary arteries, resulting in damage or complete death to part of muscle heart. The seizure is often a medical emergency that threatens the patient’s life and requires immediate medical attention [2]. MI is still one of the most serious diseases that may cause death. The latest statistics indicate that the death rate from MI is 30% within 30 days if it is not treated [3]. MI characteristics include abnormal Q wave appearance, STsegment elevation, and T-wave inversion [4]. The presence of abnormal Q waves on the 12-lead ECG signifes a prior transmural MI. The ST-segment changes on the standard ECG that are associated with acute ischemia or infarction are due to the fow of current across the boundary between the ischemic and non-ischemic zones referred to as injury current. Some of the T-wave changes are associated with the post reperfusion phase. Figure 1 shows an example of ECG signal in case ST elevation.
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/c4d92997-d430-4366-b17a-8f6243861b6d)
 
#### ECG Models 
##### Introduction
While the heart pumps blood to the entire body, different cardiac ion currents prevail. Those cardiac currents are acquitted by various electrodes, or leads, placed throughout the body. The acquitted data is plotted as waves, called electrocardiograms (ECGs) showing the difference in voltage potential by time.
ECGs are widely used for efficient and accurate monitoring of cardiac health. They are used by a variety of healthcare professionals and caretakers including cardiologists, physicians, emergency-room paramedics, and nurses. The accuracy of an ECG has exceeded 92% for as long as 20 years, which proves it is a longstanding method of cardiac health assessment that will continue in the future.
Professionals use manual methods to diagnose heart problems, even life-threatening ones such as Myocardial Infarction (or colloquially known “heart attacks”), from ECG signals. The main problem with manual diagnosis of heart attacks, like many other time-series data, lies in the difficulty of detecting and categorizing different waveforms and morphologies in the signal. Another less severe difficulty is the understanding of certain “hidden” patterns. For a human, a proper diagnosis is both extensively time-consuming and prone to errors. Consider life threatening conditions that account for approximately one-third of all deaths around the globe. Therefore, it is paramount to obtain the best categorization and diagnosis from ECGs.
This section focuses on models for this purpose. Numerous models have been implemented with the same purpose, but had the following drawbacks:
•	Majority of the studies conducted utilized very complex models.
•	Extremely large datasets were used which proves to be ineffective due to time and cost constraints.
•	Because the models are of high complexity, the code complexity increases and so does the hardware memory requirements.
The models and algorithms discussed in this section aim to solve the aforementioned issues by: a) separating categorization from diagnosis b) developing end to end diagnosis models that can seamlessly be optimized.
There are two main datasets used in this section for both training and testing purposes. Both datasets have 2 different forms, one form is that of a regular table of numbered features, and the other is the patient-separate form. The first dataset is the Physikalisch-Technische Bundesanstalt (PTB) Diagnostic ECG dataset. This dataset has only two classes, if it’s in the form of a table. If it’s in the form of separate patient files, it contains 9 classes. A larger version of this dataset is found as well. (Wagner & Strodthof, 2020)
The second dataset is the MIT Arrhythmia Dataset. Once again, this dataset has 2 forms. In both forms there are 5 classes. After briefing the trialed datasets, the details of our contribution algorithms are explained in further detail below.
 
###### Overview of ECG Model architecture
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/c9b5c1bf-a325-4c02-b0d0-842a3d71b938)


#### Categorization Models
##### Random Fourier Feature Generalized Learning Model for Multiclass Datasets (Goldberger AL & Amaral LAN, 2013) (Moody GB & Mark RG, 2001)
1)	Naïve Approach: Sparse Generalized Learning Model
a)	First the data is explored, and the target variables are understood.
b)	Then the data is preprocessed: nulls are removed, the data is normalized or smoothed by Gaussian discrete smoothing. Finally, to combat any imbalances, the data is resampled through a rolling window and gradients of the data are checked.
  
###### Data correlation matrix before processing
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/8d37f1b4-eb7a-4e41-a64a-96a38129b960)

 
###### PTB Data skewness before processing
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/5a055ce1-49f8-4d51-8520-c68ecf8c02bf)



###### Sample after Gaussian smoothing
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/4d9a567b-a975-4944-8881-ec6ba255aac0)
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/f0bce7c9-b9bd-48f0-a384-11049d49a320)
###### Sample after data gradients are assessed
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/476c8ba8-2e58-40f9-9330-9a5ad177373f)


c)	Fit a logistic regression with a one-versus-rest approach for multilabel classification. Regularization after fitting is done using the L2 penalty.
d)	To fully classify the data, find a sparsity matrix.
 
###### Results from MIT dataset before introducing nonlinearities
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/bc3190f1-03a1-4024-a566-fe04bb838e3a)

2)	Nonlinearities: Introducing a Random Fourier GLM
a)	Find the Euclidean distance between all data pairs with each run.
b)	Use a probabilistic algorithm that assumes data labels are unknown. The algorithm tested here is the expectation-maximization. 
c)	Find the flattened mean of the algorithm.
d)	Use Radial Basis Function (with the mean data point distance as its factor) to set random weights to the unknown classes. Fit the preprocessed gradients to the approximated labels.
 
###### Example of RBF fitting
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/c4933a5d-c66e-4930-ad28-4c919bdf07da)
 
 
###### After functional transform
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/d88e2553-74a0-44fd-8ee1-1ec5e139aa92)
 
###### Results after introduction of RBF
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/e5a50e2d-5f2a-4bcc-b7bc-b7920a3cfc7a)
 
###### Model Evaluation
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/8659e32b-f71b-41e2-8737-81a0486ac2e7)

##### Machine Learning Regressors for Binary class Datasets (Hassaballah & Wazery, 2023)
In addition to introducing nonlinearities to a heuristic, machine learning regressors can be applied to categorize ECGs. Two examples were tested below.
1)	Gaussian Naïve Bayes
a)	After the data was combined and randomly split, the data was fitted into a regressor. Training and testing were carried out to provide the following results. 
 
###### Gaussian Naïve Bayes on PTB dataset
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/f5c452e1-ae43-4bf1-9185-2666cb8ae68a)

2)	Random Forest
a)	After the data was combined and randomly split, the data was fitted into multiple decision trees. Training and testing were carried out to provide the following results. 
 
###### Random Forest on PTB dataset
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/4ddcd6b1-e3bb-4074-b516-17a5d983a444)

#### Diagnosis Models
##### Deep 1D Convolutional Neural Network   (Kachuee & Fazeli, 2018)
1. Training a Deep Convolutional Neural Network: The study describes the training of a deep convolutional neural network with residual connections for the classification of heartbeats. 
2. Transferable Representation Learning: The approach focuses on learning transferable representation from the ECG signals, enabling the acquired knowledge to be applied to other related tasks beyond the initial arrhythmia classification. This step has been simplified into a 1D CNN and will be continued further in this study.
3. Evaluation on Myocardial Infarction (MI) Classification: The method's effectiveness is evaluated in the context of MI classification, demonstrating its ability to achieve high accuracy in this task as well. 
4. Visualization and Comparison: The paper includes visualizations and comparisons to illustrate the effectiveness of the proposed approach in learning transferable representations and its performance in both arrhythmia classification and MI classification tasks.  
###### Data Annotation of Separate patient files
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/a1f9717e-814a-4947-8c32-488ead07875b)
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/e8850a67-5761-4b64-b84a-c0603938c11d)
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/82854026-83ac-4bbe-912e-e397bd9e7cf1)
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/ec8f64e8-b54e-47f1-8c27-475a5b5299ef)

 
###### Trial of 1D CNN model and its model accuracy
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/5adff15c-6c83-45c5-9194-9fc622de29d2)

##### AlexNet Deep Neural Network (Hammad & Alkinani, 2020)
This model is very similar to the past CNN model, only this model employs a longer technique and handles data imbalances algebraically. On the other hand, it also works best on binary classifications and employs binary cross entropy to predict whether the diagnosis is the same as the ground truth.
 
###### Recording example from PTB database
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/94ebaad0-2fa2-4696-a5b1-ec12adc69fff)
  
 
###### Cross entropy for a positive and a negative class.
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/62eee09d-8cb7-4123-a54c-ccde42b536d3)
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/e40b6391-b18d-405d-8e94-dc6b38d1181c)
 
###### Focal loss and its adjustment parameters
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/85643669-40a6-40c7-b32d-570179021a8d)
 
###### Main architecture of the model
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/21b7ffe2-c3fa-42ca-933d-cf8526626127)
   
###### Confusion Matrix without focal loss
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/30d9694f-e36c-48d1-8de6-aa7480b15c58)

##### ConvNetQuake Cardiologist Level Deep Neural Network (Houssein, Hassaballah, & Ibrahim, 2020)
A systematic analysis is conducted to identify critical ECG leads for accurate myocardial infarction detection. We adapted the ConvNetQuake neural network model, originally designed for earthquake identification, to achieve state-of-the-art classification results for myocardial infarction. Importantly, their multi-ECG-channel neural network achieved cardiologist-level performance without the need for manual feature extraction or data pre-processing. This approach demonstrates the potential of deep learning in providing accurate and timely diagnoses for myocardial infarction, with implications for improving healthcare support for patients with cardiovascular diseases.
 
 
###### Architecture of the model
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/74c5a390-d682-4590-9137-ab736a09e6db)
![image](https://github.com/PerfectionistAF/ECG_processing/assets/77901496/d1bbe6ee-575a-4c71-99c6-9d6689376f9f)

1)	Start by segmenting every patient separately.
2)	Then stack the model and start the forward training loop using Adam’s optimizer. Normalize after each forward feed.
3)	Split patients into batches of healthy and unhealthy. Stochastic batches were applied to this trial. (Patients were labelled according to title of their file case.)
Throughout this section, we’ve gone through a comprehensive pipeline that allows for the accurate diagnosis of heart attacks, or heart-attack causing classes. The models discussed provide valuable insights into a comparison and an analysis of each. In addition to furthering the development to encapsulate more varied datasets as well as an attempt to embed the models in frameworks or physician-friendly products.   
#### References
Fradi, M., & Khriji, L. (2021). Automatic heart disease class detection using convolutional neural network architecture-based various optimizers-networks. 

Goldberger AL, & Amaral LAN. (2013). PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. 

Hammad, M., & Alkinani, M. (2020). Myocardial infarction detection based on deep neural network on imbalanced data. 

Hassaballah, M., & Wazery, Y. (2023). ECG Heartbeat Classification Using Machine Learning and Metaheuristic Optimization for Smart Healthcare Systems. 

Houssein, E. H., Hassaballah, M., & Ibrahim, I. E. (2020). An Automatic Arrhythmia Classification Model Based on Improved Marine. 

Houssein, E., Hassaballah, M., & Ibrahim, I. E. (2020). An Automatic Arrhythmia Classification Model Based on Improved Marine. 

Kachuee, M., & Fazeli, S. (2018). ECG Heartbeat Classification: A Deep Transferable. 

Kher, R. (2019). Signal Processing Techniques for Removing Noise from ECG Signals. 

Moody GB, & Mark RG. (2001). The impact of the MIT-BIH Arrhythmia Database. 

Wagner, P., & Strodthof, N. (2020). PTB-XL, a large publicly available electrocardiography dataset. 

