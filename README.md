# ECG Diagnosis Model

Myocardial infarction (MI) is the ischemic death of myocardial tissue, often causing acute coronary syndrome (ACS). ST-elevation myocardial infarction (STEMI), diagnosed by ECG elevation, is the most fatal heart attack and increases heart arrhythmia and heart failure risk. Deep learning (DL) techniques are used in medical diagnoses to categorize electrocardiogram (ECG) delineations into feature maps to detect disease-causing abnormalities. The training of features depends on the diagnosis, and severe episodes of ischemic heart disease and STEMI increase the risk of physician error. This chapter presents a diagnostic DL model with an architecture modulated using Particle Swarm Optimization (PSO), a nature-inspired optimization, to improve the identification of unhealthy classes in the data. The process done to automate the selection of the optimal architecture is called Neural Architecture Search (NAS). Various DL architectures, including ConvNetQuake, channel-based, and transfer-based ResNet50, were evaluated, with the best performing model selected for final testing. Because memory profiling is essential to ensure optimization without the risk of out-of-memory(OOM) errors, the resource consumption of each model is reported. The results demonstrated that channel-split PSO ResNet50 achieved the maximum results with a precision of 88%. Additionally, dynamic memory profiling recorded an average 72% processing power usage. The results provide insights into enhancing cardiovascular health monitoring and diagnosis thus emphasizing the link between network architecture and computational resource consumption.

## Graphical Abstract

![graph_abstract (2)](https://github.com/user-attachments/assets/bb47d14f-bc7e-4191-8625-610b6ac1afea)


## Acknowledgments
- Walaa H. Elashmawi 
  
  Faculty of Computers & Informatics, Suez Canal University, Ismailia, Egypt
  
  Faculty of Computer Science, Misr International University, Cairo, Egypt

- Ahmed Ali
  
  College of Computer Engineering and Sciences, Prince Sattam Bin Abdulaziz University, Alkharj Saudi Arabia
  
  Higher Future Institute for Specialized Technological Studies, Cairo, Egypt
