# Breast-cancer-diagnosis-using-Machine-Learning
Machine learning is widely used in bio informatics and particularly in breast cancer diagnosis.
In this project, we have used certain classification methods such as K-nearest neighbors (K-NN) and Support Vector Machine (SVM) which is a supervised learning method to detect breast cancer. 
Cancer diagnosis is one of the most studied problems in the medical domain.
Several researchers have focused in order to improve performance and achieved to obtain satisfactory results.
Early detection of cancer is essential for a rapid response and better chances of cure. 
Unfortunately, early detection of cancer is often difﬁcult because the symptoms of the disease at the beginning are absent.
Thus, it is necessary to discover and interpret new knowledge to prevent and minimize the risk adverse consequences.


To understand this problem more precisely, tools are needed to help oncologists to choose
the treatment required for healing or prevention of recurrence by reducing the harmful effects of certain treatments and their costs. 
In artiﬁcial intelligent, machine learning is a discipline which allows the machine to evolve through a process.
Wisconsin Diagnostic Breast Cancer (WDBC) dataset obtained by the university of Wisconsin Hospital is used to classify tumors as benign or malignant.

#k- Nearest Neighbour (k-NN) classification technique:


k-NN is a non- parametric method used for classification. In this classification, the output is a class membership.
An object is classified by a majority vote of its neighbors,
with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). 
If k = 1, then the object is simply assigned to the class of that single nearest neighbor. It is the simplest algorithm among all the machine learning algorithms.

#Support Vector Machine (SVM) classification Technique:


Support Vector Machine (SVM) is a supervised machine learning algorithm which can be used for both  classification or regression challenges. However, it is mostly used in classification problems. In this algorithm, we plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane that differentiate the two classes very well

#Outcome


1.	It can be seen that as the training data size increases, SVM performs better than kNN and has more accuracy.

2.	kNN is quite a good classifier but its performance depends on the value of k. It gives poor results for lower values of k and best results as the value of k increases.

3.	PCA is more sensitive to SVM than kNN .As the value of Principle Component (PC) is increased, SVM gives better results and accuracy score is more than kNN.

4.	When the value of PC=1 and K=9, we get the highest accuracy score (97.95).

