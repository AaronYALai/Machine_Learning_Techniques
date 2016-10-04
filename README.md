Machine Learning
========

About
--------

Implementations and homeworks of two MOOCs courses(offered by [**Hsuan-Tien Lin**](http://www.csie.ntu.edu.tw/~htlin/)):
- **Machine Learning Foundations**: 
    * Corresponds to the first half-semester of the National Taiwan University (NTU) course "Machine Learning".
    * Introduce topics ranging from "When Can Machines Learn" to "Why", "How" and beyond.

- **Machine Learning Techniques**:
    * The second half-semester of the NTU course "Machine Learning".
    * Three popular tools:
        1. embedding numerous features (kernel models, such as support vector machine)
        2. combining predictive features (aggregation models, such as adaptive boosting)
        3. distilling hidden features (extraction models, such as deep learning).

courses are based on the textbook [Learning from Data: A Short Course](http://amlbook.com/).

Syllabus
--------

### Machine Learning Foundations

When Can Machines Learn?
- The Learning Problem  |  Learning to Answer Yes/No
- Types of Learning  |  Feasibility of Learning

Why Can Machines Learn?
- Training versus Testing  |  Theory of Generalization
- The VC Dimension  |  Noise and Error

How Can Machines Learn?
- Linear Regression  |  Linear 'Soft' Classification
- Linear Classification beyond Yes/No  |  Nonlinear Transformation

How Can Machines Learn Better?
- Hazard of Overfitting  |  Preventing Overfitting I: Regularization
- Preventing Overfitting II: Validation  |  Three Learning Principles

### Machine Learning Techniques

Embedding Numerous Features
- Linear Support Vector Machine  |  Dual Support Vector Machine
- Kernel Support Vector Machine  |  Soft-Margin Support Vector Machine
- Kernel Logistic Regression  |  Support Vector Regression

Combining Predictive Features
- Bootstrap Aggregation | Adaptive Boosting
- Decision Tree | Random Forest
- Gradient Boosted Decision Tree

Distilling Hidden Features
- Neural Network  |  Deep Learning
- Radial Basis Function Network  |  Matrix Factorization

Content
--------
- Kernel SVM & Soft-Margin SVM
- Kernel Logistic Regression and Support Vector Regression
- Blending and Bagging
- Adaptive Boosting
- Decision Tree and Random Forest
- Gradient Boosted Decision Tree
- kMeans, k-Nearest Neighbors
- Radial Basis Function Network
- Neural Network and Deep Learning
- Autoencoder

Usage
--------
Clone the repo and use the [virtualenv](http://www.virtualenv.org/):

    git clone https://github.com/AaronYALai/Machine_Learning_Techniques

    cd Machine_Learning_Techniques

    virtualenv venv

    source venv/bin/activate

Install all dependencies and run the model:

    pip install -r requirements.txt

    cd Autoencoder

    python autoencoder.py


Certificate
--------
#### Machine Learning Foundations - [Course Certificate](https://www.coursera.org/account/accomplishments/records/2XGEscUkTTJKRtGU)
#### Machine Learning Techniques - [Course Certificate](https://www.coursera.org/account/accomplishments/verify/X8BGEERTNT)
