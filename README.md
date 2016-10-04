Machine Learning
========

Model implementations and homeworks of two MOOCs courses:
- **Machine Learning Foundations**: corresponds to the first half-semester of the National Taiwan University (NTU) course "Machine Learning".
- **Machine Learning Techniques**: the second half-semester of the NTU course "Machine Learning".

two courses are based on the textbook [Learning from Data: A Short Course](http://amlbook.com/).  
 
About
--------

the instructor chooses to focus on what he believes to be the core topics that every student of the subject should know. The students shall enjoy a story-like flow moving from "When Can Machines Learn" to "Why", "How" and beyond.


Syllabus
--------

### Machine Learning Foundations

When Can Machines Learn?
-- The Learning Problem
-- Learning to Answer Yes/No
-- Types of Learning
-- Feasibility of Learning

Why Can Machines Learn?
-- Training versus Testing
-- Theory of Generalization
-- The VC Dimension
-- Noise and Error

How Can Machines Learn?
-- Linear Regression
-- Linear 'Soft' Classification
-- Linear Classification beyond Yes/No
-- Nonlinear Transformation

How Can Machines Learn Better?
-- Hazard of Overfitting
-- Preventing Overfitting I: Regularization
-- Preventing Overfitting II: Validation
-- Three Learning Principles

### Machine Learning Techniques

Embedding Numerous Features
-- Linear Support Vector Machine
-- Dual Support Vector Machine
-- Kernel Support Vector Machine
-- Soft-Margin Support Vector Machine
-- Kernel Logistic Regression
-- Support Vector Regression

Combining Predictive Features
-- Bootstrap Aggregation
-- Adaptive Boosting
-- Decision Tree
-- Random Forest
-- Gradient Boosted Decision Tree

Distilling Hidden Features
-- Neural Network
-- Deep Learning
-- Radial Basis Function Network
-- Matrix Factorization

Implementation
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
At first, you should make sure you have [virtualenv](http://www.virtualenv.org/) installed.

    git clone https://github.com/AaronYALai/Machine_Learning_Techniques

after that, just cd to the folder:

    cd Machine_Learning_Techniques

Then create your virtualenv:

    virtualenv venv

Second, you need to enable the virtualenv by

    source venv/bin/activate

Install all dependencies:

    pip install -r requirements.txt

Run the model, ex.

    cd Autoencoder
    python autoencoder.py


Certificate
--------
### Machine Learning Foundations - [Course Certificate](https://www.coursera.org/account/accomplishments/records/2XGEscUkTTJKRtGU)
### Machine Learning Techniques - [Course Certificate](https://www.coursera.org/account/accomplishments/verify/X8BGEERTNT)
