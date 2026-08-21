# Video Script: Breast Cancer Detection Project

**Target Duration:** 5 Minutes
**Topic:** Automated Breast Cancer Diagnosis using Ensemble Machine Learning
**Target Audience:** Professor / Academic Evaluator

---

## [0:00 - 0:45] Introduction & Problem Statement
**Visual:** Title Slide: "Machine Learning for Social Good: Breast Cancer Detection". Transition to a simple slide showing the project's main goals.
**Audio (Student Narrator):**
"Hi Professor. Today I'll be walking you through my assignment for the 'ML for Social Good' project. I decided to focus on automated breast cancer detection. 
As we discussed in class, early detection is really the biggest factor in improving survival rates. But analyzing biopsy images manually is super time-consuming and leaves room for human error. 
So, for this project, I wanted to build a tool that could serve as a reliable 'second opinion' for pathologists. The goal was to take features extracted from breast mass cells and classify them accurately as either benign or malignant."

## [0:45 - 1:30] The Dataset & Mission
**Visual:** Screen recording showing the UCI Machine Learning Repository page. A quick slide highlighting the 30 features and the target variable.
**Audio (Student Narrator):**
"For the dataset, I went with the Breast Cancer Wisconsin Diagnostic Data Set from the UCI repository. It's a classic, but it perfectly fits the requirements. 
The dataset gives us 30 real-valued features computed from cell nuclei—things like radius, texture, and smoothness. And of course, the target variable is our diagnosis: 'M' for Malignant and 'B' for Benign.
To make sure the project is completely reproducible for grading, I wrote a quick script that automatically fetches the `wdbc.data` file right from the UCI servers using `curl`, so you don't have to download anything manually to test my code."

## [1:30 - 2:30] My Approach: Ensemble Machine Learning
**Visual:** A simple, student-made flowchart diagram showing the three models (Random Forest, Gradient Boosting, Logistic Regression) feeding into a "Voting Classifier".
**Audio (Student Narrator):**
"When I started building the model, I quickly realized that relying on just one algorithm felt a bit risky, especially for a medical dataset. So, I decided to implement an ensemble strategy to try and maximize my recall.
I combined three different approaches:
First, I used a Random Forest for **bagging**, because it's great at handling variance and feature interactions.
Second, I added Gradient Boosting to iteratively minimize bias.
And finally, I tied it all together using a **Voting Classifier** with soft voting. This takes the predicted probabilities from the Random Forest, the Gradient Boosting model, and a standard Logistic Regression model, and combines them to make the final prediction. It actually generalized way better than any of them did individually."

## [2:30 - 3:30] Implementation & Code Walkthrough
**Visual:** Screen recording of the code editor (VS Code or similar), highlighting the `StandardScaler` and `VotingClassifier` setup.
**Audio (Student Narrator):**
"Let me show you a bit of the code. 
After loading the data, I dropped the ID column since it doesn't have predictive value, and encoded the diagnosis into 1s and 0s. 
One really important step here was using `train_test_split` with `stratify=y`. I wanted to make sure the ratio of benign to malignant cases stayed consistent across my training and testing sets.
Also, because I included a Logistic Regression model in the ensemble, I had to use `StandardScaler` to scale the features. Without scaling, the distance metrics get totally thrown off by features that have larger natural ranges.
After that, setting up the `VotingClassifier` was pretty straightforward in scikit-learn."

## [3:30 - 4:15] Results & Why Recall Matters Here
**Visual:** Terminal output showing the `classification_report` and `confusion_matrix`. The mouse cursor highlights the "Recall" for class 1 (Malignant) and the False Negatives in the matrix.
**Audio (Student Narrator):**
"Moving on to the results—the overall accuracy was good, but like we've talked about in lectures, accuracy isn't everything. 
For this specific medical context, I really focused on **Recall** for the malignant class. A false positive means a patient gets an unnecessary scare, which is bad, but a false negative means a patient is told they're fine when they actually have cancer, which is catastrophic.
So, when you look at my confusion matrix, my main priority was minimizing that false negative number. The ensemble model did a really solid job of keeping that number as low as possible."

## [4:15 - 5:00] Next Steps & Conclusion
**Visual:** A final slide titled "Future Improvements" with bullet points: Cross-Validation, Hyperparameter Tuning, SHAP.
**Audio (Student Narrator):**
"If I had more time to expand on this assignment, there are a few things I'd add to make it more defensible. 
I'd definitely implement k-fold cross-validation to prove the model is stable. Right now, I'm just using base parameters, so I'd love to run a Grid Search to properly tune the ensemble.
Finally, I think adding SHAP (SHapley Additive exPlanations) would be awesome. It would let us explain exactly which cell features drove the model's decision, which is super important for interpretability in healthcare.
Thanks for taking the time to review my project, Professor! Let me know if you have any questions."
