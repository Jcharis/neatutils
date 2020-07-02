#!/usr/bin/python
# Get the Commonly Used Abbreviation for ML Estimators/Algorithms

classification_estimators = {
    "Logistic Regression": "lr",
    "K Nearest Neighbour": "knn",
    "Naives Bayes": "nb",
    "Decision Tree": "dt",
    "SVM (Linear)": "svm",
    "SVM (RBF)": "rbfsvm",
    "Gaussian Process": "gpc",
    "Multi Level Perceptron": "mlp",
    "Ridge Classifier": "ridge",
    "Random Forest": "rf",
    "Quadratic Discriminant Analysis": "qda",
    "AdaBoost": "ada",
    "Gradient Boosting Classifier": "gbc",
    "Linear Discriminant Analysis": "lda",
    "Extra Trees Classifier": "et",
    "Extreme Gradient Boosting": "xgboost",
    "Light Gradient Boosting": "lightgbm",
    "Cat Boost Classifier": "catboost",
}

regression_estimators = {
    "Linear Regression": "lr",
    "Lasso Regression": "lasso",
    "Ridge Regression": "ridge",
    "Elastic Net": "en",
    "Least Angle Regression": "lar",
    "Lasso Least Angle Regression": "llar",
    "Orthogonal Matching Pursuit": "omp",
    "Bayesian Ridge": "br",
    "Automatic Relevance Determination": "ard",
    "Passive Aggressive Regressor": "par",
    "Random Sample Consensus": "ransac",
    "TheilSen Regressor": "tr",
    "Huber Regressor": "huber",
    "Kernel Ridge": "kr",
    "Support Vector Machine": "svm",
    "K Neighbors Regressor": "knn",
    "Decision Tree": "dt",
    "Random Forest": "rf",
    "Extra Trees Regressor": "et",
    "AdaBoost Regressor": "ada",
    "Gradient Boosting Regressor": "gbr",
    "Multi Level Perceptron": "mlp",
    "Extreme Gradient Boosting": "xgboost",
    "Light Gradient Boosting": "lightgbm",
    "CatBoost Regressor": "catboost",
}

anomaly_detection_estimators = {
    "Angle-base Outlier Detection": "abod",
    "Isolation Forest": "iforest",
    "Clustering-Based Local Outlier": "cluster",
    "Connectivity-Based Outlier Factor": "cof",
    "Histogram-based Outlier Detection": "histogram",
    "k-Nearest Neighbors Detector": "knn",
    "Local Outlier Factor": "lof",
    "One-class SVM detector": "svm",
    "Principal Component Analysis": "pca",
    "Minimum Covariance Determinant": "mcd",
    "Subspace Outlier Detection": "sod",
    "Stochastic Outlier Selection": "sos",
}

nlp_estimators = {
    "Latent Dirichlet Allocation": "lda",
    "Latent Semantic Indexing": "lsi",
    "Hierarchical Dirichlet Process": "hdp",
    "Random Projections": "rp",
    "Non-Negative Matrix Factorization": "nmf",
}

clustering_estimators = {
    "K-Means Clustering": "kmeans",
    "Affinity Propagation": "ap",
    "Mean shift Clustering": "meanshift",
    "Spectral Clustering": "sc",
    "Agglomerative Clustering": "hclust",
    "Density-Based Spatial Clustering": "dbscan",
    "OPTICS Clustering": "optics",
    "Birch Clustering": "birch",
    "K-Modes Clustering": "kmodes",
}


all_estimators = {
    "Logistic Regression": "lr",
    "K Nearest Neighbour": "knn",
    "Naives Bayes": "nb",
    "Decision Tree": "dt",
    "SVM (Linear)": "svm",
    "SVM (RBF)": "rbfsvm",
    "Gaussian Process": "gpc",
    "Multi Level Perceptron": "mlp",
    "Ridge Classifier": "ridge",
    "Random Forest": "rf",
    "Quadratic Discriminant Analysis": "qda",
    "AdaBoost": "ada",
    "Gradient Boosting Classifier": "gbc",
    "Linear Discriminant Analysis": "lda",
    "Extra Trees Classifier": "et",
    "Extreme Gradient Boosting": "xgboost",
    "Light Gradient Boosting": "lightgbm",
    "Cat Boost Classifier": "catboost",
    "Linear Regression": "lr",
    "Lasso Regression": "lasso",
    "Ridge Regression": "ridge",
    "Elastic Net": "en",
    "Least Angle Regression": "lar",
    "Lasso Least Angle Regression": "llar",
    "Orthogonal Matching Pursuit": "omp",
    "Bayesian Ridge": "br",
    "Automatic Relevance Determination": "ard",
    "Passive Aggressive Regressor": "par",
    "Random Sample Consensus": "ransac",
    "TheilSen Regressor": "tr",
    "Huber Regressor": "huber",
    "Kernel Ridge": "kr",
    "Support Vector Machine": "svm",
    "K Neighbors Regressor": "knn",
    "Decision Tree": "dt",
    "Random Forest": "rf",
    "Extra Trees Regressor": "et",
    "AdaBoost Regressor": "ada",
    "Gradient Boosting Regressor": "gbr",
    "Multi Level Perceptron": "mlp",
    "Extreme Gradient Boosting": "xgboost",
    "Light Gradient Boosting": "lightgbm",
    "CatBoost Regressor": "catboost",
    "Angle-base Outlier Detection": "abod",
    "Isolation Forest": "iforest",
    "Clustering-Based Local Outlier": "cluster",
    "Connectivity-Based Outlier Factor": "cof",
    "Histogram-based Outlier Detection": "histogram",
    "k-Nearest Neighbors Detector": "knn",
    "Local Outlier Factor": "lof",
    "One-class SVM detector": "svm",
    "Principal Component Analysis": "pca",
    "Minimum Covariance Determinant": "mcd",
    "Subspace Outlier Detection": "sod",
    "Stochastic Outlier Selection": "sos",
    "Latent Dirichlet Allocation": "lda",
    "Latent Semantic Indexing": "lsi",
    "Hierarchical Dirichlet Process": "hdp",
    "Random Projections": "rp",
    "Non-Negative Matrix Factorization": "nmf",
    "K-Means Clustering": "kmeans",
    "Affinity Propagation": "ap",
    "Mean shift Clustering": "meanshift",
    "Spectral Clustering": "sc",
    "Agglomerative Clustering": "hclust",
    "Density-Based Spatial Clustering": "dbscan",
    "OPTICS Clustering": "optics",
    "Birch Clustering": "birch",
    "K-Modes Clustering": "kmodes",
    "Validation Curve": "vc",
    "Confusion Matrix": "cm/confusion_matrix",
}


def get_abbrev(estimator_name, estimator_type="all"):
    """Return the Abbreviation/Short Form for an ML Estimator/Algorithm

	# Example
	>>> get_abbrev('Logistic Regression')
	'lr'

	>>> get_abbrev('Logistic Regression','classification')
	'lr'

	"""

    if estimator_type == "classification":
        for key, value in classification_estimators.items():
            if estimator_name.title() == key:
                return value
        return False
    elif estimator_type == "regression":
        for key, value in regression_estimators.items():
            if estimator_name.title() == key:
                return value
        return False

    elif estimator_type == "clustering":
        for key, value in clustering_estimators.items():
            if estimator_name.title() == key:
                return value
        return False

    elif estimator_type == "nlp":
        for key, value in nlp_estimators.items():
            if estimator_name.title() == key:
                return value
        return False

    elif estimator_type == "anomaly":
        for key, value in anomaly_detection_estimators.items():
            if estimator_name.title() == key:
                return value
        return False
    else:
        for key, value in all_estimators.items():
            if estimator_name.title() == key:
                return value
        return False


def get_fullname(estimator_abbrev, estimator_type="all"):
    """Return the Full Name for an Abbreviated Estimator/Algorithm

	# Example
	>>> get_fullname('lr')
	'Logistic Regression'
	'Linear Regression'

	>>> get_fullname('dt')
	'Decision Tree'
	"""

    for key, value in all_estimators.items():
        if estimator_abbrev.lower() == value:
            return key
    return False
