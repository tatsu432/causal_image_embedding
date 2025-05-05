import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import torch

class EstimatorATE:
    def __init__(self, ATE_regression, ATE_ipw, ATE_dr):
        self._regression = ATE_regression
        self._ipw = ATE_ipw
        self._dr = ATE_dr

    def error_reg(self, true_ATE):
        return (self._regression - true_ATE)**2
    
    def error_ipw(self, true_ATE):
        return (self._ipw - true_ATE)**2
    
    def error_dr(self, true_ATE):
        return (self._dr - true_ATE)**2

    @property
    def regression(self):
        return self._regression
    
    @property
    def ipw(self):
        return self._ipw
    
    @property
    def dr(self):
        return self._dr
    
    def __str__(self):
        return f"Regression estimate: {self._regression}, IPW estimate: {self._ipw}, DR estimate: {self._dr}"
    

class ATE:
    def __init__(self, true_ATE: float, biased_ATE: EstimatorATE, naive_ATE: EstimatorATE, debiased_ATE: EstimatorATE):
        self._true_ATE = true_ATE
        self._biased_ATE = biased_ATE
        self._naive_ATE = naive_ATE
        self._debiased_ATE = debiased_ATE

    @property
    def true_ATE(self):
        return self._true_ATE
    
    @property
    def biased_ATE(self):
        return self._biased_ATE
    
    @property
    def naive_ATE(self):
        return self._naive_ATE
    
    @property
    def debiased_ATE(self):
        return self._debiased_ATE
    
    def __str__(self):
        return f"True ATE: {self._true_ATE}\n Biased ATE: {self._biased_ATE}\n Naive ATE: {self._naive_ATE}\n Debiased ATE: {self._debiased_ATE}"



def compute_ATE(dataset, type = "true", covariate_image = None):

    ALLOWED_TYPES = ["true", "biased", "learned_covariate_image"]
    if type not in ALLOWED_TYPES:
        raise ValueError(f"Invalid type: {type}. Must be one of: {ALLOWED_TYPES}")

    # Define the predictors, intervened predictors, control predictors, and outcome for each type
    if type == "true":
        predictors_outcome_model = torch.cat((dataset["treatment"].view(-1, 1), dataset["covariate"], dataset["covariate_image"]), dim=1)
        intervened_predictors = torch.cat((torch.ones_like(dataset["treatment"].view(-1, 1)), dataset["covariate"], dataset["covariate_image"]), dim=1)
        control_predictors = torch.cat((torch.zeros_like(dataset["treatment"].view(-1, 1)), dataset["covariate"], dataset["covariate_image"]), dim=1)
    elif type == "biased":
        predictors_outcome_model = torch.cat((dataset["treatment"].view(-1, 1), dataset["covariate"]), dim=1)
        intervened_predictors = torch.cat((torch.ones_like(dataset["treatment"].view(-1, 1)), dataset["covariate"]), dim=1)
        control_predictors = torch.cat((torch.zeros_like(dataset["treatment"].view(-1, 1)), dataset["covariate"]), dim=1)
    elif type == "learned_covariate_image":
        predictors_outcome_model = torch.cat((dataset["treatment"].view(-1, 1), dataset["covariate"], covariate_image), dim=1)
        intervened_predictors = torch.cat((torch.ones_like(dataset["treatment"].view(-1, 1)), dataset["covariate"], covariate_image), dim=1)
        control_predictors = torch.cat((torch.zeros_like(dataset["treatment"].view(-1, 1)), dataset["covariate"], covariate_image), dim=1)
    outcome = dataset["outcome"]

    # Convert tensors to NumPy arrays
    predictors_outcome_model = predictors_outcome_model.numpy()
    intervened_predictors = intervened_predictors.numpy()
    control_predictors = control_predictors.numpy()
    outcome = outcome.numpy()

    # Fit the linear regression model
    outcome_model = LinearRegression()
    outcome_model.fit(predictors_outcome_model, outcome)

    # Predict the outcome for the intervened and control groups
    hat_E_Y_1 = outcome_model.predict(intervened_predictors)
    hat_E_Y_0 = outcome_model.predict(control_predictors)

    # Define the covariates and treatment for the propensity score model for each type
    if type == "true":
        covariates_propensity_model = torch.cat((dataset["covariate"], dataset["covariate_image"]), dim=1)
    elif type == "biased":
        covariates_propensity_model = dataset["covariate"]
    elif type == "learned_covariate_image":
        covariates_propensity_model = torch.cat((dataset["covariate"], covariate_image), dim=1)
    treatment = dataset["treatment"]

    # Convert tensors to NumPy arrays
    covariates_propensity_model = covariates_propensity_model.numpy()
    treatment = treatment.numpy()

    # Fit the logistic regression model
    true_propensity_score_model = LogisticRegression()
    true_propensity_score_model.fit(covariates_propensity_model, treatment)

    propensity_scores = true_propensity_score_model.predict_proba(covariates_propensity_model)[:, 1]

    propensity_scores = np.clip(propensity_scores, 1e-2, 1 - 1e-2)

    # Regression estimator
    ATE_regression = np.mean(hat_E_Y_1 - hat_E_Y_0)

    # Stabilized IPW estimator
    ATE_ipw = np.mean(outcome * treatment / propensity_scores) / np.mean(treatment / propensity_scores) - np.mean(outcome * (1 - treatment) / (1 - propensity_scores)) / np.mean((1 - treatment) / (1 - propensity_scores))

    # DR estimator
    ATE_dr = np.mean(hat_E_Y_1 + treatment * (outcome - hat_E_Y_1) / propensity_scores - hat_E_Y_0 - (1 - treatment) * (outcome - hat_E_Y_0) / (1 - propensity_scores))

    return EstimatorATE(ATE_regression, ATE_ipw, ATE_dr)





