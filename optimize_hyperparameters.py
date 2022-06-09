import sys
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, HalvingGridSearchCV
import pandas as pd
from joblib import dump, load
from scipy.stats import loguniform

UKBB_PATH = '../glaucoma_project/UKBB_Data/'


def main():
    t0 = time.time()
    model_id = sys.argv[1]
    grid = eval(sys.argv[2])
    
    if model_id in ['1a','1b','2','3']:
        all_clinical_data = pd.read_pickle(UKBB_PATH+'processed_data/imputed_matrix_selected_clinical_features.pkl')
        y_train = pd.read_pickle(UKBB_PATH+'processed_data/glaucoma_diagnosis.pkl')
        print('Hyperparameter optimization time...')
        
        classification_dict = load(UKBB_PATH+'processed_data/classification_dict.joblib')
        
        if model_id == '1a':
            columns_to_select = [col for col in classification_dict.keys() if classification_dict[col][1] == '1a']
            model = LogisticRegression(class_weight='balanced',max_iter=300)
            # https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/
            # Initial grid: '{"C":[1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100], "penalty":["none", "l1", "l2", "elasticnet"], "solver":["newton-cg", "lbfgs", "liblinear"]}'
            # Best params {'C': 0.001, 'penalty': 'l2', 'solver': 'liblinear'}
            
        if model_id == '1b':
            columns_to_select = [col for col in classification_dict.keys() if classification_dict[col][1] in ['1a','1b']]
            # https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
            model = GradientBoostingClassifier(random_state=42)
            # Grid: {"min_samples_split":[2,20,200,2000,4000], "max_depth":[1,3,5], "n_estimators":[10,50,100,500,1000]}     
            # Best params {'max_depth': 1, 'min_samples_split': 2, 'n_estimators': 1000}
            
        if model_id == '2':
            columns_to_select = [col for col in classification_dict.keys() if classification_dict[col][1] in ['1a','1b', '2']]
            model = GradientBoostingClassifier(random_state=42)
            # Grid: {"min_samples_split":[2,20,200,2000,4000], "max_depth":[1,3,5], "n_estimators":[10,50,100,500,1000]}     
            # Best params {'max_depth': 1, 'min_samples_split': 2, 'n_estimators': 1000}

            
        if model_id == '3':
            columns_to_select = all_clinical_data.columns
            model = GradientBoostingClassifier(random_state=42)
            # Grid: {"min_samples_split":[2,20,200,2000,4000], "max_depth":[1,3,5], "n_estimators":[10,50,100,500,1000]}     
            # Best params {'max_depth': 1, 'min_samples_split': 2, 'n_estimators': 1000}
             
        X_train = all_clinical_data[columns_to_select]
        
        

    
    if model_id == 'G':
        print('Hyperparameter optimization time... for genetic data.')
        X_train = pd.read_pickle(UKBB_PATH+'processed_data/imputed_genes.pkl')
        X_train = X_train[X_train.columns[2:]]
        y_train = pd.read_pickle(UKBB_PATH+'processed_data/glaucoma_diagnosis_for_genetic_ml.pkl')
        # Grid: {"hidden_layer_sizes":[(50,),(100,),(200,),(50,50),(100,100),(200,200)],"alpha":[0.01,0.1,1]}
        model = MLPClassifier(random_state=42)
    
    print('Defining grid search...')
    # define search
    search = GridSearchCV(model, grid, scoring='roc_auc', n_jobs=1, cv=5, verbose=2, error_score=0.0)
    print('Executing grid search...')
    # execute search
    result = search.fit(X_train, y_train)
    print('Best params',search.best_params_)
    print('Best score',search.best_score_)
    params_as_string = '_'.join([param+'_'+str(search.best_params_[param]) for param in search.best_params_.keys()])
    dump(search,'../glaucoma_project/Models/'+model_id+'_'+params_as_string+'_AUC_'+str(search.best_score_)+'.joblib')
    
    print('Time',time.time() - t0)
    


if __name__ == '__main__':
    main()