import math
import numpy as np
import scipy.stats as stats

from sklearn.utils import shuffle, resample
from sklearn.ensemble import IsolationForest, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier

from tqdm import tqdm


class HNFCL():
    def __init__(self, cfg_if, cfg_gbdt, cfg_rf, cfg_et, random_state=42, verbose=True):
        
        self.random_state = random_state
        self.verbose = verbose
        
        self.cfg_if = cfg_if
        self.cfg_gbdt = cfg_gbdt
        self.cfg_rf = cfg_rf
        self.cfg_et = cfg_et
        
        self.gbdt = GradientBoostingClassifier(n_estimators=cfg_gbdt['n_estimators'], max_depth=cfg_gbdt['max_depth'], loss=cfg_gbdt['loss'], learning_rate=cfg_gbdt['learning_rate'], random_state=self.random_state)
        self.rf = RandomForestClassifier(n_estimators=cfg_rf['n_estimators'], max_depth=cfg_rf['max_depth'], criterion=cfg_rf['criterion'], random_state=self.random_state)
        self.et = ExtraTreesClassifier(n_estimators=cfg_et['n_estimators'], max_depth=cfg_et['max_depth'], criterion=cfg_et['criterion'], random_state=self.random_state)

        self.classifiers = [self.gbdt, self.rf, self.et]
        
    
    def __detect_outliers(self, X, y, sample_weight=None):
        
        X_unlabel = []
        y_unlabel = []
        X_label = []
        y_label = []
        sample_weight_label = []
        sample_weight_unlabel = []
        
        for class_val in np.unique(y):
            class_idxs = np.where(y == class_val)
            X_current_class = X[class_idxs]
            
            iforest = IsolationForest(n_estimators=self.cfg_if['n_estimators'], max_features=self.cfg_if['max_features'], random_state=self.random_state).fit(X_current_class)
            iforest_pred = iforest.predict(X_current_class)
            
            outliers_idx = np.where(iforest_pred == -1)
            normal_idx = np.where(iforest_pred == 1)
            
            X_label_by_class = X_current_class[normal_idx]
            y_label_by_class = [class_val for i in range(len(X_label_by_class))]
            X_unlabel_by_class = X_current_class[outliers_idx]
            y_unlabel_by_class = [class_val for i in range(len(X_unlabel_by_class))]
            
            X_label.extend(X_label_by_class)
            y_label.extend(y_label_by_class)
            X_unlabel.extend(X_unlabel_by_class)
            y_unlabel.extend(y_unlabel_by_class)

            if sample_weight is not None:
                sample_weight_current_class = sample_weight[class_idxs]
                
                sample_weight_label_by_class = sample_weight_current_class[normal_idx]
                sample_weight_label.extend(sample_weight_label_by_class)

                sample_weight_unlabel_by_class = sample_weight_current_class[outliers_idx]
                sample_weight_unlabel.extend(sample_weight_unlabel_by_class)                
            
        X_label = np.array(X_label)
        y_label = np.array(y_label)
        X_unlabel = np.array(X_unlabel)
        y_unlabel = np.array(y_unlabel)
        
        X_unlabel, y_unlabel = shuffle(X_unlabel, y_unlabel, random_state=self.random_state)
        X_label, y_label = shuffle(X_label, y_label, random_state=self.random_state)

        if sample_weight is not None:
            sample_weight_label = np.array(sample_weight_label)
            sample_weight_unlabel = np.array(sample_weight_unlabel)
            X_label, y_label, sample_weight_label = shuffle(X_label, y_label, sample_weight_label, random_state=self.random_state)
            X_unlabel, y_unlabel, sample_weight_unlabel = shuffle(X_unlabel, y_unlabel, sample_weight_unlabel, random_state=self.random_state)
        else:
            X_label, y_label = shuffle(X_label, y_label, random_state=self.random_state)
            X_unlabel, y_unlabel = shuffle(X_unlabel, y_unlabel, random_state=self.random_state)
            
        return X_label, y_label, X_unlabel, y_unlabel, sample_weight_label, sample_weight_unlabel
    
    
    def __measure_error(self, X, y, j, k):
        
        pred_j = self.classifiers[j].predict(X)
        pred_k = self.classifiers[k].predict(X)
        
        wrong_index = (pred_j != y) & (pred_j == pred_k)
        
        return sum(wrong_index)/sum(pred_j == pred_k)
    
    
    def fit(self, X, y, sample_weight=None):

        X_label, y_label, X_unlabel, y_unlabel, sample_weight_label, sample_weight_unlabel = self.__detect_outliers(X, y, sample_weight)

        if self.verbose:
            print('Step 1 - Fitting classifiers using resampled labeled data')
            
        e_prime = []
        l_prime = []
        for i in range(3):
            X_resampled, y_resampled = resample(X_label, y_label, random_state=self.random_state+i)  # BootstrapSample(L)
            if sample_weight is not None:
                self.classifiers[i].fit(X_resampled, y_resampled, sample_weight=sample_weight_label)  # Learn(Si)
            else:
                self.classifiers[i].fit(X_resampled, y_resampled)
            e_prime.append(0.5)
            l_prime.append(0)

        if self.verbose:
            print('Step 1 done!')

        update = [False, False, False]
        changes = True
        e = [0, 0, 0]

        Li_X = [[], [], []]
        Li_y = [[], [], []]
        Li_w = [[], [], []]

        if self.verbose:
            print('Step 2 - Introducing unlabeled (anomalous) data')
            
        iteration = 0
        while changes:
            iteration += 1
            for i in range(3):
                
                update[i] = False
                
                j, k = [elem for elem in [0,1,2] if elem!=i]
                e[i] = self.__measure_error(X_label, y_label, j, k)

                if e[i] < e_prime[i]:
                    pred_j_unlabeled = self.classifiers[j].predict(X_unlabel)
                    pred_k_unlabeled = self.classifiers[k].predict(X_unlabel)

                    Li_X[i] = X_unlabel[pred_j_unlabeled == pred_k_unlabeled] # when the other two models agree on the label
                    Li_y[i] = pred_j_unlabeled[pred_j_unlabeled == pred_k_unlabeled]
                    if sample_weight is not None:
                        Li_w[i] = sample_weight_unlabel[pred_j_unlabeled == pred_k_unlabeled]

                    if l_prime[i] == 0: # not updated before
                        l_prime[i]  = math.floor((e[i]/(e_prime[i] - e[i])) + 1)

                    if l_prime[i] < len(Li_y[i]):
                        
                        if (e[i] * len(Li_y[i])) < (e_prime[i] * l_prime[i]):
                            update[i] = True
                            
                        elif l_prime[i] > (e[i]/(e_prime[i] - e[i])):
                            #L_index = np.random.choice(len(Li_y[i]), math.ceil(((e_prime[i] * l_prime[i]) / e[i]) - 1)) # subsample from proxy labeled data
                            
                            # Calculate the number of indices to select
                            num_indices = math.ceil(((e_prime[i] * l_prime[i]) / e[i]) - 1)

                            # Generate random indices using sklearn's resample
                            L_index = resample(
                                range(len(Li_y[i])),
                                n_samples=num_indices,
                                replace=False,
                                random_state=self.random_state
                            )

                            
                            Li_X[i] = Li_X[i][L_index]
                            Li_y[i] = Li_y[i][L_index]
                            if sample_weight is not None:
                                Li_w[i] = Li_w[i][L_index]
                            update[i] = True

            for i in range(3):
                
                if update[i]:
                    if sample_weight is not None:
                        self.classifiers[i].fit(np.append(X_label, Li_X[i], axis=0), np.append(y_label, Li_y[i], axis=0), sample_weight=np.append(sample_weight_label, Li_w[i], axis=0)) # train the classifier on integrated dataset
                    else:
                        self.classifiers[i].fit(np.append(X_label, Li_X[i], axis=0), np.append(y_label, Li_y[i], axis=0)) # train the classifier on integrated dataset
                    e_prime[i] = e[i]
                    l_prime[i] = len(Li_y[i])

            if not any(update):
                changes = False # if no classifier was updated, no improvement

            if self.verbose:
                print(f'Iter {iteration} - {sum(update)} updates')

        if self.verbose:
            print('Step 2 done!')
    
    def predict(self, X_test):
        # Collect predictions from all three classifiers
        predictions = np.array([clf.predict(X_test) for clf in self.classifiers])

        # Perform majority voting along axis 0 (classifiers) for each test sample
        majority_vote = stats.mode(predictions).mode

        return majority_vote

    def save(self, directory, filename):

        with open(os.path.join(directory, f'{filename}_gbdt.pkl'), 'wb') as f:
            pickle.dump(self.gbdt, f)
        with open(os.path.join(directory, f'{filename}_rf.pkl'), 'wb') as f:
            pickle.dump(self.rf, f)
        with open(os.path.join(directory, f'{filename}_et.pkl'), 'wb') as f:
            pickle.dump(self.et, f)

    def load(self, directory, filename):
        
        with open(os.path.join(directory, f'{filename}_gbdt.pkl'), "rb") as f:
            self.gbdt = pickle.load(f)
        with open(os.path.join(directory, f'{filename}_rf.pkl'), "rb") as f:
            self.rf = pickle.load(f)
        with open(os.path.join(directory, f'{filename}_et.pkl'), "rb") as f:
            self.et = pickle.load(f)
