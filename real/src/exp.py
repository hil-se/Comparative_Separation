from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from load_data import load
import numpy as np
from preprocessor import *
from metrics import Metrics
from pdb import set_trace

class Exp:
    def __init__(self, data, treatment="None"):
        #  Load data
        self.data, self.A = load(data)
        # Separate independent variables and dependent variables
        independent = self.data.keys().tolist()
        dependent = independent.pop(-1)
        self.X = self.data[independent]
        self.y = np.array(self.data[dependent])
        self.treatment = treatment
        self.clf = LogisticRegression(max_iter=100000)
        self.alpha = 0.05

    def exp_separation(self, N=100, r=10000):
        violate = {protected: 0 for protected in self.A }
        for i in range(r):
            selectedr = np.random.choice(len(self.y_test), size=N, replace=False)
            m = Metrics(self.y_test[selectedr], self.preds[selectedr])
            for protected in self.A:
                ps = m.separation(np.array(self.X_test[protected][selectedr]), stats=False)
                if min((ps)) < self.alpha:
                    violate[protected] += 1
        for protected in self.A:
            violate[protected] = violate[protected] / r
        return violate

    def exp_comp_separation(self, N=100, r=10000):
        violate = {protected: 0 for protected in self.A }
        for i in range(r):
            selected1 = np.random.choice(len(self.y_test), size=N, replace=False)
            selected2 = np.random.choice(len(self.y_test), size=N, replace=False)
            y = self.y_test[selected1] - self.y_test[selected2]
            pred = self.preds[selected1] - self.preds[selected2]
            m = Metrics(y, pred)
            for protected in self.A:
                ps = m.comparative_separation(np.array(self.X_test[protected][selected1]), np.array(self.X_test[protected][selected2]))
                if min((ps)) < self.alpha:
                    violate[protected] += 1
        for protected in self.A:
            violate[protected] = violate[protected] / r
        return violate

    # def one_exp(self):
    #     self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split(test_size=0.5)
    #     #########################################
    #     self.data_preprocess(self.X_train)
    #     #########################################
    #
    #     sample_weight = self.treat(self.X_train, self.y_train)
    #     self.fit(self.X_train, self.y_train, sample_weight)
    #     self.preds = self.predict(self.X_test)
    #     m = Metrics(self.y_test, self.preds)
    #     for protected in self.A:
    #         print("Protected Attribute: %s" %protected)
    #         print("Separation")
    #         ps = m.separation(self.X_test[protected], stats=True)
    #         print(ps)
    #
    #     violate = self.exp_separation(N=200, r=10000)
    #     print("Separation violation rate (N=200): " )
    #     print(violate)
    #     violate = self.exp_separation(N=400, r=10000)
    #     print("Separation violation rate (N=400): " )
    #     print(violate)
    #     violate = self.exp_comp_separation(N=200, r=10000)
    #     print("Comparative separation violation rate (N=200): ")
    #     print(violate)
    #     violate = self.exp_comp_separation(N=400, r=10000)
    #     print("Comparative separation violation rate (N=400): ")
    #     print(violate)

    def random_exp(self, violate_sep, violate_comp):
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split(test_size=0.5)
        #########################################
        self.data_preprocess(self.X_train)
        #########################################
        N = len(self.y_test)
        sample_weight = self.treat(self.X_train, self.y_train)
        self.fit(self.X_train, self.y_train, sample_weight)
        self.preds = self.predict(self.X_test)
        m_sep = Metrics(self.y_test, self.preds)
        selected1 = np.random.choice(N, size=N*2, replace=True)
        selected2 = np.random.choice(N, size=N*2, replace=True)
        comp_y = self.y_test[selected1] - self.y_test[selected2]
        comp_pred = self.preds[selected1] - self.preds[selected2]
        m_comp = Metrics(comp_y, comp_pred)
        for protected in self.A:
            ps = m_sep.separation(self.X_test[protected], stats=False)
            if min((ps)) < self.alpha:
                violate_sep[protected]+=1
            ps = m_comp.comparative_separation(np.array(self.X_test[protected][selected1]), np.array(self.X_test[protected][selected2]), stats=False)
            if min((ps)) < self.alpha:
                violate_comp[protected]+=1


    def exp_train(self):
        #########################################
        self.data_preprocess(self.X)
        #########################################

        sample_weight = self.treat(self.X, self.y)
        self.fit(self.X, self.y, sample_weight)
        preds = self.predict(self.X)
        m = Metrics(self.y, preds)
        return m


    def fit(self, X, y, sample_weight=None):
        X_train_processed = self.preprocessor.fit_transform(X)
        self.clf.fit(X_train_processed, y, sample_weight=sample_weight)

    def predict(self, X):
        X_processed = self.preprocessor.transform(X)
        preds = self.clf.predict(X_processed)
        return preds

    def data_preprocess(self, X):
        numerical_columns_selector = selector(dtype_exclude=object)
        categorical_columns_selector = selector(dtype_include=object)

        numerical_columns = numerical_columns_selector(X)
        categorical_columns = categorical_columns_selector(X)

        categorical_preprocessor = OneHotEncoder(handle_unknown = 'ignore')
        numerical_preprocessor = StandardScaler()
        self.preprocessor = ColumnTransformer([
            ('OneHotEncoder', categorical_preprocessor, categorical_columns),
            ('StandardScaler', numerical_preprocessor, numerical_columns)])

    def treat(self, X_train, y_train):
        if self.treatment == "Reweighing":
            sample_weight = Reweighing(X_train, y_train, self.A)
        elif self.treatment == "FairBalance":
            sample_weight = FairBalance(X_train, y_train, self.A)
        else:
            sample_weight = None
        return sample_weight

    def train_test_split(self, test_size=0.3):
        # Split training and testing data proportionally across each group
        groups = {}
        for i in range(len(self.y)):
            key = tuple([self.X[a][i] for a in self.A] + [self.y[i]])
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        train = []
        test = []
        for key in groups:
            testing = list(np.random.choice(groups[key], int(len(groups[key])*test_size), replace=False))
            training = list(set(groups[key]) - set(testing))
            test.extend(testing)
            train.extend(training)
        X_train = self.X.iloc[train]
        X_test = self.X.iloc[test]
        y_train = self.y[train]
        y_test = self.y[test]
        X_train.index = range(len(X_train))
        X_test.index = range(len(X_test))
        return X_train, X_test, y_train, y_test

