from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd
from scipy.stats import chi2
from scipy.linalg import sqrtm
from collections import Counter

class MobTree():
    def __init__(self, node_features, leaf_features, max_depth=20, min_size=30, trim = 0.1,alpha = 0.01):
        self.max_depth = max_depth
        self.min_size = min_size
        self.trim = trim
        self.depth = 1
        self.node_feature_num = len(node_features)
        self.leaf_feature_num = len(leaf_features)
        self.node_features = node_features
        self.leaf_features = leaf_features
        self.alpha = alpha
        beta = pd.read_csv('./beta.csv')
        self.beta = np.array(beta)

    def estfun(self,obj):
        return (obj['residuals'].reshape(-1,1)*np.hstack((np.ones(len(obj['x'])).reshape(-1,1),obj['x']))).astype(float)

    def supLM(self, x, k, tlambda,beta):
        ## use Hansen (1997) approximation
        m = beta.shape[1]-1
        if tlambda<1:
            tau = tlambda
        else:
            tau = 1/(1+np.sqrt(tlambda))
        beta = beta[(k-1)*25:(k*25),:]
        dummy = beta[:,0:m].dot(np.power(x,np.array([t for t in range(m)])))
        dummy = dummy*(dummy>0)
        pp = np.log(chi2.pdf(dummy, beta[:,m]))
        if tau==0.5:
            p = np.log(chi2.pdf(x, k))
        elif tau <= 0.01:
            p = pp[25]
        elif tau >= 0.49:
            p = np.log((np.exp(np.log(0.5-tau) + pp[1]) + np.exp(np.log(tau-0.49) + np.log(chi2.pdf(x, k))))*100)
        else:
            taua = (0.51-tau)*50
            tau1 = int(np.floor(taua))
            p = np.log(np.exp(np.log(tau1 + 1 - taua) + pp[tau1-1]) + np.exp(np.log(taua-tau1) + pp[tau1]))

        return p

    def mob_fit_fluctests(self, x, y, minsplit, trim, partvar ):
        lr = LinearRegression()
        lr.fit(x,y)
        obj = {}
        obj['model'] = lr
        obj['residuals'] = np.array(y - lr.predict(x))
        obj['x'] = x
        
        ## set up return values
        m = len(partvar[0])
        n = len(partvar)
        pval = np.zeros(m)
        stat = np.zeros(m)
        ifac = [False for _ in range(m)]
        

        # ## extract estimating functions  
        process = self.estfun(obj)
        k = len(process[0])

        # ## scale process
        process = process/np.sqrt(n)
        J12 = sqrtm(process.T.dot(process))
        process = (np.linalg.inv(J12).dot(process.T)).T

        # ## select parameters to test
        tfrom = int(trim) if trim > 1 else int(np.ceil(n * trim))
        tfrom = max(tfrom, minsplit)
        to = n - tfrom
        tlambda = ((n-tfrom)*to)/(tfrom*(n-to))
        
        beta = self.beta
        

        ## compute statistic and p-value for each ordering
        for i in range(m):
            pvi = partvar[:,i]
            if type(pvi[0])==str:
                proci = process[np.argsort(pvi),:]
                ifac[i] = True

                # # re-apply factor() added to drop unused levels
                pvi = pvi[np.argsort(pvi)]
                # # compute segment weights
                count_info = Counter(pvi)
                segweights = {} ## tapply(ww, pvi, sum)/n      
                for key in count_info.keys():
                    segweights[key] = count_info[key]/n
                # compute statistic only if at least two levels are left
                if len(segweights) < 2:
                    stat[i] = 0
                    pval[i] = None
                else:
                    tsum = 0 
                    for j in range(k):
                        df = pd.DataFrame({'proci':proci[:,j],'pvi':pvi}).groupby('pvi').sum()
                        for key in segweights.keys():
                            tsum += np.power(float(df.loc[key]),2)/segweights[key]
                    stat[i] = tsum
                    pval[i] = np.log(chi2.pdf(stat[i], k*(len(segweights)-1)))
            else:
                oi =  np.argsort(pvi)
                proci = process[oi,:]
                proci = np.cumsum(proci,axis=0)
            
                if tfrom < to:
                    xx = np.sum(np.power(proci,2),axis = 1)
                    xx = xx[tfrom:to]
                    tt = np.array([t for t in range(tfrom,to)])/n
                    stat[i] = max(xx/(tt * (1-tt)))	  
                else:
                    stat[i] = 0
                
                if tfrom < to:
                    pval[i] = self.supLM(stat[i], k, tlambda, beta) 
                else:
                    pval[i] = None
        ## select variable with minimal p-value
        try:
            best = np.argmin(pval)
        except:
            print("catchproblem")
            best = -1
        rval = {}
        rval['pval'] = np.exp(pval)
        rval['stat'] = stat
        rval['best'] = best
        
        return rval

    def test_split_numeric(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return np.array(left), np.array(right)

    def test_split_string(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] == value:
                left.append(row)
            else:
                right.append(row)
        return np.array(left), np.array(right)

    def mape(self, Y_actual, Y_Predicted):
        mape = np.median(np.abs((Y_actual - Y_Predicted) / Y_actual)) * 100
        return mape

    def rms(self, Y_actual, Y_Predicted):
        return mean_squared_error(Y_Predicted,Y_actual, squared=False)

    def get_group_mse(self, group):
        group_array = np.array(group)
        size = float(len(group))
        if size <= 2:
            return 0
        score = self.metric(group_array[:, self.node_feature_num:-1], group_array[:, -1])
        return score

    def groupscore_metric(self, groups):
        n_instances = float(sum([len(group) for group in groups]))
        mse = 0.0
        for group in groups:
            size = float(len(group))
            score = self.get_group_mse(group)
            mse += score * (size / n_instances)
        return mse

    # Select the best split point for a dataset
    def get_split(self, dataset):
        rval = self.mob_fit_fluctests(dataset[:,self.node_feature_num:-1],dataset[:,-1],minsplit=self.min_size,trim = self.trim,partvar = dataset[:,:self.node_feature_num])
        if rval['pval'][rval['best']]>self.alpha:
            return None
        b_index, b_value, b_score, b_groups = 999, 999, 999999999, None
        if self.types[rval['best']]['type'] == 'numeric':
            unique_val = set(dataset[:,rval['best']])
            for val in unique_val:
                groups = self.test_split_numeric(rval['best'], val, dataset)

                if len(groups[0]) < self.min_size or len(groups[1]) < self.min_size:
                    continue
                mse = self.groupscore_metric(groups) 
                if mse < b_score:
                    b_index, b_value, b_score, b_groups = rval['best'], val, mse, groups
        else:
            unique_val = set(dataset[:,rval['best']])
            for val in unique_val:
                groups = self.test_split_string(rval['best'], val, dataset)

                if len(groups[0]) < self.min_size or len(groups[1]) < self.min_size:
                    continue
                mse = self.groupscore_metric(groups) 
                if mse < b_score:
                    b_index, b_value, b_score, b_groups = rval['best'], val, mse, groups
        if b_groups == None:
            return None
        else:
            # print(b_index)
            return {'index': b_index, 'value': b_value, 'groups': b_groups}

    # Create a terminal node value
    def to_terminal(self, group):
        group_array = np.array(group)
        lr = LinearRegression()
        lr.fit(group_array[:, self.node_feature_num:-1], group_array[:, -1])
        coef = lr.coef_
        intercept = lr.intercept_
        return [coef, intercept]

    def plot_terminal(self, y_pred, y_true):
        plt.scatter(y_pred, y_true, s=1)
        plt.xlabel('predict_time')
        plt.ylabel('actual_time')
        x0 = [t for t in range(int(np.max(y_pred)) + 2)]
        y0 = [t for t in x0]
        plt.plot(x0, y0, 'r')
        plt.show()

    def get_terminal_value(self, terminal_params, variable):
        coef = terminal_params[0]
        intercept = terminal_params[1]
        res = 0
        for i in range(len(coef)):
            res += variable[i] * coef[i]
        res += intercept
        return res

    # Create child splits for a node or make terminal
    def split(self, node, max_depth, min_size, depth):
        if depth > self.depth:
            self.depth = depth
        if node:
            left, right = node['groups']
            del (node['groups'])
        else:
            return
        # check for a no split

        if not len(left) or not len(right):
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # process left child
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            split_res = self.get_split(left)
            if split_res:
                node['left'] = split_res
                self.split(node['left'], max_depth, min_size, depth + 1)
            else:
                node['left'] = self.to_terminal(left)
        # process right child
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            split_res = self.get_split(right)
            if split_res:
                node['right'] = split_res
                self.split(node['right'], max_depth, min_size, depth + 1)
            else:
                node['right'] = self.to_terminal(right)

    # Build a tree
    def build_tree(self, train):
        if self.node_feature_num == 0:
            root = {}
            root['left'] = self.to_terminal(train)
        else:
            root = self.get_split(train)
            if root:
                self.split(root, self.max_depth, self.min_size, 1)
            else:
                root = {}
                root['left'] = self.to_terminal(train)
        return root

    def get_column_type(self, dataset, weights=None):
        types = {}
        for t in range(len(dataset[0])):
            if isinstance(dataset[0][t],str):
                value = set(dataset[:,t])
                types[t]={"type":"category","value":value}
            else:
                types[t]={"type":"numeric"}
        return types

    def fit(self, train, weight=None):
        self.types = self.get_column_type(train, weight)
        self.root = self.build_tree(train)

    def predict(self, X_test):
        result = []
        for i in range(len(X_test)):
            result.append(self.__predict(self.root, X_test[i]))
        return result

    def __predict(self, node, row):
        if 'index' not in node.keys():
            return self.get_terminal_value(node['left'], row[self.node_feature_num:])
        if self.types[node['index']]['type'] == 'numeric' or self.types[node['index']]['type'] == 'bucket':
            if row[node['index']] < node['value']:
                if isinstance(node['left'], dict):
                    return self.__predict(node['left'], row)
                else:
                    return self.get_terminal_value(node['left'], row[self.node_feature_num:])
            else:
                if isinstance(node['right'], dict):
                    return self.__predict(node['right'], row)
                else:
                    return self.get_terminal_value(node['right'], row[self.node_feature_num:])
        else:
            if row[node['index']] == node['value']:
                if isinstance(node['left'], dict):
                    return self.__predict(node['left'], row)
                else:
                    return self.get_terminal_value(node['left'], row[self.node_feature_num:])
            else:
                if isinstance(node['right'], dict):
                    return self.__predict(node['right'], row)
                else:
                    return self.get_terminal_value(node['right'], row[self.node_feature_num:])

    def __rsquared_compute(self, y_pred, y_true):
        mean = np.mean(y_true)
        SST = np.sum(np.square(y_true - mean)) + np.exp(-8)
        SSReg = np.sum(np.square(y_true - y_pred))
        score = 1 - SSReg / SST
        return score

    def metric(self, x, y_true):
        lr = LinearRegression()
        lr.fit(x,y_true)
        result = lr.predict(x)
        return self.rms(y_true,result)

    def score(self, X_test):
        pred = self.predict(X_test[:, :-1])
        return self.__rsquared_compute(pred, X_test[:, -1])

    # Print a tree
    def print_tree(self, node, depth=0):
        if isinstance(node, dict) and 'index' not in node:
            print('%s[%s]' % ((depth * ' ', node)))
        elif isinstance(node, dict):
            if self.types[node['index']]['type'] == 'numeric' or self.types[node['index']]['type'] == 'bucket':
                print('%s[%s < %.3f]' % ((depth * ' ', (self.node_features[node['index']]), node['value'])))
            else:
                print('%s[%s == %s]' % ((depth * ' ', (self.node_features[node['index']]), node['value'])))
            self.print_tree(node['left'], depth + 1)
            self.print_tree(node['right'], depth + 1)
        else:
            print('%s[%s]' % ((depth * ' ', node)))

from sklearn.datasets import load_boston
if __name__ == "__main__":

    boston = load_boston()
    data = pd.DataFrame(boston.data)
    data.columns = boston.feature_names
    data['Price'] = boston.target
    data['LSTAT'] = np.log(data['LSTAT'])
    data['RM'] = np.power(data['RM'],2)
    data['CHAS'] = data['CHAS']#.apply(lambda t: 'No' if t==0 else 'Yes')
    node_features = ['ZN','INDUS','CHAS','NOX','AGE','DIS','RAD','TAX','CRIM','PTRATIO','B']
    leaf_features = ['LSTAT','RM']
    target_feature = ['Price']
    tree = MobTree(min_size=40, max_depth=10, node_features=node_features,
                                    leaf_features=leaf_features, trim = 0.1,alpha=0.001)
    tree.fit(np.array(data[node_features + leaf_features + target_feature]))
    tree.print_tree(tree.root)
    result = tree.predict(np.array(data[node_features + leaf_features + target_feature]))
    
    print("Train rms:%.2f, mse:%.2f"%(mean_squared_error(data[target_feature],result,squared=False),mean_squared_error(data[target_feature],result)))
    
