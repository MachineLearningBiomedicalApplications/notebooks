#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 13:02:36 2018

Decision Tree Predictor, using gini coefficient as cost

@author: E Robinson
"""
import numpy as np
from  copy import deepcopy 
import sys

def gini_coefficient(class_membership):
    
    """
        Estimates Gini Coefficient for a given class split
        input:
            class_membership: list of length k (where k= number of classes)
                              values at each index reflect number of instances 
                              of each class, for this proposed split
                             
        output:
            gini: gini coefficient for this split 
    """
    # estimating total number of samples in split (by summing contents of class_membership list)
    split_size=np.sum(class_membership)
    gini=1
    # iterating over all items in the array
    for class_total in class_membership :
        # estimating p*p for this class label; subtracting from current gini total
        gini-=(class_total/split_size)*(class_total/split_size)
    
        
    return gini

def split_cost(split,classes): 
    
    """
        Estimates the cost for a proposed split 
        input:
            splits: tuple or form (L,R) where L reflects the data for the left split and
                    R reflects data for left split
            classes: list of class values i.e. [0,1]
                             
        output:
            cost: sum of gini coefficient for left and right sides of the split
    """
    cost=0
    total_samples=0
    
    # estimate the relative size of each branch
    for branch in split:
        total_samples+=branch.shape[0]
    
    # for each (left/right) split on the proposed tree
    for br_index,branch in enumerate(split):
        # initialise list of class counts for this branch
        class_counts_for_branch=[]
        # for each class value, count total of data examples (rows) that have for this class, in this branch 
        for class_val in classes:
            
            if branch.shape[0] == 0: # don't continue if size of split is 0
                continue
           
            # slice data to return only rows with this specific class value  
            branch_per_class=branch[np.where(branch[:,-1]==class_val)]
            # sum up the number of rows in for this class in this branchand append 
            total_rows=branch_per_class.shape[0]
            class_counts_for_branch.append(total_rows)

        # estimate the gini coefficient for this split   
        cost+=gini_coefficient(class_counts_for_branch)* (branch.shape[0]/total_samples)
                        
        
    return cost

def test_split(index, value, dataset):
    """
        Split a dataset based on an attribute and an attribute value 
        input:
            index = feature/attribute index (i.e. data column index) on which to split on
            value = threshold value (everything below this goes to left split, 
                    everything above goes to right)
            dataset = array (n_samples,n_features+1) 
                    rows are examples 
                    last column indicates class membership
                    remaining columns reflect features/attributes of data
                             
        output:
            left,right: data arrays reflecting data split into left and right branches, respectively
    """
    left=[]
    right = []
    for row in dataset:
        # if the value of this feature for this row is less than the (threshold) value, 
        # split into left branch, else split into right
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
	
    return np.asarray(left), np.asarray(right)

def get_best_split(dataset):
    """
        Search through all attributes and all possible thresholds to find the best split for the data
        input:
            dataset = array (n_samples,n_features+1) 
                    rows are examples 
                    last column indicates class membership
                    remaining columns reflect features/attributes of data
                             
        output:
            dict containing: 1) 'index' : index of feature used for splittling on
                             2)  'value': value of threshold split on
                             3) 'branches': tuple of data arrays reflecting the optimal split into left and right branches
                             
    """
    class_values=np.unique(dataset[:,-1])
    # initalising optimal values prior to refinment
    best_cost=sys.float_info.max # initialise to max float
    best_value=sys.float_info.max # initialise to max float
    best_index=dataset.shape[1]+1 # initialise as greater than total number of features
    best_split=tuple() # the best_split variable should contain the output of test_split that corresponds to the optimal cost

    #iterating over all features/attributes (columns of dataset)
    for index in np.arange(dataset.shape[1]-1):

        #Trialling splits defined by each row value for this attribute
        for r_index,row in enumerate(dataset):
            branches=test_split(index, row[index], dataset)

            cost=split_cost(branches,class_values)
            if cost < best_cost:
                best_cost=cost
                best_split=branches
                best_index=index
                best_value=row[index]
                #print('Best cost={}; Best feature={}; Best row={}'.format(best_cost,index,r_index) )
                
    return {'index':best_index, 'value':best_value, 'branches':best_split}


# Create a terminal node value
def to_terminal(group):
    
    """
        Assigns a label according to the most common class label of the data
        input:
            group = array (n_samples,n_features+1) 
                    rows are examples 
                    last column indicates class membership
                    remaining columns reflect features/attributes of data
                             
        output:
            class label for this terminal node
    """
    outcomes = group[:,-1]
    counts = np.bincount(outcomes.astype(int))
    return np.argmax(counts)
              
def run_split(node, max_depth, min_size, depth):
     
    """
        Recursively splits nodes until termination criterion is met
        input:
            node = dict containing: 1) 'index' : index of feature used for splittling on
                             2)  'value': value of threshold split on
                             3) 'branches': tuple of data arrays reflecting the optimal split into left and right branches
            max_depth: int determining max allowable depth for the tree
            min_size : int determining minimum number of examples allowed for any branch
            depth: current depth of tree              
            
            
        Output:
            node: is returned by value and returns a recursion of dicts representing the structure of the whole tree
    """
    left, right = node['branches']
    del(node['branches'])
    # check for whether all data has been assigned to one branch; if so assign both branches the same label
    if left.shape[0]==0 :
        node['left'] = node['right'] = to_terminal(right)       
        return
    if right.shape[0]==0 :
        node['left'] = node['right'] = to_terminal(left)       
        return
    # check for max depth; if exceeded then estimate labels for both branches
    if max_depth != None and depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
        # in first instance check whether the number of examples reaching the left node are less than the allowed limit
        # if so assign as a terminal node, if not then split again
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_best_split(left)
        run_split(node['left'], max_depth, min_size, depth+1)
    
    # process right child as for left
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_best_split(right)
        run_split(node['right'], max_depth, min_size, depth+1)
        
def build_tree(train, max_depth=None, min_size=1):
    """
    Builds and returns final decision tree
    
    input:
        train : training data array (n_samples,n_features)
        max_depth: user defined max tree depth (int)
        min_size: user defined minimum number of examples per tree tree depth (int)
    """
    # create a root node split by calling get_best_split on the full training set
    root = get_best_split(train)
    # now build the tree using run_split
    run_split(root, max_depth, min_size, 1)
    return root

def print_tree(node, depth=0):
    """
    Print a decision tree, by interogating node branches recursively
    
    input:
        node = dict containing: 1) 'index' : index of feature used for splittling on
                                2)  'value': value of threshold split on
                                3) 'branches': tuple of data arrays reflecting the optimal split into left and right branches
        depth = current depth of tree        
       
    """
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))
            
def predict_row(node, row):
    
    """
    Predict from a decision tree, by interogating node branches recursively
    
    input:
        node = decision tree represented as dict containing: 
                1) 'index' : index of feature used for splittling on
                2)  'value': value of threshold split on
                3) 'branches': tuple of data arrays reflecting the optimal split into left and right branches
        row       
       
    """
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict_row(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict_row(node['right'], row)
        else:
            return node['right']

def predict(tree,testdata):
    
    predictions=[]
    for row in testdata:
        node=deepcopy(tree)
        predictions.append(predict_row(node, row))
        
    return predictions

def score(testlabels, prediction):
    
    score=0
    for i in np.arange(len(testlabels)):
        if testlabels[i]==prediction[i]:
            score+=1
           
    return score/len(testlabels)

dataset = np.asarray([[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,0],
	[3.961043357,2.61995032,0],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]])


testdata=np.asarray([8.5,4.32,1]).reshape((1,3))
#split = get_best_split(np.asarray(dataset) )         
# =============================================================================
# 
# tree = build_tree(np.asarray(dataset), 3, 1)
# print('Decision Tree: \n {}'.format(tree))
# 
# print_tree(tree)
# prediction=predict_row(tree, testdata)
# print('Expected=%d, Got=%d' % (testdata[:,-1], prediction))
# 
# 
# 
# =============================================================================
