import numpy as np



# unweighted accuracy
def UA(pred, ground):
    # convert to one hot encoding if neccessary
    if len(ground.shape) != 2:
        ground = ground.astype(int)
        num_class = np.max(ground) + 1
        ground = np.eye(num_class)[ground]
        
    correct = ground[ground.argmax(axis=1) == pred.argmax(axis=1)].sum(axis=0)
    total   = ground.sum(axis=0)
    
    accuracies = correct / total
    
    return accuracies.mean()


# weighted accuracy
def WA(pred, ground):
    # convert to one hot encoding if neccessary
    if len(ground.shape) != 2:
        ground = ground.astype(int)
        num_class = np.max(ground) + 1
        ground = np.eye(num_class)[ground]
        
    correct = ground[ground.argmax(axis=1) == pred.argmax(axis=1)].sum(axis=0)
    total   = ground.sum(axis=0)

    return correct.sum() / total.sum()


# unweighted f1 score
def U_F1(pred, ground):
    # convert to one hot encoding if neccessary
    if len(ground.shape) != 2:
        ground = ground.astype(int)
        num_class = np.max(ground) + 1
        ground = np.eye(num_class)[ground]
        
    correct    = ground[ground.argmax(axis=1) == pred.argmax(axis=1)].sum(axis=0)
    total_gt   = ground.sum(axis=0)
    total_pred = pred.sum(axis=0)
    
    # calculate the f1 score, add epsilon to avoid zero division
    recall    = correct / total_gt
    precision = correct / total_pred
    f1 = 2 * recall * precision / ((recall + precision) + np.finfo(float).eps)
    
    return (f1 * total_gt).sum() / total_gt.sum()


# weighted f1 score
def W_F1(pred, ground):
    # convert to one hot encoding if neccessary
    if len(ground.shape) != 2:
        ground = ground.astype(int)
        num_class = np.max(ground) + 1
        ground = np.eye(num_class)[ground]
        
    correct = ground[ground.argmax(axis=1) == pred.argmax(axis=1)].sum(axis=0)
    total_gt   = ground.sum(axis=0)
    total_pred = pred.sum(axis=0)
    
    # calculate the f1 score, add epsilon to avoid zero division
    recall    = correct / total_gt
    precision = correct / total_pred
    f1 = 2 * recall * precision / ((recall + precision) + np.finfo(float).eps)
    
    return f1.mean()
