import sys
import numpy as np
from collections import Counter
from scipy.special import kl_div
from scipy.special import softmax
from scipy.spatial import distance
from scipy.stats import wasserstein_distance

#Tools for unique
#统计标签
def unique_label(labels, sort=False):
    label_set = set()
    if labels is not None:
        for l in labels:
            label_set.add(l)
    if sort:
        label_set = list(label_set)
        label_set.sort(reverse=False)
    return label_set

#统计Y的种类 Y{'apple':n1, 'orange':n2, 'apple':n3, ...}
def unique_count(Y, category=None):
    results = {}
    if category is not None:
        for c in category:
            results[c] = 0
    for y in Y:
        if type(y) == int or isinstance(y, np.integer):
            if y not in results:results[y] = 0
            results[y]+=1
        else:
            y_ = np.argmax(y)
            if y_ not in results:results[y_] = 0
            results[y_]+=1
    return results # 返回一个字典

#统计不同y分布的总类，errors为误差的小数位
def unique_arr_count(Y, errors=3):
    results = {}
    for y in Y:
        y_ = str(np.round(y, errors))[1:-1].replace('.','')
        if y_ not in results:results[y_] = 0
        results[y_]+=1
    return results # 返回一个字典

# 用于缓存，加速计算
# 返回说明：长度m=len(X)，n=unique(X[feature])
# medians：[medians*n] sorted[->]
# threshold_indice：[index*m] sorted[->] 按照特征值排序后的下标
# unique_threshold_indice：{unique(X[feature]):index}*n 用于查询index缓存
# threshold_prob：[index*prob] shape=m*len(unique(key))
def caching_feature(X, Y, feature):
    medians = [] #前后平均数[<=N]
    threshold_indice = [] #[特征值：下标] 例如[[low,1],[median,2],[high,3],[median,4]] [N]
    threshold_prob_left = [] #[特征值：[概率] 例如[low,[0.1,0.5,0.4]] 从左往右计数
    threshold_prob_right = [] #从右往左计数
    threshold_prob_sum = np.sum(Y, axis=0)
    unique_threshold_indice = {} #直接索引{median: 最末位下标} 索引最低为1
    for i in range(len(X)): #>0
        ti = [X[i][feature], i]
        threshold_indice.append(ti)
    threshold_indice = sorted(threshold_indice, key=lambda x:x[0]) #按照特征值进行低到高排序[[low,1],[median,2],[median,4],[high,3]]
    threshold_prob_left.append(np.zeros_like(Y[0])) #追加第一个分类索引 len > 2
    threshold_prob_right.append(threshold_prob_sum) # len = len(left)
    last_threshold = threshold_indice[0][0]
    for i in range(1, len(threshold_indice)): #建立median和快速键值 i>0
        current_threshold = threshold_indice[i][0]
        if current_threshold != last_threshold:
            median = (last_threshold + current_threshold)/2
            unique_threshold_indice[median] = i #必须为相同特征值的最后一个计数
            last_threshold = current_threshold
            medians.append(median)
        threshold_prob_left.append(threshold_prob_left[i-1] + Y[threshold_indice[i-1][1]])
        threshold_prob_right.append(threshold_prob_right[i-1] - Y[threshold_indice[i-1][1]])
        threshold_prob_left[i-1] /= (float(i-1) if i > 1 else 1)
        threshold_prob_right[i-1] /= float(len(threshold_indice) - i + 1)
    threshold_prob_left.append(threshold_prob_left[len(threshold_indice)-1] + Y[threshold_indice[len(threshold_indice)-1][1]])
    threshold_prob_left[len(threshold_indice)-1] /= float(len(threshold_indice)-1)
    threshold_prob_left[-1] /= float(len(threshold_indice))
    threshold_prob_right.append(np.zeros_like(Y[0]))
    threshold_indice = [threshold_indice[i][1] for i in range(len(threshold_indice))] #去除特征列,[1,2,4,3]
    return medians, threshold_indice, unique_threshold_indice, threshold_prob_left, threshold_prob_right

#集合划分系列
#index1,index2,_,_ = divideset(X[index], None, paths[i].content['feature'], paths[i].content['threshold'])
#input: X  [N, features]
#       Y  [N, probability distribution]
#output x1,x2 [N1, features],[N2, features]
#       p1,p2 [labels],[labels]
def divideset(X, Y, feature, threshold, 
              threshold_indice=None, median_indice=None, threshold_prob_left=None, threshold_prob_right=None):
    #定义一个函数，判断当前数据行属于第一组还是第二组
    split_function = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_function = lambda x:x[feature] < threshold
    else:
        split_function = lambda x:x[feature] == threshold #如果特征不是数值，那么将使用是否等同进行判断
    # 将数据集拆分成两个集合，并返回
    x1,x2,p1,p2 = [],[],None,None
    X_,Y_ = X.copy(),(Y.copy() if Y is not None else Y)
    if threshold_indice is None or median_indice is None or \
        threshold_prob_left is None or threshold_prob_right is None: #没有索引
        if Y is None: #Y可以不提供
            for i in range(len(X_)):
                x = X_[i]
                if split_function(x):
                    x1.append(i)
                else:
                    x2.append(i)
        else:
            p1,p2 = [],[]
            for i in range(len(X_)):
                x = X_[i]
                y = Y_[i] #y is a vector
                if split_function(x):
                    x1.append(i)
                    p1.append(y)
                else:
                    x2.append(i)
                    p2.append(y)
            p1,p2 = np.mean(p1,axis=0) if len(p1) > 0 else np.array(p1),\
                np.mean(p2,axis=0) if len(p2) > 0 else np.array(p2) #可能存在数组被划分为空
        return x1,x2,p1,p2 
    else: #存在索引
        divide_index = median_indice[threshold] #divide_index >= 1
        x1 = threshold_indice[:divide_index]
        x2 = threshold_indice[divide_index:]
        p1,p2 = None,None
        if threshold_prob_left is not None and threshold_prob_right is not None:
            p1 = threshold_prob_left[divide_index]
            p2 = threshold_prob_right[divide_index]
        return x1,x2,p1,p2

# retrieve topk candidates by their gain for optimization of rashomon feature
# input: cache_table //from above, rashomon_gain //base gain
# output: features, thresholds, gains
def topk_candidate(cache_table, max_gain, sampling=1.0):
    rashomon_features,rashomon_threshold,rashomon_gain = [],[],[]
    for feature in cache_table: #GLOBAL IMPURITY [N * F] very large
        for value,gain in zip(cache_table[feature]['value'], cache_table[feature]['gain']):
            if gain <= max_gain: #search the maxmium gain of each features
                rashomon_features.append(feature)
                rashomon_threshold.append(value)
                rashomon_gain.append(gain)
    if sampling < 1.0 and len(rashomon_features)*sampling > 1: #sampling from Rashomon Features which will lower the FSP calculation
        sampling_indice = np.random.choice(range(len(rashomon_features)), size=int(len(rashomon_features)*sampling), replace=False)
        rashomon_features = np.array(rashomon_features)[sampling_indice].tolist()
        rashomon_threshold = np.array(rashomon_threshold)[sampling_indice].tolist()
        rashomon_gain = np.array(rashomon_gain)[sampling_indice].tolist()
    return rashomon_features,rashomon_threshold,rashomon_gain

#gini
# [N, LABEL]
# gini = 1-(p1)^2-(p2)^2...
# output: gini [0,1]
def gini(YL, YR=None):
    assert YL is not None or YR is not None
    gini_left,gini_right = 0.0,0.0
    YL = [] if YL is None else np.argmax(YL, axis=-1)
    YR = [] if YR is None else np.argmax(YR, axis=-1)
    unique_y_left,unique_y_right = Counter(YL),Counter(YR)
    for k in unique_y_left:
        gini_left += (unique_y_left[k]/len(YL))**2
    for k in unique_y_right:
        gini_right += (unique_y_right[k]/len(YR))**2
    return (len(YL)/(len(YL)+len(YR))) * (1.0 - gini_left) + (len(YR)/(len(YL)+len(YR))) * (1.0 - gini_right)

#using divergence default is KL DIV
def divergence(Y1, Y2, method='kl', split_with_softmax=True):
    assert Y1 is not None and Y2 is not None
    if len(Y1)==0 or len(Y2)==0: #防止出现空数组
        return sys.float_info.max
    else:
        if split_with_softmax:
            Y1_,Y2_ = softmax(Y1, axis=-1), softmax(Y2, axis=-1)
        else:
            Y1_,Y2_ = Y1,Y2
        if method == 'kl':
            return np.sum(kl_div(Y1_, Y2_), axis=-1) #每行数据的和 [n,d] -> n
        elif method == 'js':
            return (np.sum(kl_div(Y1_, Y2_), axis=-1)+np.sum(kl_div(Y2_, Y1_), axis=-1))/2.0
        else:
            wd = 0
            for y1,y2 in zip(Y1_,Y2_):
                wd += wasserstein_distance(y1,y2)
            return wd

#using fsp loss
def fsp_worker(data_chunk):
    loss_chunk = []
    Y_, Y, G = data_chunk['Y1'],data_chunk['Y2'],data_chunk['G']
    for i in range(len(G)):
        loss_chunk.append(np.sum((np.outer(Y_[i], Y) - G[i])**2))
    return loss_chunk

'''
Y_: [N, C_]
Y: [C] #because all leaves are the same probablity
G: [N, C_, C]
only fsp_loss for mult_processor
'''
def fsp_loss(Y_, Y, G, mult_processor=None):
    assert Y_ is not None and Y is not None
    assert Y_.shape[-1] == G.shape[1]
    assert len(G) == len(Y_)
    assert len(Y) == G.shape[2] #Y is of [d] 
    if len(Y_)==0 or len(Y)==0: #prevent null array
        return sys.float_info.max
    else:
        loss = []
        if mult_processor is not None:
            batch_num = mult_processor._processes
            if len(G) > batch_num:
                data_chunks = [] #euqal to batch_num
                index = list(range(len(G)))
                index_chunks = np.array_split(index, batch_num)
                for i in range(batch_num):
                    data_chunks.append({'Y1':Y_[index_chunks[i]],\
                         'Y2':np.array(Y), 'G':G[index_chunks[i]]})
                batch_results = mult_processor.map(fsp_worker, data_chunks)
                for i in range(len(batch_results)):
                    loss += batch_results[i]
                return loss
        single_chunk = {'Y1':Y_, 'Y2':Y, 'G':G}
        loss = fsp_worker(single_chunk)
        return loss

#find the most frequent value
def find_frequent_value(elements):
    counter = Counter(elements)
    # Find the most common element and its frequency
    most_common_element = counter.most_common(1)[0]  # Returns a tuple (element, count)
    most_common_value = most_common_element[0]
    return most_common_value

'''
input: X  [N, features]
       Y  [N, probability distribution] or [2, N, probability distribution of last/current layer] when G is not None
       G  [N, probability distribution of last layer, probability distribution of current layer]
output: feature,threshold,loss,index,cache,original_gains if need
'''
def criterion(X, Y, method='kl', features=None, thresholds=None, \
                split_with_softmax=True, need_original_gain=False, mult_processor=None):
    candidate_div = []
    candidate_feature = []
    candidate_theshold = []
    #candidate_probability = [] 可能不需要使用
    candidate_indice = [] #[(indice_x1, indice_x2)]
    feature_count = len(X[0]) #特征数
    label_count = len(Y[0])
    y_categories = unique_arr_count(Y)
    original_gains = None
    if need_original_gain:
        if method == 'gini':
            original_gains = gini(Y)
        else:
            y_values = np.argmax(Y, axis=-1)
            freq_value = find_frequent_value(y_values)
            prob = np.zeros(label_count)
            prob[freq_value] = 1.0
            original_gains = np.mean(divergence(Y, prob, method, split_with_softmax))
    if features is not None and thresholds is not None: #给定特征与阈值，直接计算gain
        assert len(features) == len(thresholds)
        gains,feature_value_div = [], None
        for feature,threshold in zip(features, thresholds):
            indexL,indexR,probL,probR = divideset(X, Y, feature, threshold) #根据给定的特征和阈值进行划分
            if method == 'gini':
                feature_value_div = gini(Y[indexL], Y[indexR])
            else:
                feature_value_div = np.mean(divergence(Y[indexL], probL , method, split_with_softmax))\
                                    *(len(indexL)/len(indexL+indexR)) + \
                                    np.mean(divergence(Y[indexR], probR, method, split_with_softmax))\
                                    *(len(indexR)/len(indexL+indexR))
            gains.append(feature_value_div)
        return gains,original_gains
    else:
        cache_table = {} #{feature_index:{'value':[], 'gain':[]}
        feature_indice = list(range(0, feature_count))
        np.random.shuffle(feature_indice)
        for feature in feature_indice: #遍历特征
            cache_feature = {'value':[],'gain':[]} #已经排好顺序
            medians, threshold_indice, unique_threshold_indice, threshold_prob_left, threshold_prob_right =\
                caching_feature(X, Y, feature)
            #根据这一列中的每个值，尝试对数据集进行拆分
            feature_candidate_div = sys.float_info.max if len(y_categories) > 1 else 0.0
            feature_candidate_theshold = -1
            feature_candidate_indice = ([],[])
            #feature_candidate_probability = ([],[])
            for i in range(len(medians)): #遍历特征中位数值
                value = medians[i]
                feature_value_div = None
                indexL,indexR,probL,probR = divideset(X, Y, feature, value, \
                    threshold_indice=threshold_indice, median_indice=unique_threshold_indice, \
                    threshold_prob_left=threshold_prob_left, threshold_prob_right=threshold_prob_right) #根据该第feature个特征的阈值进行划分
                if method == 'gini':
                    feature_value_div = gini(Y[indexL], Y[indexR])
                else:
                    feature_value_div = np.mean(divergence(Y[indexL], probL, method, split_with_softmax))\
                                        *(len(indexL)/len(indexL+indexR)) + \
                                        np.mean(divergence(Y[indexR], probR, method, split_with_softmax))\
                                        *(len(indexR)/len(indexL+indexR))
                cache_feature['value'].append(value)
                cache_feature['gain'].append(feature_value_div)
                if feature_value_div < feature_candidate_div:
                    feature_candidate_div = feature_value_div
                    feature_candidate_indice = (indexL,indexR)
                    feature_candidate_theshold = value
                    #feature_candidate_probability = (probL,probR)                    
            candidate_feature.append(feature)
            candidate_div.append(feature_candidate_div)
            candidate_indice.append(feature_candidate_indice)
            candidate_theshold.append(feature_candidate_theshold)
            #candidate_probability.append(feature_candidate_probability)
            cache_table[feature] = cache_feature
        return candidate_feature,candidate_theshold,candidate_div,candidate_indice,cache_table,original_gains
