import sys
import uuid
import queue
import pstats
import pickle
import cProfile
import numpy as np
import multiprocessing as mp
from sklearn.base import BaseEstimator
from tree_criterion import divideset,topk_candidate,criterion,divergence

INT_TYPES = [int, np.int8, np.int16, np.int32, np.int64]
FLOAT_TYPES = [float, np.float16, np.float32, np.float64]

def one_hot_encode_array(x):
    unique_elements = np.unique(x)
    length = len(unique_elements)
    one_hot_matrix = np.eye(length)
    one_hot_encoded = one_hot_matrix[x]
    return one_hot_encoded

# 性能分析装饰器定义
def do_cprofile(filename, DO_PROF=True):
    """
    Decorator for function profiling.
    """
    def wrapper(func):
        def profiled_func(*args, **kwargs):
            # Flag for do profiling or not.
            if DO_PROF:
                print('*************** DO PROFILING AT ' + str(filename) + '***************')
                profile = cProfile.Profile()
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                # Sort stat by internal time.
                sortby = "tottime"
                ps = pstats.Stats(profile).sort_stats(sortby)
                ps.dump_stats(filename)
            else:
                print('*************** NOT THING TO PROFILE ***************')
                result = func(*args, **kwargs)
            return result
        return profiled_func
    return wrapper

# empty the multiprocessors because they can be saved
def empty_multiprocessor(fspt):
    fspt.multiproccessor = None
    for i in range(len(fspt.clusters)):
        fspt.clusters[i].mult_processor = None

def save_tree(path, model):
    with open(path,'wb') as f:
        if type(model) is not DecisionTreeClassifier: 
            empty_multiprocessor(model)
        pickle.dump(model, f)

def load_tree(path):
    model = None
    with open(path,'rb') as f:
        model = pickle.load(f)
        if type(model) is not DecisionTreeClassifier: 
            if model.use_multproc:
                mult_processor = mp.Pool(mp.cpu_count())
                model.multiproccessor = mult_processor
                for i in range(len(model.clusters)):
                    model.clusters[i].mult_processor = mult_processor
    return model

#先序计算，返回先序列
def preorder_dfs(node, index):
    if node:
        index.append(node)
        preorder_dfs(node.left_child, index)
        preorder_dfs(node.right_child, index)
    
#中序计算
def inorder_dfs(node, index):
    if node:
        inorder_dfs(node.left_child, index)
        index.append(node)
        inorder_dfs(node.right_child, index)

class Node:
    def __init__(self,
                 left_child=None,
                 right_child=None,
                 parent=None,
                 position='left',
                 explanation=None,
                 selection=None,
                 content={}
            ):
        self.tag = str(uuid.uuid4()) #唯一的标记
        self.left_child = left_child
        self.right_child = right_child
        self.parent = parent
        self.position = position #only 'left' or 'right'
        self.explanation = explanation #only for llmbt
        self.selection = selection #only for llmbt
        self.content = {}
        self.content['is_leave'] = False
        self.content['feature'] = -2 #Feature used for splitting the node
        self.content['threshold'] = -1 #Threshold value at the node
        self.content['impurity'] = 0 #Impurity of the node (i.e., the value of the criterion)
        self.content['n_sample'] = 0 #number of support sample
        self.content['value'] = [] #value for np.mean(Y, axis=0)
        self.content['output_label'] = [] #np.argmax(output_prob)
        if content:
            for k in content:
                self.content[k] = content[k]

class LLMBoostingClassifier(BaseEstimator):
    def __init__(self,
                 criterion='gini', #only 'gini','kl','js','wd' are supported.
                 splitter='best', #only 'best','random','llm'
                 split_call=None, #call if splitter='llm'
                 build_method='bfs', #only 'bfs'
                 max_features=None, #int, float or {“sqrt”, “log2”}, default=None work when splitter is 'random'
                 min_impurity_decrease=0.0, #允许分裂最低增益/熵降
                 max_depth=-1, #允许全局树最深度，与min_impurity_decrease是or关系，出现一者即停止分裂
                 min_samples_split=2, #The minimum number of samples required to split an internal node
                 split_with_softmax=True, #是否在计算相似度前使用softMax
                 use_multproc=False, #use multiprocessing or not
                 feature_name=None,
                 class_name=None,
                 random_state=0,
                 logs=True):
        assert criterion.lower() in ['kl','js','wd','gini']
        assert splitter.lower() in ['best','random','llm']
        assert build_method.lower() in ['bfs']
        #super(DecisionTreeClassifier, self).__init__()
        self.criterion = criterion.lower()
        self.splitter = splitter.lower()
        self.split_call = split_call
        self.build_method = build_method.lower()
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.split_with_softmax = split_with_softmax
        self.random_state = random_state
        self.feature_name = feature_name
        self.feature_importances_ = None
        self.class_name = class_name
        self._is_fitted = False
        self._n_features = 0
        self._n_labels = 0
        self._n_datas = 0
        self.root = None #if none mean not fit
        self.logs = logs
        np.random.seed(random_state)
        self.use_multproc = use_multproc
        if use_multproc:
            #print('tree build using ' + str(mp.cpu_count()) + ' processor.')
            self.multiproccessor = mp.Pool(processes=mp.cpu_count())
        else:
            #print('tree build without multiproccessor. ')
            self.multiproccessor = None

    # 建树的过程
    # 以BFS方式构造树
    # 结束条件：1.数据耗尽 2.没有节点的返回
    def _build_bfs(self, X, y):
        if len(X)==0 : return None
        q = queue.Queue()
        root = Node()
        q.put((root, X, y, 1))
        node_index,last_depth = 0,0
        while(not q.empty()):
            (node, x_, y_, current_depth) = q.get()
            node.tree = self
            reach_max_depth = False
            if current_depth >= self.max_depth and current_depth > 0 and self.max_depth > 0:
                reach_max_depth = True
            if current_depth != last_depth:
                node_index = 0
                last_depth = current_depth
            premise = node.parent.explanation if node.parent else None                
            print('================== depth: ' + str(current_depth-1) + ' index: ' + str(node_index) + ' ==================')
            node_index += 1
            is_leave,res,explanation,selection = self._split(x_, y_, min_impurity_decrease=self.min_impurity_decrease, 
                min_samples_split=self.min_samples_split, is_leave=reach_max_depth, premise=premise)

            #explanation is tuple
            if is_leave:
                node.content = {**node.content, **res} #node has been assign
            else:
                c,data1,data2 = res[0],res[1],res[2]
                node.content = c
                node.selection = selection #selection only for non-leaf
                node.left_child = Node(parent=node, position='left', explanation=explanation[0]) #explanation to children
                node.right_child = Node(parent=node, position='right', explanation=explanation[-1]) #explanation to children
                q.put((node.left_child, data1[0], data1[1], current_depth+1))
                q.put((node.right_child, data2[0], data2[1], current_depth+1))
        return root
    
    # 以熵降的过程计算分裂点
    # 输入：待观察数据与标签，is_leave：是否强制转化为叶子
    # premise: 前提内容str
    # 注意：XY为全局标签（用于RTree），xy为局部标签配合罗生门系数[0,1]（全局是指最终结果，局部是指某层的分类策略）
    # 当xy or xfsp_matrix为空则只考虑全局优化
    # 输出1：False, 节点内容，数据1，数据2
    # 输出2：True, 节点内容
    def _split(self, X, Y, min_impurity_decrease=0.0, min_samples_split=2, is_leave=False, premise=None):
        if len(X)==0 : return True,None

        global_value = np.mean(Y, axis=0)
        output_prob = np.mean(Y, axis=0).tolist()
        output_label = np.argmax(output_prob)
        if len(X) < min_samples_split:
            return True,{'is_leave':True, 'output_prob':output_prob, 'output_label':output_label, 'value':global_value,
                'n_sample':len(X), 'impurity':0, 'threshold':-2},None,None

        candidate_feature,candidate_theshold,candidate_gain,candidate_indice,cache_table,current_gain =\
            criterion(X, Y, method=self.criterion, split_with_softmax=self.split_with_softmax, \
            need_original_gain=True, mult_processor=None) #mult_processor is unavailable in gini
        #按照gain对所有的candidate进行排序
        candidate = zip(candidate_gain, candidate_feature, candidate_theshold, candidate_indice)
        candidate_sort = sorted(candidate, key = lambda x:x[0], reverse=False) #kl越接近0越好
        candidate_gain,candidate_feature,candidate_theshold,candidate_indice = zip(*candidate_sort) #解压排序后的数组们
        
        best_feature, best_theshold, best_gain, best_split = \
            candidate_feature[0],candidate_theshold[0],candidate_gain[0],candidate_indice[0]#全局四元组
        expanation_,selection_ = (),''
        
        if not is_leave: #是否为叶子结点
            if self.splitter == 'best':
                best_feature,best_theshold,best_gain,best_split = \
                    candidate_feature[0],candidate_theshold[0],candidate_gain[0],candidate_indice[0]
            elif self.splitter == 'random':
                topk = 1
                if isinstance(self.max_features, int): #取前max_features名
                    assert self.max_features > 0 #topk k > 0
                    topk = min(self.max_features-1, self._n_features)
                elif isinstance(self.max_features, float): #取前max_features% 【0，1】
                    topk = max(int(self.max_features * self._n_features), 0)
                elif self.max_features == 'sqrt':
                    topk = max(int(self._n_features ** 0.5), 0)
                elif self.max_features == 'log':
                    topk = max(int(np.log(self._n_features)), 0)
                if topk > 1:
                    topk = np.random.randint(0, topk, size=1)[0] #topk is not unique when max_feature > 0
                best_feature,best_theshold,best_gain,best_split = \
                    candidate_feature[topk],candidate_theshold[topk],candidate_gain[topk],candidate_indice[topk]
            else: # LLM Boosting
                splits,selection,explanation = \
                    self.split_call(candidate_feature, candidate_theshold, candidate_gain, candidate_indice, Y, premise)
                best_feature,best_theshold,best_gain,best_split = splits[0],splits[1],splits[2],splits[3]
                expanation_,selection_ = explanation,selection

        #global gain is not ok
        condition1 = best_gain >= min_impurity_decrease and best_gain != sys.float_info.max 
        #exist invalidate candidate gain
        condition2 = not (all(gain <= 0.0 for gain in candidate_gain) or all(gain == sys.float_info.max for gain in candidate_gain)) 
        #exist number of sample of nodes are large than min_samples_split should be considered as node
        condition3 = len(best_split[0]) >= min_samples_split or len(best_split[1]) >= min_samples_split
        #other condition e.g. max depth
        condition4 = is_leave == False

        if condition1 and condition2 and condition3 and condition4: #become a node
            content={'is_leave':False, 'feature':best_feature, 'threshold':best_theshold, 'impurity':current_gain, 'value':global_value,
                    'n_sample':len(best_split[0])+len(best_split[1]), 'output_prob':output_prob, 'output_label':output_label}
            return False, \
                (
                    content, 
                    (X[best_split[0]], Y[best_split[0]]), #data split
                    (X[best_split[1]], Y[best_split[1]])
                ), \
                expanation_, \
                selection_
        else: #基尼系数为0或者所有特征收益相近，表示没有显著的区分特征或者已经完全划分；
            return True,{'is_leave':True, 'output_prob':output_prob, 'output_label':output_label, 'value':global_value,
                'n_sample':len(X), 'impurity':current_gain, 'threshold':-2}, expanation_, selection_
        
    # 获取深度
    def get_depth(self):
        return max(self._get_depth(self.root.left_child), self._get_depth(self.root.right_child))+1
    
    # 获取深度
    def _get_depth(self, node=None):
        if node is None:
            return 0
        else:
            return max(self._get_depth(node.left_child), self._get_depth(node.right_child))+1
    
    # 根据BFS获取所有节点
    def _get_nodes(self, root=None):
        root = self.root if root is None else root
        q = queue.Queue()
        q.put(root)
        nodes = []
        while(not q.empty()):
            node = q.get()
            if node is not None:
                nodes.append(node)
                if node.left_child is not None:
                    q.put(node.left_child)
                if node.right_child is not None:
                    q.put(node.right_child)
        return nodes

    # 根据给定的节点node获取对应的子节点
    def _get_leave(self, node=None):
        leave = []
        q = queue.Queue() #采用bfs
        q.put((self.root if node is None else node))
        while(not q.empty()):
            node = q.get()
            if node.left_child is None and node.right_child is None:
                 leave.append(node)
            else:
                if node.left_child is not None:
                    q.put(node.left_child)
                if node.right_child is not None:
                    q.put(node.right_child)
        return leave
    
    # 根据id寻找相应的节点
    def _find_node_by_tag(self, tag):
        nodes = []
        preorder_dfs(self.root, nodes)
        for i in range(len(nodes)):
            if nodes[i].tag == tag:
                return nodes[i]
        return None

    # 根据根节点到叶子的路径，对数据进行过滤
    # 输入：X全局，INDEX下标
    # 返回下标（全局）
    def _filter_by_node(self, root, leave, X, INDEX):
        paths = [] #记录从root到leave上的节点
        self._find_leave(root, leave, paths) #获取路径
        index = INDEX #输出（全局下标）
        if len(paths)==0:
            return None
        else:
            paths.reverse() #从根高层到低层
            #print([p.content for p in paths])
            for i in range(len(paths)):
                if i+1 < len(paths):
                    index1,index2,_,_ = divideset(X[index], None, \
                        paths[i].content['feature'], paths[i].content['threshold'])
                    assert paths[i+1] == paths[i].left_child or paths[i+1] == paths[i].right_child
                    if paths[i+1] == paths[i].left_child:
                        index = index[index1]
                    else:
                        index = index[index2]
                else:
                    return index
    
    # 配合_filter_by_node工作
    def _find_leave(self, node, leave, paths):
        if node is None or leave is None: return False
        if node.tag != leave.tag:
            if node.left_child is None and node.right_child is None:
                return False
            else:
                if self._find_leave(node.left_child, leave, paths):
                    paths.append(node)
                    return True
                elif self._find_leave(node.right_child, leave, paths):
                    paths.append(node)
                    return True
                else:
                    return False
        else:
            paths.append(node)
            return True

    # Return the id of the leaf that each sample is predicted as.
    # [n_sample] #id
    def apply(self, X):
        return [self._apply_instance(x) for x in X]

    def _apply_instance(self, x): #return the id
        current_node = self.root
        while(not current_node.content['is_leave']):
            feature_ = current_node.content['feature']
            threshold_ = current_node.content['threshold']
            if x[feature_] < threshold_:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child
        return current_node.tag
    
    #change a leaf node to branch node
    def leaf2branch(self, leaf, feature, threshold, left_child, right_child, selection, explanation, n_sample=None):
        leaf.content['is_leave'] = False
        leaf.content['feature'] = feature
        leaf.content['threshold'] = threshold
        leaf.content['impurity'] = -1 #unknown
        leaf.content['value'] = -1 #unknown
        leaf.content['output_label'] = -1 #unknown
        leaf.content['n_sample'] = n_sample
        leaf.selection = selection
        leaf.explanation = explanation
        leaf.left_child = left_child
        leaf.right_child = right_child
        return leaf
    
    #预测函数，根绝给定的x预测，从node开始递归
    def _classify(self, x, node):
        if node.content['is_leave']: #叶子结点
            return node.content['output_label'],node.content['output_prob']
        else:
            next_child = None
            value = x[node.content['feature']]
            if isinstance(value,int) or isinstance(value,float):
                if value < node.content['threshold']: 
                    next_child = node.left_child
                else: 
                    next_child = node.right_child
            else:
                if value==node.content['threshold']: 
                    next_child = node.left_child
                else: 
                    next_child = node.right_child
            return self._classify(x, next_child)
    
    #打印函数
    def _print_node(self, node, indent, feature_name=None):
        if node.content['is_leave']:
            print('*' + ''.join(indent) + ' : ' + str(node.content['n_sample']) + ' ' + 'label: '\
                  + str(node.content['output_label']) + ' prob:' + str(np.round(node.content['output_prob'], 3))\
                  + ' ' + self.criterion + ' : ' + str(np.round(node.content['impurity'], 3)))
        else:
            if isinstance(node.content['threshold'], int) and isinstance(node.content['threshold'], float):
                if feature_name:
                    print(' ' + ''.join(indent) + ' : ' + feature_name[node.content['feature']] +
                          ' < ' + str(np.round(node.content['threshold'], 3)) + ' ' + self.criterion + ' : ' +\
                          str(np.round(node.content['impurity'], 3)) + ' prob: ' + str(np.round(node.content['output_prob'], 3)))
                else:
                    print(' ' + ''.join(indent) + ' : ' + 'feature_' + str(node.content['feature']) + 
                          ' < '  + str(np.round(node.content['threshold'], 3)) + ' ' + self.criterion + ' : ' +\
                          str(np.round(node.content['impurity'], 3)) + ' prob: ' + str(np.round(node.content['output_prob'], 3)))
            else:
                if feature_name:
                    print(' ' + ''.join(indent) + ' : ' + feature_name[node.content['feature']] + 
                          ' < ' + str(np.round(node.content['threshold'], 3)) + ' ' + self.criterion + ' : ' +\
                          str(np.round(node.content['impurity'], 3)) + ' prob: ' + str(np.round(node.content['output_prob'], 3)))
                else:
                    print(' ' + ''.join(indent) + ' : ' + 'feature_' + str(node.content['feature']) + 
                          ' < '  + str(np.round(node.content['threshold'], 3)) + ' ' + self.criterion + ' : ' +\
                          str(np.round(node.content['impurity'], 3)) + ' prob: ' + str(np.round(node.content['output_prob'], 3)))
        
    def _feature_importance(self, nodes=None):
        if nodes is None:
            nodes = self._get_nodes(self.root)
        feature_importances_ = np.zeros(self._n_features)
        for i in range(len(nodes)):
            node = nodes[i]
            if node.left_child is not None and node.right_child is not None:
                feature = node.content['feature']
                current_gain = node.content['impurity']
                wc = node.content['n_sample']/self._n_datas
                wl = node.left_child.content['n_sample']/self._n_datas
                wr = node.right_child.content['n_sample']/self._n_datas
                left_gain,right_gain = node.left_child.content['impurity'],node.right_child.content['impurity']
                reduce_gain = wc*current_gain - wl*left_gain - wr*right_gain
                feature_importances_[feature] += reduce_gain
        if np.sum(feature_importances_) > 0.0:
            feature_importances_ /= np.sum(feature_importances_)
        return feature_importances_

    #@do_cprofile("./DECISION_TREE.prof", False)
    def fit(self, X, y): 
        #X, y = check_X_y(X, y, multi_output=True)
        if len(X) == 0:
            self._n_features = 0
            self._n_labels = 0
            self._n_datas = 0
            self._is_fitted = False
            self.root = Node()
        else:
            if y.ndim == 1:
                y = one_hot_encode_array(y.copy())
            self._n_features = len(X[0])
            self._n_labels = len(y[0])
            self._n_datas = len(X)
            assert self.build_method.lower() in ['bfs','dfs']
            if self.build_method == 'bfs':
                self.root = self._build_bfs(np.array(X), np.array(y))
            else:
                self.root = self._build_dfs(np.array(X), np.array(y), 1)
            self.feature_importances_ = self._feature_importance()
            self._is_fitted = True
        
    def predict(self, X):
        #X = check_array(X)
        assert len(X) > 0 and len(X[0]) == self._n_features
        result = []
        for i in range(len(X)):
            result.append(self._classify(X[i], self.root)[0]) #[0]是指获取label,[0][0]是指获取最大可能的那个标签
        return result
        
    def predict_proba(self, X):
        #X = check_array(X)
        assert len(X) > 0 and len(X[0]) == self._n_features
        result = []
        for i in range(len(X)):
            _,prob = self._classify(X[i], self.root)
            result.append(list(prob))
        return result
        
    def score(self, X, y, criterion='acc'):
        assert len(X) == len(y)
        if criterion.lower() == 'acc':
            res = self.predict(X) #maximum possibility
            if y.ndim == 1:
                return np.sum(np.array(np.array(res)==y, dtype=np.int16))/len(y)
            else:
                return np.sum(np.array(np.array(res)==np.argmax(y, axis=-1), dtype=np.int16))/len(y)
        else:
            assert criterion.lower() in ['kl','js','wd','gini'] 
            res = self.predict_proba(X) #maximum possibility
            return np.mean(divergence(res, y, criterion))

    def export_text(self, from_node=None,feature_name=None):
        depth = 0
        q = queue.Queue()
        if from_node is None:
            q.put((self.root, depth))
        else:
            q.put((from_node, depth))
        indent = []
        if feature_name:
            assert len(feature_name) == self._n_features
        while(not q.empty()):
            node,depth = q.get()
            indent = ' |' * depth
            self._print_node(node, indent, feature_name)
            if node.left_child: q.put((node.left_child, depth+1))
            if node.right_child: q.put((node.right_child, depth+1))
