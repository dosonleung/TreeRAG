import queue
import sklearn
import colorsys
import numpy as np
import matplotlib as mpl
import sklearn_json as skljson
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
#from stat import *

classification_colors = ['red', 'blue', 'darkgreen', 'orange', 'pink', 'purple']
strategy_colors = ['red', 'blue', 'darkgreen', 'orange', 'pink', 'purple']

def generate_colors(amount, brightness=0.5):
    assert amount <= 10
    colors = []
    for i in range(amount):
        hue = i/amount
        # Convert the HSV color into RGB values
        rgb = colorsys.hsv_to_rgb(hue, 1, 1)
        # Adjust brightness based on the provided parameter
        r, g, b = [int(255 * brightness * val) for val in rgb]
        # Convert the RGB values into hex format and append it to the list
        colors.append('#' + ''.join([('%02x' % val) for val in (r, g, b)]))
    return colors

def to_skl_tree(tree, is_classifier=True):
    tree_dict = {}
    #common part
    tree_dict['n_features_in_'] = tree._n_features
    tree_dict['feature_importances_'] = tree.feature_importances_
    tree_dict['max_features_'] = tree._n_features
    tree_dict['n_outputs_'] = 1
    tree_dict['tree_'] = {}
    tree_dict['tree_']['max_depth'] = get_depth(tree.root)-1
    tree_dict['tree_']['nodes'],tree_dict['tree_']['values'] = create_nodes_values(tree, is_classifier)
    tree_dict['tree_']['node_count'] = len(tree_dict['tree_']['nodes'])
    tree_dict['tree_']['nodes_dtype'] = ['<i8', '<i8', '<i8', '<f8', '<f8', '<i8', '<f8'] #目测是固定
    tree_dict['params'] = {
        'ccp_alpha': 0.0,
        'class_weight': None,
        'criterion': tree.criterion,
        'max_depth': tree.max_depth,
        'max_features': tree.max_features,
        'max_leaf_nodes': None,
        'min_impurity_decrease': tree.min_impurity_decrease,
        'min_samples_leaf': 2,
        'min_samples_split': tree.min_samples_split,
        'min_weight_fraction_leaf': 0.0,
        'random_state': tree.random_state,
        'splitter': tree.splitter
    }
    #classifier part
    if is_classifier:
        tree_dict['meta'] = 'decision-tree'
        tree_dict['n_classes_'] = len(tree.class_name)
        tree_dict['classes_'] = list(tree.class_name)
    else:
        tree_dict['meta'] = 'decision-tree-regression'
    return tree_dict
    
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
    
#获取深度
def get_depth(node):
    if node:
        return max(get_depth(node.left_child), get_depth(node.right_child))+1
    return 0
    
#先序遍历树，同时计算 
#1.序号 2.前驱节点 3.置0 4.threshold 5.impurity 6.n_samples 7.n_samples
#以及values
def create_nodes_values(tree, is_classifier):
    preorder_nodes = []
    preorder_dfs(tree.root, preorder_nodes) #获取了先序遍历下的node序列
    #对先序遍历进行下标
    for i in range(len(preorder_nodes)):
        preorder_nodes[i].content['preorder_index'] = i #把前序的id放置到nodes中
    #创建nodes
    nodes = [[0]*7 for i in range(len(preorder_nodes))]
    #填充第一、二列
    for i in range(len(preorder_nodes)):
        if preorder_nodes[i].left_child:
            nodes[i][0] = preorder_nodes[i].left_child.content['preorder_index']
        else:
            nodes[i][0] = -1
        if preorder_nodes[i].right_child:
            nodes[i][1] = preorder_nodes[i].right_child.content['preorder_index']
        else:
            nodes[i][1] = -1
    #填充第三,四，五，六，七列 【第三列统一填充为0】
    for i in range(len(preorder_nodes)):
        nodes[i][2] = preorder_nodes[i].content['feature']
        nodes[i][3] = preorder_nodes[i].content['threshold']
        nodes[i][4] = preorder_nodes[i].content['impurity']
        nodes[i][5] = preorder_nodes[i].content['n_sample']
        nodes[i][6] = float(preorder_nodes[i].content['n_sample'])
    #创建values
    values = []
    for i in range(len(preorder_nodes)):
        if is_classifier:
            values.append([preorder_nodes[i].content['value']])
        else:
            values.append([[preorder_nodes[i].content['threshold']]])
    return nodes, values

# transfer tree to sklearn tree
def to_sklearn_tree(tree):
    tree_dict = skljson.to_dict(tree)
    return skljson.from_dict(tree_dict)

# 返回，区分值
# 4,0.33 前者为区分下标，后者为透明度
# -1 不能区分
def get_color_intense(value):
    if len(value) == 0: return -1
    value = np.array(value)
    index = np.where(value==np.max(value))[0] #index 是二维数组，坑
    if len(index) < len(value):
        return value[index[0]]/np.sum(value)
    else:
        return -1

def color_gradient(c1, c2, mix=0): 
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c2 + mix*c1)

#### 获取树的节点
def get_nodes(tree):
    nodes = []
    q = queue.Queue() #采用bfs
    q.put(tree.root)
    while(not q.empty()):
        node = q.get()
        nodes.append(node)
        if node.left_child is not None:
            q.put(node.left_child)
        if node.right_child is not None:
            q.put(node.right_child)
    return nodes

def tree_legend(ax, tree, fontsize, alpha, size):
    legend = ax.legend(fontsize=fontsize, fancybox=True, framealpha=0.0, labelspacing=1.5)
    legend.get_frame().set_edgecolor('#FFFFFF00')
    legend.get_frame().set_facecolor('#FFFFFF00')

#output_prob
def tree_regressor_visual(tree, artists, feature_names, class_names, show_content, max_alpha=1.0, min_alpha=0.0):
    preorder_nodes = []
    preorder_dfs(tree.root, preorder_nodes)
    assert len(preorder_nodes)==len(artists)
    assert max_alpha >= 0.0 and max_alpha <= 1.0
    assert min_alpha >= 0.0 and min_alpha <= 1.0
    assert max_alpha > min_alpha
    for i in range(len(preorder_nodes)):
        text = ''
        node = preorder_nodes[i]
        box = artists[i].get_bbox_patch()
        #box.set_boxstyle('round', rounding_size=1.0)
        if 'feature' in node.content:
            if node.content['feature'] >= 0:
                feature_name = feature_names[node.content['feature']]
                if len(feature_name) > 10:
                    feature_name = feature_name[:9] + '...'
                text += feature_name + '<=' + str(np.round(node.content['threshold'], 3)) + '\n' 
            else:#is leave
                box.set_linestyle('dashed')
        value_str = str(list(np.round(node.content['output_prob'], 2))).replace(' ', '')
        if len(node.content['output_prob']) > 3:
            value_split = list(i for i,value in enumerate(value_str) if value == ',')
            value_str = value_str[:value_split[2]] + '\n' + value_str[value_split[2]:]
        if node.content['is_leave']:
            text += 'value:' + '\n' + value_str
        else:
            if 'gain' in show_content:
                text += 'gain:' + str(np.round(node.content['impurity'], 4)) + '\n'
            text += 'value:' + '\n' + value_str
        if 'output_prob' in show_content:
            text += '\n' + 'output_prob:' + '\n' + str(np.round(node.content['output_prob'], 3))
        index_ = node.content['output_prob'].index(np.max(node.content['output_prob']))
        base_color = classification_colors[index_] 
        intense = get_color_intense(node.content['output_prob'])
        if intense == -1:
            box.set_facecolor('white')
        else:
            box.set_facecolor(color_gradient(base_color, 'white', min_alpha + (max_alpha-min_alpha)*intense))
        artists[i].set_text(text)
        
def tree_classifier_visual(tree, artists, feature_names, class_names, show_content, max_alpha=1.0, min_alpha=0.0):
    preorder_nodes = []
    preorder_dfs(tree.root, preorder_nodes)
    assert len(preorder_nodes)==len(artists)
    assert max_alpha >= 0.0 and max_alpha <= 1.0
    assert min_alpha >= 0.0 and min_alpha <= 1.0
    assert max_alpha > min_alpha
    for i in range(len(preorder_nodes)):
        text = ''
        node = preorder_nodes[i]
        box = artists[i].get_bbox_patch()
        index_ = np.argmax(node.content['value'])
        intense = get_color_intense(node.content['value'])
        base_color = classification_colors[index_]
        #box.set_boxstyle('round', rounding_size=1.0)
        if node.content['feature'] >= 0:
            feature_name = feature_names[node.content['feature']]
            if len(feature_name) > 10:
                feature_name = feature_name[:9] + '...'
            text += feature_name + '<=' + str(np.round(node.content['threshold'], 3)) + '\n' 
        if node.content['is_leave']:
            #index_ = node.content['value'].index(np.max(node.content['value']))
            if 'gain' in show_content:
                text += 'gain = ' + str(np.format_float_scientific(node.content['impurity'],precision=3)) + '\n'
            if 'value' in show_content:
                text += 'value = ' + '\n' + str(list(np.round(node.content['value'], 2))) + '\n' 
            box.set_linestyle('dashed')
            #box.set_facecolor('white')
        else:
            if 'gain' in show_content:
                text += 'gain = ' + str(np.format_float_scientific(node.content['impurity'],precision=3)) + '\n'
            if 'value' in show_content:
                text += 'value = ' + '\n' + str(list(np.round(node.content['value'], 2)))
            #box.set_facecolor('white')
        box.set_facecolor(color_gradient(base_color, 'white', min_alpha + (max_alpha-min_alpha)*intense))
        artists[i].set_text(text)
        
# 打印树
def print_tree(tree, feature_names, class_names, is_classifier=True, title=None, show_content=['gain','value','impurity'],
                max_alpha=0.8, min_alpha=0.2, title_size=20, title_loc=(2.0, -0.01), fontsize=16, figsize=(16,16), dpi=300):
    tree_ = skljson.from_dict(to_skl_tree(tree, is_classifier=is_classifier))
    fig,ax = plt.subplots(ncols=1, figsize=figsize, dpi=dpi)
    artists = sklearn.tree.plot_tree(tree_, 
                       feature_names=feature_names,  
                       class_names=class_names,
                       filled=True,
                       node_ids=True,
                       impurity=True,
                       rounded=True,
                       fontsize=fontsize,
                       ax=ax)
    ax.set_title(title, fontsize=title_size, x=title_loc[0], y=title_loc[1])
    #ax.set_xlabel(second_title, fontsize=titlesize-1, loc='center')
    if is_classifier:
        tree_classifier_visual(tree, artists, feature_names, class_names, show_content, max_alpha=max_alpha, min_alpha=min_alpha)
        tree_legend(ax, tree, (fontsize * 1.5), max_alpha, dpi*0.5)
    else:
        tree_regressor_visual(tree, artists, feature_names, class_names, show_content, max_alpha=max_alpha, min_alpha=min_alpha)
    plt.tight_layout()
    plt.show()
