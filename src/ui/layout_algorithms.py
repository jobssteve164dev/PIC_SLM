import networkx as nx


class LayoutAlgorithms:
    """图形布局算法集合"""
    
    @staticmethod
    def custom_tree_layout(G, root=None):
        """自定义树形布局算法，不依赖pygraphviz"""
        if root is None:
            # 尝试找到根节点（入度为0的节点）
            roots = [n for n in G.nodes() if G.in_degree(n) == 0]
            if not roots:
                # 如果没有入度为0的节点，则选择第一个节点作为根
                root = list(G.nodes())[0]
            else:
                root = roots[0]
                
        pos = {}  # 节点位置字典
        visited = set([root])
        current_level = [root]
        level_count = 0
        nodes_per_level = {}  # 记录每层的节点数
        
        # 使用BFS遍历图以分配节点到层次
        while current_level:
            next_level = []
            for node in current_level:
                # 检查是否有子节点
                children = [n for n in G.successors(node) if n not in visited]
                for child in children:
                    visited.add(child)
                    next_level.append(child)
            
            # 更新每层的节点数
            nodes_per_level[level_count] = len(current_level)
            level_count += 1
            current_level = next_level
            
        # 计算总层数
        total_levels = level_count
        
        # 重置访问状态
        visited = set([root])
        current_level = [root]
        level_count = 0
        
        # 再次使用BFS遍历并分配坐标
        while current_level:
            width = 1.0
            y_coord = 1.0 - (level_count / (total_levels if total_levels > 0 else 1))
            
            for i, node in enumerate(current_level):
                # 计算x坐标
                if nodes_per_level[level_count] > 1:
                    x_coord = (i / (nodes_per_level[level_count] - 1)) * width
                else:
                    x_coord = 0.5 * width
                
                # 分配节点位置
                pos[node] = (x_coord, y_coord)
                
            next_level = []
            for node in current_level:
                # 检查是否有子节点
                children = [n for n in G.successors(node) if n not in visited]
                for child in children:
                    visited.add(child)
                    next_level.append(child)
            
            level_count += 1
            current_level = next_level
            
        # 检查是否有未访问的节点（未连接到主树）
        unvisited = set(G.nodes()) - visited
        if unvisited:
            # 将未访问的节点放在底部
            y_coord = -0.1
            for i, node in enumerate(unvisited):
                x_coord = (i / (len(unvisited) if len(unvisited) > 1 else 1)) * width
                pos[node] = (x_coord, y_coord)
                
        return pos
    
    @staticmethod
    def get_layout_by_index(subgraph, layout_idx):
        """根据索引获取对应的布局"""
        if layout_idx == 0:  # 分层布局
            return nx.multipartite_layout(subgraph, subset_key='depth')
        elif layout_idx == 1:  # 树形布局
            return LayoutAlgorithms.custom_tree_layout(subgraph)
        elif layout_idx == 2:  # 放射布局
            return nx.kamada_kawai_layout(subgraph)
        elif layout_idx == 3:  # 圆形布局
            return nx.circular_layout(subgraph)
        else:  # 随机布局
            return nx.spring_layout(subgraph, k=0.5, iterations=50)
    
    @staticmethod
    def get_layout_names():
        """获取所有可用的布局名称"""
        return ["分层布局", "树形布局", "放射布局", "圆形布局", "随机布局"] 