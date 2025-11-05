"""图谱存储 - NetworkX图存储"""

import os
import logging
import networkx as nx
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class GraphStorage:
    """图谱存储类
    
    使用NetworkX存储和管理知识图谱。
    """

    def __init__(self, graph_storage_dir: str = "storage/kg_graphs"):
        """初始化图谱存储
        
        Args:
            graph_storage_dir: 图谱存储目录
        """
        self.graph_storage_dir = graph_storage_dir
        os.makedirs(self.graph_storage_dir, exist_ok=True)
        self.graph_file_path = os.path.join(
            self.graph_storage_dir, "knowledge_graph.graphml"
        )
        self.knowledge_graph = self._load_graph()

    def _load_graph(self) -> nx.Graph:
        """加载NetworkX图，如果文件不存在则创建新图"""
        if os.path.exists(self.graph_file_path):
            try:
                graph = nx.read_graphml(self.graph_file_path)
                logger.info(
                    f"已加载知识图谱: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边"
                )
                return graph
            except Exception as e:
                logger.error(f"加载图谱文件失败: {str(e)}, 创建新图")
                return nx.Graph()
        else:
            logger.info("创建新的知识图谱")
            return nx.Graph()

    def save(self):
        """保存NetworkX图到文件"""
        try:
            nx.write_graphml(
                self.knowledge_graph, self.graph_file_path, encoding="utf-8"
            )
            logger.info(
                f"图谱已保存: {self.knowledge_graph.number_of_nodes()} 个节点, {self.knowledge_graph.number_of_edges()} 条边"
            )
        except Exception as e:
            logger.error(f"保存图谱失败: {str(e)}")

    def add_entity(self, entity_data: Dict):
        """向NetworkX图中添加实体节点"""
        entity_id = entity_data["entity_id"]

        # 准备节点属性
        node_attrs = {
            "entity_name": entity_data["entity_name"],
            "entity_type": entity_data["entity_type"],
            "description": entity_data["description"],
            "document_id": str(entity_data["document_id"]),
            "created_at": self._get_current_timestamp(),
        }

        # 添加或更新节点
        self.knowledge_graph.add_node(entity_id, **node_attrs)
        logger.debug(f"向图谱添加实体节点: {entity_id}")

    def add_relation(self, relation_data: Dict):
        """向NetworkX图中添加关系边"""
        source_entity = relation_data["source_entity"]
        target_entity = relation_data["target_entity"]

        # 准备边属性
        edge_attrs = {
            "relation_type": relation_data["relation_type"],
            "description": relation_data["description"],
            "keywords": relation_data["keywords"],
            "weight": relation_data["weight"],
            "document_id": str(relation_data["document_id"]),
            "created_at": self._get_current_timestamp(),
        }

        # 添加边
        self.knowledge_graph.add_edge(source_entity, target_entity, **edge_attrs)
        logger.debug(f"向图谱添加关系: {source_entity} -> {target_entity}")

    def get_graph(self) -> nx.Graph:
        """获取知识图谱"""
        return self.knowledge_graph

    def get_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        num_nodes = self.knowledge_graph.number_of_nodes()
        if num_nodes > 0:
            avg_degree = sum(dict(self.knowledge_graph.degree()).values()) / num_nodes
        else:
            avg_degree = 0.0

        return {
            "total_nodes": num_nodes,
            "total_edges": self.knowledge_graph.number_of_edges(),
            "avg_degree": avg_degree,
            "graph_file": self.graph_file_path,
        }

    def search_entities_by_name(
        self, entity_name: str, fuzzy: bool = True
    ) -> List[Dict]:
        """根据实体名称搜索实体"""
        results = []

        for node_id, node_data in self.knowledge_graph.nodes(data=True):
            if fuzzy:
                # 模糊匹配
                if entity_name.lower() in node_data.get("entity_name", "").lower():
                    result = node_data.copy()
                    result["entity_id"] = node_id
                    results.append(result)
            else:
                # 精确匹配
                if entity_name == node_data.get("entity_name", ""):
                    result = node_data.copy()
                    result["entity_id"] = node_id
                    results.append(result)

        return results

    def get_entity_relations(
        self, entity_id: str, max_depth: int = 2
    ) -> Dict[str, Any]:
        """获取实体的关系网络"""
        if entity_id not in self.knowledge_graph:
            return {"entity_id": entity_id, "found": False}

        # 使用BFS获取指定深度的子图
        subgraph_nodes = set([entity_id])
        current_level = set([entity_id])

        for depth in range(max_depth):
            next_level = set()
            for node in current_level:
                neighbors = set(self.knowledge_graph.neighbors(node))
                next_level.update(neighbors)
                subgraph_nodes.update(neighbors)
            current_level = next_level - subgraph_nodes
            if not current_level:  # 没有新的邻居节点
                break

        # 构建子图
        subgraph = self.knowledge_graph.subgraph(subgraph_nodes)

        # 构建返回结果
        nodes = []
        edges = []

        for node_id, node_data in subgraph.nodes(data=True):
            node_info = node_data.copy()
            node_info["entity_id"] = node_id
            nodes.append(node_info)

        for source, target, edge_data in subgraph.edges(data=True):
            edge_info = edge_data.copy()
            edge_info["source_entity"] = source
            edge_info["target_entity"] = target
            edges.append(edge_info)

        return {
            "entity_id": entity_id,
            "found": True,
            "max_depth": max_depth,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "nodes": nodes,
            "edges": edges,
        }

    def _get_current_timestamp(self) -> str:
        """获取当前时间的TIMESTAMP格式字符串"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

