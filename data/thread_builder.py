# data/thread_builder.py

import pandas as pd
import networkx as nx
import logging
from typing import Dict
import re

logger = logging.getLogger(__name__)

class ThreadBuilder:
    """Build Reddit comment thread trees from parent-child relationships."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize ThreadBuilder with comment data.
        
        Args:
            df: DataFrame with 'id', 'parent_id', 'link_id' columns
        """
        self.df = df.copy()
        
        self.df['parent_id_clean'] = self.df['parent_id'].apply(self._clean_reddit_id)
        
        self.graphs: Dict[str, nx.DiGraph] = {}
        logger.info(f"Initialized ThreadBuilder with {len(df)} comments")
    
    @staticmethod
    def _clean_reddit_id(reddit_id: str) -> str:
        """
        Remove Reddit type prefix from ID.
        
        Reddit uses prefixes like:
        - t1_ for comments
        - t3_ for submissions
        
        Args:
            reddit_id: Reddit ID with prefix (e.g., 't1_abc123')
            
        Returns:
            Clean ID without prefix (e.g., 'abc123')
        """
        if pd.isna(reddit_id):
            return reddit_id
        
        # Remove t1_, t3_, etc. prefix
        return re.sub(r'^t\d+_', '', str(reddit_id))
    
    def build_thread_graphs(self) -> Dict[str, nx.DiGraph]:
        """
        Build directed graph for each thread (link_id).
        
        Returns:
            Dictionary mapping link_id to NetworkX DiGraph
        """
        logger.info("Building thread graphs...")
        
        grouped = self.df.groupby('link_id')
        
        for link_id, group in grouped:
            G = nx.DiGraph()
            
            # Add all comments as nodes
            for idx, row in group.iterrows():
                G.add_node(row['id'], **row.to_dict())
            
            # Add edges (parent -> child) using cleaned parent_id
            for idx, row in group.iterrows():
                parent_clean = row['parent_id_clean']
                if pd.notna(parent_clean) and parent_clean in G.nodes:
                    G.add_edge(parent_clean, row['id'])
            
            self.graphs[link_id] = G
        
        logger.info(f"Built {len(self.graphs)} thread graphs")
        return self.graphs
    
    def calculate_depths(self) -> pd.DataFrame:
        """
        Calculate depth of each comment in its thread.
        
        Returns:
            DataFrame with added 'depth' column
        """
        logger.info("Calculating comment depths...")
        
        if not self.graphs:
            self.build_thread_graphs()
        
        depth_list = []
        
        for link_id, G in self.graphs.items():
            # Find root nodes (comments whose parent is not in the dataset)
            root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
            
            for root in root_nodes:
                # BFS from each root to calculate depths
                depths = nx.single_source_shortest_path_length(G, root)
                for node, depth in depths.items():
                    depth_list.append({
                        'id': node,
                        'link_id': link_id,
                        'depth': depth
                    })
        
        depth_df = pd.DataFrame(depth_list)
        self.df = self.df.merge(depth_df[['id', 'depth']], on='id', how='left')
        
        # Fill NaN depths with 0 (isolated nodes)
        self.df['depth'] = self.df['depth'].fillna(0)
        
        logger.info(f"Depth distribution:\n{self.df['depth'].value_counts().sort_index()}")
        return self.df
    
    def filter_deep_threads(self, min_depth: int = 3) -> pd.DataFrame:
        """
        Filter threads with depth >= min_depth.
        
        Args:
            min_depth: Minimum thread depth to retain
            
        Returns:
            Filtered DataFrame
        """
        if 'depth' not in self.df.columns:
            self.calculate_depths()
        
        thread_max_depths = self.df.groupby('link_id')['depth'].max()
        deep_threads = thread_max_depths[thread_max_depths >= min_depth].index
        
        filtered_df = self.df[self.df['link_id'].isin(deep_threads)].copy()
        
        logger.info(f"Filtered to {len(deep_threads)} threads with depth >= {min_depth}")
        logger.info(f"Total comments: {len(filtered_df)}")
        
        return filtered_df
    
    def get_parent_child_pairs(self) -> pd.DataFrame:
        """
        Extract all parent-child comment pairs.
        
        Returns:
            DataFrame with parent and child information
        """
        logger.info("Extracting parent-child pairs...")
        
        # Use cleaned parent_id for matching
        children = self.df[self.df['parent_id_clean'].notna()].copy()
        
        # Get emotion columns
        emotion_cols = [col for col in self.df.columns 
                       if col not in ['id', 'text', 'text_clean', 'author', 
                                     'subreddit', 'link_id', 'parent_id', 
                                     'parent_id_clean', 'created_utc', 'rater_id', 
                                     'example_very_unclear', 'depth']]
        
        # Merge with parent using cleaned parent_id
        merge_cols = ['id', 'text']
        if 'text_clean' in self.df.columns:
            merge_cols.append('text_clean')
        merge_cols.extend(emotion_cols)
        
        pairs = children.merge(
            self.df[merge_cols],
            left_on='parent_id_clean',
            right_on='id',
            suffixes=('_child', '_parent'),
            how='inner'
        )
        
        logger.info(f"Extracted {len(pairs)} parent-child pairs")
        return pairs
    
    def get_thread_statistics(self) -> pd.DataFrame:
        """
        Get statistics for each thread.
        
        Returns:
            DataFrame with thread-level statistics
        """
        if not self.graphs:
            self.build_thread_graphs()
        
        stats = []
        for link_id, G in self.graphs.items():
            root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
            max_depth = 0
            
            for root in root_nodes:
                depths = nx.single_source_shortest_path_length(G, root)
                max_depth = max(max_depth, max(depths.values()) if depths else 0)
            
            stats.append({
                'link_id': link_id,
                'num_comments': G.number_of_nodes(),
                'num_root_comments': len(root_nodes),
                'max_depth': max_depth,
                'num_edges': G.number_of_edges()
            })
        
        return pd.DataFrame(stats)