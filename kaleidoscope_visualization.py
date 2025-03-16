"""
Kaleidoscope AI - Visualization Module
====================================
Provides visualization capabilities for the Kaleidoscope AI system.
"""

import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional

class KaleidoscopeVisualizer:
    def __init__(self):
        self.default_layout = {
            'template': 'plotly_dark',
            'font': dict(family='Roboto, sans-serif'),
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)'
        }

    def create_knowledge_graph(self, graph: nx.Graph) -> go.Figure:
        """Create an interactive visualization of the knowledge graph"""
        pos = nx.spring_layout(graph, k=1, iterations=50)
        
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=10
            ))

        # Add edges
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        # Add nodes
        for node in graph.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                **self.default_layout,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40)
            )
        )

        return fig

    def create_3d_scatter(self, data: np.ndarray, labels: Optional[List[str]] = None) -> go.Figure:
        """Create a 3D scatter plot for data points"""
        trace = go.Scatter3d(
            x=data[:, 0],
            y=data[:, 1],
            z=data[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=data[:, 2],
                colorscale='Viridis',
                opacity=0.8
            ),
            text=labels
        )

        fig = go.Figure(data=[trace])
        fig.update_layout(
            **self.default_layout,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        return fig

    def create_heatmap(self, matrix: np.ndarray, labels: Optional[List[str]] = None) -> go.Figure:
        """Create a heatmap visualization"""
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale='Viridis'
        ))
        
        fig.update_layout(**self.default_layout)
        return fig

    def create_quantum_state_plot(self, state_vector: np.ndarray) -> go.Figure:
        """Create visualization of quantum state amplitudes"""
        n_states = len(state_vector)
        states = [f"|{format(i, f'0{int(np.log2(n_states))}b')}⟩" for i in range(n_states)]
        
        fig = go.Figure(data=[
            go.Bar(
                name='Real',
                x=states,
                y=state_vector.real,
                marker_color='royalblue'
            ),
            go.Bar(
                name='Imaginary',
                x=states,
                y=state_vector.imag,
                marker_color='red'
            )
        ])
        
        fig.update_layout(
            **self.default_layout,
            barmode='group',
            title='Quantum State Visualization',
            xaxis_title='Basis State',
            yaxis_title='Amplitude'
        )
        return fig
