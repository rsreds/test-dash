import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from dash import Dash, html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Multi-Objective Optimization Visualization"

#  data generation
num_objectives = 3
target_point_id = 657
np.random.seed(42)
n_points = 2000

def find_pareto_front(points):
    """Find Pareto optimal points"""
    n_points = points.shape[0]
    pareto_mask = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                    pareto_mask[i] = False
                    break
    return pareto_mask

decision_vars = np.random.uniform(0, 1, (n_points, num_objectives))
objectives = []
for i in range(num_objectives):
    target_point = np.zeros(num_objectives)
    target_point[i] = 1.0  
    obj = np.sum((decision_vars - target_point)**2, axis=1)
    objectives.append(obj)
objectives = np.column_stack(objectives)

# Find Pareto front
pareto_mask = find_pareto_front(objectives)
pareto_objectives = objectives[pareto_mask]

# Find the target point
target_point = objectives[target_point_id]
print(f"Highlighting Point {target_point_id}")
coords_str = ", ".join([f"Obj{i+1}={target_point[i]:.3f}" for i in range(num_objectives)])
print(f"Coordinates: {coords_str}")
is_pareto = pareto_mask[target_point_id]
print(f"On Pareto Front: {'Yes' if is_pareto else 'No'}")

# Convert to interactive Plotly version
obj_names = [f'Objective {i+1}' for i in range(num_objectives)]

# Create subplot matrix
subplot_titles = []
for i in range(num_objectives):
    for j in range(num_objectives):
        if i == j:
            subplot_titles.append(obj_names[i])
        else:
            subplot_titles.append(f'{obj_names[j]} vs {obj_names[i]}')

fig = make_subplots(
    rows=num_objectives, 
    cols=num_objectives,
    subplot_titles=subplot_titles,
    vertical_spacing=0.08,
    horizontal_spacing=0.08
)

# Add traces to each subplot
for i in range(num_objectives):
    for j in range(num_objectives):
        row = i + 1
        col = j + 1
        
        if i == j:  # Diagonal - just add text
            fig.add_trace(
                go.Scatter(
                    x=[0.5], y=[0.5],
                    mode='text',
                    text=[obj_names[i]],
                    textfont=dict(size=16, color='black'),
                    showlegend=False,
                    hoverinfo='none'
                ),
                row=row, col=col
            )
            fig.update_xaxes(range=[0, 1], showticklabels=False, row=row, col=col)
            fig.update_yaxes(range=[0, 1], showticklabels=False, row=row, col=col)
        else:
            # All points
            fig.add_trace(
                go.Scatter(
                    x=objectives[:, j],
                    y=objectives[:, i],
                    mode='markers',
                    marker=dict(size=4, color='lightblue', opacity=0.5),
                    name='All Points',
                    showlegend=(i == 0 and j == 1),  # Show legend only once
                    hovertemplate=f'<b>All Points</b><br>' +
                                f'{obj_names[j]}: %{{x:.3f}}<br>' +
                                f'{obj_names[i]}: %{{y:.3f}}<br>' +
                                '<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Pareto front points
            fig.add_trace(
                go.Scatter(
                    x=pareto_objectives[:, j],
                    y=pareto_objectives[:, i],
                    mode='markers',
                    marker=dict(size=6, color='blue', opacity=0.7),
                    name='Pareto Front',
                    showlegend=(i == 0 and j == 1),  # Show legend only once
                    hovertemplate=f'<b>Pareto Front</b><br>' +
                                f'{obj_names[j]}: %{{x:.3f}}<br>' +
                                f'{obj_names[i]}: %{{y:.3f}}<br>' +
                                '<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Target point
            fig.add_trace(
                go.Scatter(
                    x=[target_point[j]],
                    y=[target_point[i]],
                    mode='markers',
                    marker=dict(
                        size=15, 
                        color='red', 
                        symbol='star',
                        line=dict(width=2, color='darkred')
                    ),
                    name=f'Point {target_point_id}',
                    showlegend=(i == 0 and j == 1),  # Show legend only once
                    hovertemplate=f'<b>Point {target_point_id}</b><br>' +
                                f'{obj_names[j]}: %{{x:.3f}}<br>' +
                                f'{obj_names[i]}: %{{y:.3f}}<br>' +
                                f'On Pareto Front: {"Yes" if is_pareto else "No"}<br>' +
                                '<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Update axis labels
            fig.update_xaxes(title_text=obj_names[j], row=row, col=col)
            fig.update_yaxes(title_text=obj_names[i], row=row, col=col)

# Update layout
fig.update_layout(
    title=f'{num_objectives}x{num_objectives} Interactive Scatter Plot Matrix',
    height=num_objectives * 300,
    width=num_objectives * 300,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Show the interactive plot
fig.show()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)