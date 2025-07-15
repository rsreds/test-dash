import optimizer
import base64
import io
import dill as pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, State

app = Dash(__name__)
app.title = "PSO Visualization"

app.layout = html.Div([
    html.H2("Upload Pickle File"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Pickle File')]),
        style={
            'width': '50%', 'height': '60px',
            'lineHeight': '60px', 'borderWidth': '1px',
            'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'margin': '10px'
        },
        multiple=False,
        accept='.pkl,.pickle'
    ),
    html.Div([
        html.Label("Target Point ID:"),
        dcc.Input(
            id='target-point-input',
            type='number',
            value=6,
            min=0,
            style={'margin': '10px'}
        )
    ], id='controls', style={'margin': '20px'}),
    html.Div(id='info-output', style={'margin': '20px'}),
    dcc.Graph(id='plot-output')
])

def extract_data_from_pso(pso_object):
    """Extract objectives data from PSO object"""
    # Extract fitness from all particles
    all_objectives = []
    for particle in pso_object.particles:
        all_objectives.append(particle.fitness)

    # Extract fitness from pareto front
    pareto_objectives_list = []
    for particle in pso_object.pareto_front:
        pareto_objectives_list.append(particle.fitness)

    # Convert to numpy arrays
    objectives = np.array(all_objectives)
    pareto_objectives = np.array(pareto_objectives_list)
    
    return objectives, pareto_objectives

def create_visualization(objectives, pareto_objectives, target_point_id=6):
    """Create the interactive scatter plot matrix"""
    num_objectives = objectives.shape[1]
    
    # Validate target point ID
    if target_point_id >= len(objectives):
        target_point_id = 0
    
    target_point = objectives[target_point_id]

    # Check if target point is on pareto front
    is_pareto = any(np.allclose(target_point, pf_point, rtol=1e-10) for pf_point in pareto_objectives)

    # Create objective names
    obj_names = [f'Objective {i+1}' for i in range(num_objectives)]

    # Create subplot titles
    subplot_titles = []
    for i in range(num_objectives):
        for j in range(num_objectives):
            if i == j:
                subplot_titles.append(obj_names[i])
            else:
                subplot_titles.append(f'{obj_names[j]} vs {obj_names[i]}')

    # Create subplots
    fig = make_subplots(
        rows=num_objectives, 
        cols=num_objectives,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )

    # Add traces
    for i in range(num_objectives):
        for j in range(num_objectives):
            row = i + 1
            col = j + 1
            
            if i == j:
                # Diagonal elements - just show objective name
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
                        marker=dict(size=4, color='grey', opacity=0.5),
                        name='All Points',
                        showlegend=(i == 0 and j == 1),
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
                        showlegend=(i == 0 and j == 1), 
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
                            size=10, 
                            color='red', 
                            symbol='star',
                            line=dict(width=2, color='darkred')
                        ),
                        name=f'Point {target_point_id}',
                        showlegend=(i == 0 and j == 1), 
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

    return fig

@app.callback(
    [Output('plot-output', 'figure'),
     Output('info-output', 'children'),
     Output('controls', 'style')],
    [Input('upload-data', 'contents'),
     Input('target-point-input', 'value')],
    [State('upload-data', 'filename')]
)
def update_visualization(contents, target_point_id, filename):
    if contents is None:
        return {}, "", {'display': 'none'}
    
    try:
        # Decode the uploaded file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Load the pickle file
        pso_object = pickle.load(io.BytesIO(decoded))
        
        # Extract data
        objectives, pareto_objectives = extract_data_from_pso(pso_object)
        
        # Validate target point ID
        if target_point_id is None or target_point_id < 0 or target_point_id >= len(objectives):
            target_point_id = 0
        
        # Create visualization
        fig = create_visualization(objectives, pareto_objectives, target_point_id)
        
        # Create info text
        info_text = html.Div([
            html.P(f"File: {filename}"),
            html.P(f"Number of particles: {len(objectives)}"),
            html.P(f"Number of objectives: {objectives.shape[1]}"),
            html.P(f"Pareto front size: {len(pareto_objectives)}"),
            html.P(f"Target point ID: {target_point_id}")
        ])
        
        return fig, info_text, {'margin': '20px'}
        
    except Exception as e:
        error_msg = html.Div([
            html.P(f"Error processing file: {str(e)}", style={'color': 'red'}),
            html.P("Please ensure you've uploaded a valid PSO pickle file.")
        ])
        return {}, error_msg, {'display': 'none'}

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
