import base64
import io
import dill as pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, State, ALL

app = Dash(__name__)
app.title = "PSO Visualization"

# Global variables to store PSO data
pso_data = {
    'objectives': None,
    'pareto_objectives': None,
    'positions': None,
    'lb': None,
    'ub': None,
    'filename': None,
    'param_names': None
}

app.layout = html.Div([
    html.H2("Upload Pickle File"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Pickle File')]),
        style={
            'width': '50%', 'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px auto'
        },
        multiple=False,
        accept='.pkl,.pickle'
    ),

    html.Div(id='file-info', style={'margin': '20px', 'textAlign': 'center'}),
    
    html.Div([
        html.Label("Target Point ID:"),
        dcc.Input(id='target-input', type='number', value=0, min=0, style={'margin': '10px'})
    ], id='target-container', style={'margin': '20px', 'textAlign': 'center', 'display': 'none'}),
    
    dcc.Graph(id='main-plot', style={'margin': 'auto', 'width': '90%'}),
    
    html.Div(id='slider-container', style={'margin': '20px', 'display': 'none'})
])

def create_scatter_matrix(objectives, pareto_objectives, target_point_id=0):
    num_objectives = objectives.shape[1]
    
    if target_point_id >= len(objectives):
        target_point_id = 0
        
    target_point = objectives[target_point_id]
    obj_names = [f'Objective {i+1}' for i in range(num_objectives)]
    
    subplot_titles = []
    for i in range(num_objectives):
        for j in range(num_objectives):
            subplot_titles.append(f'{obj_names[j]} vs {obj_names[i]}' if i != j else obj_names[i])
    
    fig = make_subplots(
        rows=num_objectives, cols=num_objectives,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08, horizontal_spacing=0.08
    )
    
    for i in range(num_objectives):
        for j in range(num_objectives):
            row, col = i + 1, j + 1
            
            if i == j:
                fig.add_trace(
                    go.Scatter(x=[0.5], y=[0.5], mode='text', text=[obj_names[i]],
                              textfont=dict(size=16), showlegend=False, hoverinfo='none'),
                    row=row, col=col
                )
                fig.update_xaxes(range=[0, 1], showticklabels=False, row=row, col=col)
                fig.update_yaxes(range=[0, 1], showticklabels=False, row=row, col=col)
            else:
                # All points
                fig.add_trace(
                    go.Scatter(x=objectives[:, j], y=objectives[:, i], mode='markers',
                              marker=dict(size=4, color='grey', opacity=0.5),
                              name='All Points', showlegend=(i == 0 and j == 1)),
                    row=row, col=col
                )
                # Pareto front
                fig.add_trace(
                    go.Scatter(x=pareto_objectives[:, j], y=pareto_objectives[:, i], mode='markers',
                              marker=dict(size=6, color='blue', opacity=0.7),
                              name='Pareto Front', showlegend=(i == 0 and j == 1)),
                    row=row, col=col
                )
                # Target point
                fig.add_trace(
                    go.Scatter(x=[target_point[j]], y=[target_point[i]], mode='markers',
                              marker=dict(size=10, color='red', symbol='star'),
                              name=f'Point {target_point_id}', showlegend=(i == 0 and j == 1)),
                    row=row, col=col
                )
                fig.update_xaxes(title_text=obj_names[j], row=row, col=col)
                fig.update_yaxes(title_text=obj_names[i], row=row, col=col)
    
    fig.update_layout(
        title=f'{num_objectives}×{num_objectives} Scatter Plot Matrix',
        height=num_objectives * 300,
        showlegend=True
    )
    return fig

def create_slider_component(param_index, param_name, param_min, param_max):
    """Create a single slider component for a parameter"""
    slider_marks = {param_min: f'{param_min:.1f}', param_max: f'{param_max:.1f}'}
    
    return html.Div([
        html.P(f"Filter by {param_name}:", style={'marginBottom': '5px', 'marginTop': '15px'}),
        dcc.RangeSlider(
            id={'type': 'param-slider', 'index': param_index},
            min=param_min,
            max=param_max,
            step=(param_max - param_min) / 100 if param_max > param_min else 0.01,
            value=[param_min, param_max],
            marks=slider_marks,
            tooltip={"placement": "bottom", "always_visible": False},
            updatemode='drag'
        )
    ])

@app.callback(
    [Output('file-info', 'children'),
     Output('slider-container', 'children'),
     Output('slider-container', 'style'),
     Output('target-container', 'style')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def process_file(contents, filename):
    global pso_data
    
    if contents is None:
        return "", [], {'display': 'none'}, {'display': 'none'}
    
    try:
        # Load PSO data
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        pso_object = pickle.load(io.BytesIO(decoded))
        
        # Store data globally
        pso_data['objectives'] = np.array([p.fitness for p in pso_object.particles])
        pso_data['pareto_objectives'] = np.array([p.fitness for p in pso_object.pareto_front])
        pso_data['positions'] = np.array([p.position for p in pso_object.particles])
        pso_data['lb'] = pso_object.lower_bounds
        pso_data['ub'] = pso_object.upper_bounds
        pso_data['filename'] = filename
        
        # Extract parameter names
        num_params = len(pso_object.lower_bounds)
        param_names = getattr(pso_object, 'param_names', None)
        if param_names is None or len(param_names) != num_params:
            param_names = [f'Parameter {i}' for i in range(num_params)]
        
        pso_data['param_names'] = param_names
        
        # Create info display
        info_text = html.Div([
            html.P(f"File: {filename}"),
            html.P(f"Particles: {len(pso_data['objectives'])}"),
            html.P(f"Objectives: {pso_data['objectives'].shape[1]}"),
            html.P(f"Parameters: {len(pso_data['param_names'])}"),
            html.P(f"Pareto front: {len(pso_data['pareto_objectives'])}")
        ])
        
        # Create sliders dynamically for all parameters
        slider_components = []
        for i, param_name in enumerate(pso_data['param_names']):
            param_min = float(pso_data['lb'][i])
            param_max = float(pso_data['ub'][i])
            slider_components.append(
                create_slider_component(i, param_name, param_min, param_max)
            )
        
        return (info_text,
                slider_components,
                {'margin': '20px', 'display': 'block'}, 
                {'margin': '20px', 'textAlign': 'center', 'display': 'block'}) 
                
    except Exception as e:
        error_msg = html.P(f"Error: {str(e)}", style={'color': 'red'})
        return error_msg, [], {'display': 'none'}, {'display': 'none'}

# Callback to update plot
@app.callback(
    Output('main-plot', 'figure'),
    [Input({'type': 'param-slider', 'index': ALL}, 'value'),
     Input('target-input', 'value')],
    prevent_initial_call=True
)
def update_plot(slider_values, target_id):
    global pso_data
    
    # Check if data is loaded
    if pso_data['objectives'] is None:
        return {}
    
    # Start with all particles
    mask = np.ones(len(pso_data['positions']), dtype=bool)
    
    # Apply filters for all parameters
    for i, slider_range in enumerate(slider_values):
        if slider_range and len(slider_range) == 2 and i < pso_data['positions'].shape[1]:
            low, high = slider_range
            param_mask = (pso_data['positions'][:, i] >= low) & (pso_data['positions'][:, i] <= high)
            mask = mask & param_mask
    
    # Apply combined filter
    filtered_objectives = pso_data['objectives'][mask]
    
    # Handle empty filtered data
    if len(filtered_objectives) == 0:
        return {}
    
    # Validate target ID
    if target_id is None or target_id >= len(filtered_objectives) or target_id < 0:
        target_id = 0
    
    # Create plot
    fig = create_scatter_matrix(filtered_objectives, pso_data['pareto_objectives'], target_id)
    return fig

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)