import base64
import io
import dill as pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, State

app = Dash(__name__)
app.title = "PSO Visualization"

# Global variables to store PSO data
pso_data = {
    'objectives': None,
    'pareto_objectives': None,
    'positions': None,
    'lb': None,
    'ub': None,
    'filename': None
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
    
    html.Div([
        html.P("Filter by Parameter 0:"),
        dcc.RangeSlider(
            id='param-slider',
            min=0, max=10, step=0.01,
            value=[0, 10],
            marks={0: '0', 10: '10'},
            tooltip={"placement": "bottom", "always_visible": False},
            updatemode='drag'
        ),
    ], id='slider-container', style={'margin': '20px', 'display': 'none'})
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

# Callback to load and process file
@app.callback(
    [Output('file-info', 'children'),
     Output('slider-container', 'style'),
     Output('target-container', 'style'),
     Output('param-slider', 'min'),
     Output('param-slider', 'max'),
     Output('param-slider', 'value'),
     Output('param-slider', 'marks')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def process_file(contents, filename):
    global pso_data
    
    if contents is None:
        return "", {'display': 'none'}, {'display': 'none'}, 0, 10, [0, 10], {0: '0', 10: '10'}
    
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
        
        # Set up slider
        param_min = float(pso_data['lb'][0])
        param_max = float(pso_data['ub'][0])
        slider_marks = {param_min: f'{param_min:.1f}', param_max: f'{param_max:.1f}'}
        
        info_text = html.Div([
            html.P(f"File: {filename}"),
            html.P(f"Particles: {len(pso_data['objectives'])}"),
            html.P(f"Objectives: {pso_data['objectives'].shape[1]}"),
            html.P(f"Pareto front: {len(pso_data['pareto_objectives'])}")
        ])
        
        return (info_text,
                {'margin': '20px', 'display': 'block'},  # Show slider
                {'margin': '20px', 'textAlign': 'center', 'display': 'block'},  # Show target input
                param_min, param_max, [param_min, param_max], slider_marks)
                
    except Exception as e:
        error_msg = html.P(f"Error: {str(e)}", style={'color': 'red'})
        return error_msg, {'display': 'none'}, {'display': 'none'}, 0, 10, [0, 10], {0: '0', 10: '10'}

# Simple callback to update plot
@app.callback(
    Output('main-plot', 'figure'),
    [Input('param-slider', 'value'),
     Input('target-input', 'value')],
    prevent_initial_call=True
)
def update_plot(slider_range, target_id):
    global pso_data
    
    # Check if data is loaded
    if pso_data['objectives'] is None:
        return {}
    
    # Filter data based on slider
    if slider_range and len(slider_range) == 2:
        low, high = slider_range
        mask = (pso_data['positions'][:, 0] >= low) & (pso_data['positions'][:, 0] <= high)
        filtered_objectives = pso_data['objectives'][mask]
    else:
        filtered_objectives = pso_data['objectives']
    
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