import base64
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, State, ALL, ctx

app = Dash(__name__)
app.title = "CSV PSO Visualizer"

# Global storage for PSO data
pso_data = {
    'parameters': None,
    'objectives': None,
    'pareto_objectives': None,
    'pareto_positions': None,
    'param_names': [],
    'obj_names': []
}

# --- Helper functions ---

def filter_pareto_front(points):
    """
    Filter Pareto front from a 2D numpy array of objective values.
    Assumes minimization.
    """
    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i, c in enumerate(points):
        if is_pareto[i]:
            is_dominated = np.all(points[is_pareto] <= c, axis=1) & np.any(points[is_pareto] < c, axis=1)
            is_pareto[is_pareto] = ~is_dominated
            is_pareto[i] = True 
    return is_pareto

def create_param_slider_component(index, name, min_val, max_val):
    marks = {min_val: f"{min_val:.2f}", max_val: f"{max_val:.2f}"}
    step = (max_val - min_val) / 100 if max_val > min_val else 0.01
    return html.Div([
        html.P(f"Filter by {name}:", style={'marginBottom': '5px', 'marginTop': '15px'}),
        dcc.RangeSlider(
            id={'type': 'param-slider', 'index': index},
            min=min_val,
            max=max_val,
            step=step,
            value=[min_val, max_val],
            marks=marks,
            tooltip={"placement": "bottom", "always_visible": False},
            updatemode='drag'
        )
    ])

def create_objective_slider_component(index, name, min_val, max_val):
    marks = {min_val: f"{min_val:.2f}", max_val: f"{max_val:.2f}"}
    step = (max_val - min_val) / 100 if max_val > min_val else 0.01
    return html.Div([
        html.P(f"Filter by {name}:", style={'marginBottom': '5px', 'marginTop': '15px'}),
        dcc.RangeSlider(
            id={'type': 'objective-slider', 'index': index},
            min=min_val,
            max=max_val,
            step=step,
            value=[min_val, max_val],
            marks=marks,
            tooltip={"placement": "bottom", "always_visible": False},
            updatemode='drag'
        )
    ])

def create_scatter_matrix(objectives, pareto_objectives, param_names, obj_names):
    num_objectives = objectives.shape[1]
    
    subplot_titles = []
    for i in range(num_objectives):
        for j in range(num_objectives):
            if i == j:
                subplot_titles.append(obj_names[i])
            else:
                subplot_titles.append(f'{obj_names[j]} vs {obj_names[i]}')
    
    fig = make_subplots(
        rows=num_objectives, cols=num_objectives,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08, horizontal_spacing=0.08
    )
    
    for i in range(num_objectives):
        for j in range(num_objectives):
            row, col = i + 1, j + 1
            
            if i == j:
                # Diagonal: just label, no points
                fig.add_trace(
                    go.Scatter(x=[0.5], y=[0.5], mode='text', text=[obj_names[i]],
                               textfont=dict(size=16), showlegend=False, hoverinfo='none'),
                    row=row, col=col
                )
                fig.update_xaxes(range=[0, 1], showticklabels=False, row=row, col=col)
                fig.update_yaxes(range=[0, 1], showticklabels=False, row=row, col=col)
            else:
                # All points gray
                fig.add_trace(
                    go.Scatter(x=objectives[:, j], y=objectives[:, i], mode='markers',
                               marker=dict(size=4, color='grey', opacity=0.5),
                               name='All Points', showlegend=(i == 0 and j == 1)),
                    row=row, col=col
                )
                # Pareto front blue
                fig.add_trace(
                    go.Scatter(x=pareto_objectives[:, j], y=pareto_objectives[:, i], mode='markers',
                               marker=dict(size=6, color='blue', opacity=0.7),
                               name='Pareto Front', showlegend=(i == 0 and j == 1)),
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

# --- Layout ---

app.layout = html.Div([
    html.H1("CSV-Based PSO Visualization", style={'textAlign': 'center'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select a CSV File')]),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '20px auto'
        },
        multiple=False,
        accept='.csv'
    ),
    dcc.Graph(id='pareto-graph', style={'margin': 'auto', 'width': '90%', 'height': 'auto'}),
    html.Div(id='param-slider-container', style={'margin': '20px 10%', 'maxWidth': '1000px'}),
    html.Div(id='objective-slider-container', style={'margin': '20px 10%', 'maxWidth': '1000px'}),
])

# --- Callbacks ---

@app.callback(
    [Output('param-slider-container', 'children'),
     Output('objective-slider-container', 'children'),
     Output('pareto-graph', 'figure')],
    Input('upload-data', 'contents')
)
def process_file(contents):
    if contents is None:
        return [], [], go.Figure()

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    param_data = df.iloc[:, :-3].values
    obj_data = df.iloc[:, -3:].values

    param_names = df.columns[:-3].tolist()
    obj_names = df.columns[-3:].tolist()

    # Store globally
    pso_data['parameters'] = param_data
    pso_data['objectives'] = obj_data
    pso_data['param_names'] = param_names
    pso_data['obj_names'] = obj_names

    pareto_mask = filter_pareto_front(obj_data)
    pso_data['pareto_objectives'] = obj_data[pareto_mask]
    pso_data['pareto_positions'] = param_data[pareto_mask]

    # Create sliders
    param_sliders = [html.H3("Parameter Filters")] + [
        create_param_slider_component(i, name, float(np.min(param_data[:, i])), float(np.max(param_data[:, i])))
        for i, name in enumerate(param_names)
    ]
    objective_sliders = [html.H3("Objective Filters")] + [
        create_objective_slider_component(i, name, float(np.min(obj_data[:, i])), float(np.max(obj_data[:, i])))
        for i, name in enumerate(obj_names)
    ]

    # Initial figure with all Pareto front points
    fig = create_scatter_matrix(pso_data['objectives'], pso_data['pareto_objectives'], param_names, obj_names)

    return param_sliders, objective_sliders, fig


@app.callback(
    Output('pareto-graph', 'figure'),
    [Input({'type': 'param-slider', 'index': ALL}, 'value'),
     Input({'type': 'objective-slider', 'index': ALL}, 'value')]
)
def update_filtered_plot(param_slider_values, objective_slider_values):
    if pso_data['pareto_objectives'] is None:
        return go.Figure()

    positions = pso_data['pareto_positions']
    objectives = pso_data['pareto_objectives']
    param_names = pso_data['param_names']
    obj_names = pso_data['obj_names']

    mask = np.ones(len(objectives), dtype=bool)

    # Filter by params
    for i, slider_range in enumerate(param_slider_values):
        if slider_range and len(slider_range) == 2:
            low, high = slider_range
            mask &= (positions[:, i] >= low) & (positions[:, i] <= high)

    # Filter by objectives
    for i, slider_range in enumerate(objective_slider_values):
        if slider_range and len(slider_range) == 2:
            low, high = slider_range
            mask &= (objectives[:, i] >= low) & (objectives[:, i] <= high)

    filtered_objectives = objectives[mask]
    filtered_positions = positions[mask]

    if len(filtered_objectives) == 0:
        return go.Figure()

    fig = create_scatter_matrix(filtered_objectives, filtered_objectives, param_names, obj_names)
    return fig


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
