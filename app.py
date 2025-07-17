# wanna try this too.

import base64
import io
import dill as pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, State

app = Dash(__name__)
app.title = "PSO Visualization"

param = []
lb = []
ub = []
positions = []

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

    html.Div(id='info-output', style={'margin': '20px', 'textAlign': 'center'}),
    dcc.Graph(id='plot-output', style={'margin': 'auto', 'width': '90%', 'maxWidth': '1200px'}),

    # Placeholder slider - always exists but hidden initially
    html.Div([
        html.Label("Filter by Parameter 0:", style={'marginBottom': '5px'}),
        dcc.RangeSlider(
            id="range-slider-0",
            min=0,
            max=1,
            step=0.01,
            value=[0, 1],
            tooltip={"placement": "bottom", "always_visible": False}
        )
    ], id='slider-output', style={'margin': '20px 10%', 'maxWidth': '1200px', 'display': 'none'}),

    html.Div([
        html.Label("Target Point ID:", style={'fontWeight': 'bold'}),
        dcc.Input(
            id='target-point-input',
            type='number',
            value=6,
            min=0,
            style={'margin': '10px', 'padding': '5px', 'width': '60px'}
        )
    ], id='controls', style={'margin': '20px auto', 'display': 'none', 'textAlign': 'center', 'maxWidth': '1200px'})
])

def create_scatter_matrix(objectives, pareto_objectives, target_point_id=6):
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
        rows=num_objectives,
        cols=num_objectives,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )

    for i in range(num_objectives):
        for j in range(num_objectives):
            row = i + 1
            col = j + 1

            if i == j:
                fig.add_trace(
                    go.Scatter(x=[0.5], y=[0.5], mode='text', text=[obj_names[i]],
                               textfont=dict(size=16), showlegend=False, hoverinfo='none'),
                    row=row, col=col
                )
                fig.update_xaxes(range=[0, 1], showticklabels=False, row=row, col=col)
                fig.update_yaxes(range=[0, 1], showticklabels=False, row=row, col=col)
            else:
                fig.add_trace(
                    go.Scatter(x=objectives[:, j], y=objectives[:, i], mode='markers',
                               marker=dict(size=4, color='grey', opacity=0.5),
                               name='All Points', showlegend=(i == 0 and j == 1)),
                    row=row, col=col
                )
                fig.add_trace(
                    go.Scatter(x=pareto_objectives[:, j], y=pareto_objectives[:, i], mode='markers',
                               marker=dict(size=6, color='blue', opacity=0.7),
                               name='Pareto Front', showlegend=(i == 0 and j == 1)),
                    row=row, col=col
                )
                fig.add_trace(
                    go.Scatter(x=[target_point[j]], y=[target_point[i]], mode='markers',
                               marker=dict(size=10, color='red', symbol='star',
                                           line=dict(width=2, color='darkred')),
                               name=f'Point {target_point_id}', showlegend=(i == 0 and j == 1)),
                    row=row, col=col
                )
                fig.update_xaxes(title_text=obj_names[j], row=row, col=col)
                fig.update_yaxes(title_text=obj_names[i], row=row, col=col)

    fig.update_layout(
        title=f'{num_objectives}x{num_objectives} Interactive Scatter Plot Matrix',
        height=num_objectives * 300,
        width=num_objectives * 300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# Single callback - handles all updates
@app.callback(
    [Output('plot-output', 'figure'),
     Output('info-output', 'children'),
     Output('slider-output', 'style'),
     Output('controls', 'style'),
     Output('range-slider-0', 'min'),
     Output('range-slider-0', 'max'),
     Output('range-slider-0', 'step'),
     Output('range-slider-0', 'value')],
    [Input('upload-data', 'contents'),
     Input('target-point-input', 'value'),
     Input('range-slider-0', 'value')],
    [State('upload-data', 'filename')]
)
def update_visualization(contents, target_point_id, param0_range, filename):
    global param, lb, ub, positions

    # Default values when no file is uploaded
    if contents is None:
        return {}, "", {'display': 'none'}, {'display': 'none'}, 0, 1, 0.01, [0, 1]

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        pso_object = pickle.load(io.BytesIO(decoded))

        param = pso_object.pareto_front[0].position
        lb = pso_object.lower_bounds
        ub = pso_object.upper_bounds

        # Get all particle positions (NxD)
        positions = np.array([p.position for p in pso_object.particles])

        objectives = np.array([p.fitness for p in pso_object.particles])
        pareto_objectives = np.array([p.fitness for p in pso_object.pareto_front])

        if target_point_id is None or target_point_id < 0 or target_point_id >= len(objectives):
            target_point_id = 0

        # Filter particles by parameter 0 range slider
        if param0_range is not None and len(param0_range) == 2:
            mask = (positions[:, 0] >= param0_range[0]) & (positions[:, 0] <= param0_range[1])
            filtered_objectives = objectives[mask]
            filtered_pareto_objectives = pareto_objectives 
        else:
            filtered_objectives = objectives
            filtered_pareto_objectives = pareto_objectives

        fig = create_scatter_matrix(filtered_objectives, filtered_pareto_objectives, target_point_id)

        info_text = html.Div([
            html.P(f"File: {filename}"),
            html.P(f"Number of particles (filtered): {len(filtered_objectives)}"),
            html.P(f"Number of objectives: {objectives.shape[1]}"),
            html.P(f"Pareto front size: {len(pareto_objectives)}"),
            html.P(f"Target point ID: {target_point_id}")
        ])

        # Configure slider with real data
        slider_min = lb[0]
        slider_max = ub[0]
        slider_step = (ub[0] - lb[0]) / 100 if ub[0] > lb[0] else 0.01
        slider_value = [lb[0], ub[0]]

        return (fig, info_text, 
                {'margin': '20px 10%', 'maxWidth': '1200px', 'display': 'block'},  # Show slider
                {'margin': '20px auto', 'display': 'block', 'textAlign': 'center', 'maxWidth': '1200px'},  # Show controls
                slider_min, slider_max, slider_step, slider_value)

    except Exception as e:
        error_msg = html.Div([
            html.P(f"Error processing file: {str(e)}", style={'color': 'red'}),
            html.P("Please ensure you've uploaded a valid PSO pickle file.")
        ])
        return {}, error_msg, {'display': 'none'}, {'display': 'none'}, 0, 1, 0.01, [0, 1]


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)