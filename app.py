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

    html.Div(id='info-output', style={'margin': '20px', 'textAlign': 'left'}),
    dcc.Graph(id='plot-output', style={'margin': 'auto', 'width': '90%', 'maxWidth': '1200px'}),

    html.Div(id='slider-output', style={'margin': '20px 10%', 'maxWidth': '1200px'}),

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

# Main callback - handles file upload and creates slider
@app.callback(
    [Output('plot-output', 'figure'),
     Output('info-output', 'children'),
     Output('slider-output', 'children'),
     Output('controls', 'style')],
    [Input('upload-data', 'contents'),
     Input('target-point-input', 'value')],
    [State('upload-data', 'filename')]
)
def update_visualization(contents, target_point_id, filename):
    global param, lb, ub, positions

    if contents is None:
        return {}, "", html.Div(), {'display': 'none'}

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

        # No filtering initially - show all particles
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

        slider_div = html.Div([
            html.Label("Filter by Parameter 0:", style={'marginBottom': '5px'}),
            dcc.RangeSlider(
                id="range-slider-0",
                min=lb[0],
                max=ub[0],
                step=(ub[0] - lb[0]) / 100 if ub[0] > lb[0] else 0.01,
                value=[lb[0], ub[0]],
                tooltip={"placement": "bottom", "always_visible": False}
            )
        ], style={'marginBottom': '20px'})

        return fig, info_text, slider_div, {'margin': '20px', 'display': 'block'}

    except Exception as e:
        error_msg = html.Div([
            html.P(f"Error processing file: {str(e)}", style={'color': 'red'}),
            html.P("Please ensure you've uploaded a valid PSO pickle file.")
        ])
        return {}, error_msg, html.Div(), {'display': 'none'}

# Separate callback - handles slider changes only
@app.callback(
    Output('plot-output', 'figure'),
    [Input('range-slider-0', 'value')],
    [State('upload-data', 'contents'),
     State('target-point-input', 'value'),
     State('upload-data', 'filename')]
)
def update_plot_with_slider(param0_range, contents, target_point_id, filename):
    global param, lb, ub, positions

    if contents is None:
        return {}

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        pso_object = pickle.load(io.BytesIO(decoded))

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

        return fig

    except Exception as e:
        return {}


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)