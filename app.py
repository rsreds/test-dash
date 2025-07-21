import base64
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, State, ALL, callback_context

app = Dash(__name__)
app.title = "CSV PSO Visualizer"

# Global storage for PSO data
pso_data = {
    'parameters': None,
    'objectives': None,
    'pareto_objectives': None,
    'pareto_positions': None,
    'param_names': [],
    'obj_names': [],
    'filename': None,
    'obj_mins': None,  # Store min of objectives for fixed axis
    'obj_maxs': None   # Store max of objectives for fixed axis
}

def create_slider(title, slider_id, min_val, max_val):
    return html.Div([
        html.P(f"Filter Pareto Front by {title}:", style={'marginBottom': '5px', 'marginTop': '15px'}),
        dcc.RangeSlider(
            id=slider_id,
            min=min_val,
            max=max_val,
            step=(max_val - min_val) / 100 if max_val > min_val else 0.01,
            value=[min_val, max_val],
            marks={min_val: f'{min_val:.1f}', max_val: f'{max_val:.1f}'},
            tooltip={"placement": "bottom", "always_visible": False},
            updatemode='drag'
        )
    ])

def filter_pareto_front(points):
    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i, c in enumerate(points):
        if is_pareto[i]:
            is_dominated = np.all(points[is_pareto] <= c, axis=1) & np.any(points[is_pareto] < c, axis=1)
            is_pareto[is_pareto] = ~is_dominated
            is_pareto[i] = True
    return is_pareto

def create_scatter_matrix(full_objectives, pareto_objectives, target_point_id=0, fixed_axis_ranges=None):
    num_obj = full_objectives.shape[1]
    obj_names = pso_data['obj_names']

    if target_point_id >= len(full_objectives):
        target_point_id = 0

    target_point = full_objectives[target_point_id]

    subplot_titles = []
    for i in range(num_obj):
        for j in range(num_obj):
            subplot_titles.append(f'{obj_names[j]} vs {obj_names[i]}' if i != j else obj_names[i])

    fig = make_subplots(
        rows=num_obj, cols=num_obj,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08, horizontal_spacing=0.08
    )

    for i in range(num_obj):
        for j in range(num_obj):
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
                # Plot all points (full dataset, unfiltered) in grey
                fig.add_trace(
                    go.Scatter(x=full_objectives[:, j], y=full_objectives[:, i], mode='markers',
                               marker=dict(size=4, color='grey', opacity=0.5),
                               name='All Points', showlegend=(i == 0 and j == 1)),
                    row=row, col=col
                )
                # Plot filtered Pareto points in blue
                if len(pareto_objectives) > 0:
                    fig.add_trace(
                        go.Scatter(x=pareto_objectives[:, j], y=pareto_objectives[:, i], mode='markers',
                                   marker=dict(size=6, color='blue', opacity=0.7),
                                   name='Pareto Front', showlegend=(i == 0 and j == 1)),
                        row=row, col=col
                    )
                # Plot target point in red
                fig.add_trace(
                    go.Scatter(x=[target_point[j]], y=[target_point[i]], mode='markers',
                               marker=dict(size=10, color='red', symbol='star'),
                               name=f'Point {target_point_id}', showlegend=(i == 0 and j == 1)),
                    row=row, col=col
                )
                fig.update_xaxes(title_text=obj_names[j], row=row, col=col)
                fig.update_yaxes(title_text=obj_names[i], row=row, col=col)

    # Fix axis ranges if provided (on file load)
    if fixed_axis_ranges is not None:
        obj_mins, obj_maxs = fixed_axis_ranges
        for i in range(num_obj):
            for j in range(num_obj):
                fig.update_xaxes(range=[obj_mins[j], obj_maxs[j]], row=i+1, col=j+1)
                fig.update_yaxes(range=[obj_mins[i], obj_maxs[i]], row=i+1, col=j+1)

    fig.update_layout(
        title=f'{num_obj}×{num_obj} Scatter Plot Matrix',
        height=num_obj * 300,
        showlegend=True
    )
    return fig

app.layout = html.Div([
    html.H2("Upload CSV File"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select CSV File')]),
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
        accept='.csv'
    ),

    html.Div(id='file-info', style={'margin': '20px', 'textAlign': 'center'}),

    html.Div([
        html.Label("Target Point ID:"),
        dcc.Input(id='target-input', type='number', value=0, min=0, style={'margin': '10px'})
    ], id='target-container', style={'margin': '20px', 'textAlign': 'center', 'display': 'none'}),

    dcc.Graph(id='main-plot', style={'margin': 'auto', 'width': '90%'}),

    html.Div(id='slider-container', style={'margin': '20px', 'display': 'none'})
])

@app.callback(
    [Output('file-info', 'children'),
     Output('slider-container', 'children'),
     Output('slider-container', 'style'),
     Output('target-container', 'style')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def load_csv(contents, filename):
    if contents is None:
        return '', [], {'display': 'none'}, {'display': 'none'}

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        param_data = df.iloc[:, :-3].values
        obj_data = df.iloc[:, -3:].values

        pso_data['parameters'] = param_data
        pso_data['objectives'] = obj_data
        pso_data['param_names'] = df.columns[:-3].tolist()
        pso_data['obj_names'] = df.columns[-3:].tolist()
        pso_data['filename'] = filename

        # Store global min/max for each objective (for fixed axis ranges)
        pso_data['obj_mins'] = np.min(obj_data, axis=0)
        pso_data['obj_maxs'] = np.max(obj_data, axis=0)

        pareto_mask = filter_pareto_front(obj_data)
        pso_data['pareto_objectives'] = obj_data[pareto_mask]
        pso_data['pareto_positions'] = param_data[pareto_mask]

        sliders = []

        sliders.append(html.H4("Parameter Filters:", style={'marginTop': '20px'}))
        for i, name in enumerate(pso_data['param_names']):
            min_val = float(np.min(param_data[:, i]))
            max_val = float(np.max(param_data[:, i]))
            sliders.append(create_slider(name, {'type': 'param-slider', 'index': i}, min_val, max_val))

        sliders.append(html.H4("Objective Filters:", style={'marginTop': '30px'}))
        for i, name in enumerate(pso_data['obj_names']):
            min_val = float(np.min(pso_data['pareto_objectives'][:, i]))
            max_val = float(np.max(pso_data['pareto_objectives'][:, i]))
            sliders.append(create_slider(name, {'type': 'obj-slider', 'index': i}, min_val, max_val))

        file_info = html.Div([
            html.P(f"File: {filename}"),
            html.P(f"Rows: {len(df)}"),
            html.P(f"Parameters: {len(pso_data['param_names'])}"),
            html.P(f"Objectives: {len(pso_data['obj_names'])}"),
            html.P(f"Pareto Front Points: {len(pso_data['pareto_objectives'])}")
        ])

        return file_info, sliders, {'margin': '20px', 'display': 'block'}, {'margin': '20px', 'textAlign': 'center', 'display': 'block'}

    except Exception as e:
        return html.P(f"Error loading file: {e}"), [], {'display': 'none'}, {'display': 'none'}

@app.callback(
    Output('main-plot', 'figure'),
    [Input('upload-data', 'contents'),
     Input({'type': 'param-slider', 'index': ALL}, 'value'),
     Input({'type': 'obj-slider', 'index': ALL}, 'value'),
     Input('target-input', 'value')]
)
def update_main_plot(contents, param_slider_values, obj_slider_values, target_id):
    # If no data loaded yet
    if pso_data['objectives'] is None:
        return go.Figure()

    triggered = callback_context.triggered[0]['prop_id'].split('.')[0]

    # On file upload, show full plot with fixed axis ranges and no filtering
    if triggered == 'upload-data':
        return create_scatter_matrix(
            pso_data['objectives'],
            pso_data['pareto_objectives'],
            target_point_id=0,
            fixed_axis_ranges=(pso_data['obj_mins'], pso_data['obj_maxs'])
        )

    # Otherwise, apply filters to Pareto front only
    mask = np.ones(len(pso_data['pareto_objectives']), dtype=bool)

    for i, slider_range in enumerate(param_slider_values):
        low, high = slider_range
        mask &= (pso_data['pareto_positions'][:, i] >= low) & (pso_data['pareto_positions'][:, i] <= high)

    for i, slider_range in enumerate(obj_slider_values):
        low, high = slider_range
        mask &= (pso_data['pareto_objectives'][:, i] >= low) & (pso_data['pareto_objectives'][:, i] <= high)

    filtered_positions = pso_data['pareto_positions'][mask]
    filtered_objectives = pso_data['pareto_objectives'][mask]

    if len(filtered_objectives) == 0:
        filtered_objectives = np.empty((0, pso_data['pareto_objectives'].shape[1]))

    if target_id is None or target_id >= len(pso_data['objectives']) or target_id < 0:
        target_id = 0

    # No fixed axis ranges on filtering to prevent zoom reset
    return create_scatter_matrix(pso_data['objectives'], filtered_objectives, target_id)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)



print(f"Full objectives shape: {pso_data['objectives'].shape if pso_data['objectives'] is not None else None}")
print(f"Pareto objectives shape: {pso_data['pareto_objectives'].shape if pso_data['pareto_objectives'] is not None else None}")

