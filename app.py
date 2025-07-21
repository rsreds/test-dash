import base64
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, State, ctx

app = Dash(__name__)
app.title = "CSV pso Visualizer"

# Global storage for PSO data
pso_data = {
    'parameters': None,
    'objectives': None,
    'pareto_objectives': None,
    'pareto_positions': None,
    'param_names': [],
    'obj_names': []
}

def create_range_slider_component(name, min_val, max_val, step=0.01):
    return html.Div([
        html.Label(f'{name} Range:'),
        dcc.RangeSlider(
            id={'type': 'dynamic-slider', 'index': name},
            min=min_val,
            max=max_val,
            step=step,
            value=[min_val, max_val],
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={'margin': '10px'})

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
            is_pareto[i] = True  # Ensure self isn't marked false
    return is_pareto

app.layout = html.Div([
    html.H1("CSV-Based PSO Visualization"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ', html.A('Select a CSV File')
        ]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'margin': '10px'
        },
        multiple=False,
        accept='.csv'
    ),
    html.Div(id='sliders-container'),
    dcc.Graph(id='pareto-graph')
])

@app.callback(
    Output('sliders-container', 'children'),
    Output('pareto-graph', 'figure'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def process_file(contents, filename):
    if contents is None:
        return [], go.Figure()

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    num_cols = df.shape[1]
    param_data = df.iloc[:, :-3].values
    obj_data = df.iloc[:, -3:].values

    param_names = df.columns[:-3].tolist()
    obj_names = df.columns[-3:].tolist()

    pso_data['parameters'] = param_data
    pso_data['objectives'] = obj_data
    pso_data['param_names'] = param_names
    pso_data['obj_names'] = obj_names

    pareto_mask = filter_pareto_front(obj_data)
    pso_data['pareto_objectives'] = obj_data[pareto_mask]
    pso_data['pareto_positions'] = param_data[pareto_mask]

    sliders = []
    for i, name in enumerate(param_names + obj_names):
        data = param_data[:, i] if i < len(param_names) else obj_data[:, i - len(param_names)]
        sliders.append(create_range_slider_component(name, float(np.min(data)), float(np.max(data))))

    return sliders, update_plot([])

@app.callback(
    Output('pareto-graph', 'figure'),
    Input({'type': 'dynamic-slider', 'index': ALL}, 'value')
)
def update_plot(slider_values):
    if pso_data['pareto_objectives'] is None:
        return go.Figure()

    positions = pso_data['pareto_positions']
    objectives = pso_data['pareto_objectives']
    param_names = pso_data['param_names']
    obj_names = pso_data['obj_names']

    all_filters = slider_values
    param_filters = all_filters[:len(param_names)]
    obj_filters = all_filters[len(param_names):]

    mask = np.ones(len(objectives), dtype=bool)
    for i, (min_val, max_val) in enumerate(param_filters):
        mask &= (positions[:, i] >= min_val) & (positions[:, i] <= max_val)

    for i, (min_val, max_val) in enumerate(obj_filters):
        mask &= (objectives[:, i] >= min_val) & (objectives[:, i] <= max_val)

    filtered_obj = objectives[mask]
    fig = make_subplots(
        rows=len(obj_names), cols=len(obj_names),
        subplot_titles=[f"{obj_names[j]} vs {obj_names[i]}" for i in range(len(obj_names)) for j in range(len(obj_names))]
    )

    for i in range(len(obj_names)):
        for j in range(len(obj_names)):
            if i == j:
                fig.add_trace(go.Scatter(x=filtered_obj[:, j], y=filtered_obj[:, i],
                                         mode='markers', marker=dict(color='blue'),
                                         name='Pareto Optimal'), row=i+1, col=j+1)
            else:
                fig.add_trace(go.Scatter(x=objectives[:, j], y=objectives[:, i],
                                         mode='markers', marker=dict(color='gray', opacity=0.4),
                                         name='All Points', showlegend=(i == 0 and j == 1)),
                              row=i+1, col=j+1)
                fig.add_trace(go.Scatter(x=filtered_obj[:, j], y=filtered_obj[:, i],
                                         mode='markers', marker=dict(color='blue'),
                                         name='Pareto Optimal', showlegend=(i == 0 and j == 1)),
                              row=i+1, col=j+1)

    fig.update_layout(height=300 * len(obj_names), width=300 * len(obj_names),
                      title_text="Scatter Plot Matrix of Objectives")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
