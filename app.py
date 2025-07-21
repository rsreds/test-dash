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

def create_param_slider_component(index, name, min_val, max_val):
    return html.Div([
        html.Label(f'{name} Range:'),
        dcc.RangeSlider(
            id={'type': 'param-slider', 'index': index},
            min=min_val,
            max=max_val,
            step=(max_val - min_val) / 100 if max_val > min_val else 0.01,
            value=[min_val, max_val],
            tooltip={"placement": "bottom", "always_visible": True},
            allowCross=False
        )
    ], style={'margin': '15px'})

def create_objective_slider_component(index, name, min_val, max_val):
    return html.Div([
        html.Label(f'{name} Range:'),
        dcc.RangeSlider(
            id={'type': 'objective-slider', 'index': index},
            min=min_val,
            max=max_val,
            step=(max_val - min_val) / 100 if max_val > min_val else 0.01,
            value=[min_val, max_val],
            tooltip={"placement": "bottom", "always_visible": True},
            allowCross=False
        )
    ], style={'margin': '15px'})

def filter_pareto_front(points):
    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i, c in enumerate(points):
        if is_pareto[i]:
            is_dominated = np.all(points[is_pareto] <= c, axis=1) & np.any(points[is_pareto] < c, axis=1)
            is_pareto[is_pareto] = ~is_dominated
            is_pareto[i] = True
    return is_pareto

def create_scatter_matrix(positions, objectives, param_names, obj_names):
    num_obj = len(obj_names)
    fig = make_subplots(
        rows=num_obj, cols=num_obj,
        subplot_titles=[f"{obj_names[j]} vs {obj_names[i]}" for i in range(num_obj) for j in range(num_obj)],
        vertical_spacing=0.07,
        horizontal_spacing=0.07
    )

    for i in range(num_obj):
        for j in range(num_obj):
            row, col = i + 1, j + 1
            if i == j:
                fig.add_trace(go.Scatter(x=objectives[:, j], y=objectives[:, i],
                                         mode='markers',
                                         marker=dict(color='gray'),
                                         name='Pareto Front' if (i == 0 and j == 0) else None,
                                         showlegend=(i == 0 and j == 0)),
                              row=row, col=col)
            else:
                fig.add_trace(go.Scatter(x=objectives[:, j], y=objectives[:, i],
                                         mode='markers',
                                         marker=dict(color='blue'),
                                         name='Pareto Front' if (i == 0 and j == 1) else None,
                                         showlegend=(i == 0 and j == 1)),
                              row=row, col=col)

    fig.update_layout(
        height=300 * num_obj,
        width=300 * num_obj,
        title_text="Scatter Plot Matrix of Objectives"
    )

    # Axis labels
    for i in range(num_obj):
        fig.update_xaxes(title_text=obj_names[i], row=num_obj, col=i+1)
        fig.update_yaxes(title_text=obj_names[i], row=i+1, col=1)

    return fig

app.layout = html.Div([
    html.H1("CSV-Based PSO Visualization"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select a CSV File')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False,
        accept='.csv'
    ),
    dcc.Graph(id='pareto-graph', style={'marginBottom': '40px'}),
    html.Div(id='param-slider-container'),
    html.Div(id='objective-slider-container'),
])

@app.callback(
    [Output('param-slider-container', 'children'),
     Output('objective-slider-container', 'children'),
     Output('pareto-graph', 'figure')],
    [Input('upload-data', 'contents'),
     Input({'type': 'param-slider', 'index': ALL}, 'value'),
     Input({'type': 'objective-slider', 'index': ALL}, 'value')],
    prevent_initial_call=True
)
def update_sliders_and_figure(contents, param_slider_values, objective_slider_values):
    if contents is None:
        return [], [], go.Figure()

    triggered = ctx.triggered_id

    # On file upload, create sliders and initial plot
    if triggered == 'upload-data':
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

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

        param_sliders = [html.H3("Parameter Filters")] + [
            create_param_slider_component(i, name, float(np.min(param_data[:, i])), float(np.max(param_data[:, i])))
            for i, name in enumerate(param_names)
        ]

        objective_sliders = [html.H3("Objective Filters")] + [
            create_objective_slider_component(i, name, float(np.min(obj_data[:, i])), float(np.max(obj_data[:, i])))
            for i, name in enumerate(obj_names)
        ]

        fig = create_scatter_matrix(pso_data['pareto_positions'], pso_data['pareto_objectives'], param_names, obj_names)

        return param_sliders, objective_sliders, fig

    # On slider change, update plot filtering by sliders
    else:
        if pso_data['pareto_objectives'] is None:
            return dash.no_update, dash.no_update, go.Figure()

        positions = pso_data['pareto_positions']
        objectives = pso_data['pareto_objectives']
        param_names = pso_data['param_names']
        obj_names = pso_data['obj_names']

        mask = np.ones(len(objectives), dtype=bool)

        # Filter by params sliders
        for i, slider_range in enumerate(param_slider_values):
            if slider_range and len(slider_range) == 2:
                low, high = slider_range
                mask &= (positions[:, i] >= low) & (positions[:, i] <= high)

        # Filter by objectives sliders
        for i, slider_range in enumerate(objective_slider_values):
            if slider_range and len(slider_range) == 2:
                low, high = slider_range
                mask &= (objectives[:, i] >= low) & (objectives[:, i] <= high)

        filtered_positions = positions[mask]
        filtered_objectives = objectives[mask]

        if len(filtered_objectives) == 0:
            return dash.no_update, dash.no_update, go.Figure()

        fig = create_scatter_matrix(filtered_positions, filtered_objectives, param_names, obj_names)

        return dash.no_update, dash.no_update, fig


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
