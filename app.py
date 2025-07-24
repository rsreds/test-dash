#works perfectly with adjustments
import base64
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, State, ALL, callback_context
import dash_bootstrap_components as dbc
from datetime import datetime

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Interactive CSV PSO Visualizer"
app.config.suppress_callback_exceptions = True

# Global storage for PSO data
pso_data = {
    'parameters': None,
    'objectives': None,
    'original_objectives': None,
    'pareto_objectives': None,
    'pareto_positions': None,
    'param_names': [],
    'obj_names': [],
    'filename': None,
    'obj_mins': None,
    'obj_maxs': None,
    'selected_indices': set(),
    'current_clicked_point': None,  # Track last clicked point
    'activity_log': [],
    'displayed_objectives': None,
    'max_objectives': 0,
    'show_param_plots': True  # Track parameter plot visibility
}

def log_activity(message):
    """Add message to activity log with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    pso_data['activity_log'].append(f"[{timestamp}] {message}")
    if len(pso_data['activity_log']) > 50:
        pso_data['activity_log'] = pso_data['activity_log'][-50:]

def get_displayed_objectives():
    """Get the currently displayed objectives"""
    if pso_data['objectives'] is None:
        return None, []
    return pso_data['objectives'], pso_data['obj_names']

def create_parameter_mini_plot(param_data, param_name, param_index, filter_mask=None):
    """Create a small scatter plot for parameter distribution"""
    if param_data is None or len(param_data) == 0 or param_index >= param_data.shape[1]:
        return go.Figure()

    if filter_mask is None:
        filter_mask = np.ones(len(param_data), dtype=bool)

    param_values = param_data[:, param_index]

    # Create y-values as just indices for scatter plot
    y_values = np.arange(len(param_values))

    # Color points based on filter
    colors = ['blue' if mask else 'lightgray' for mask in filter_mask]
    opacities = [0.8 if mask else 0.3 for mask in filter_mask]
    sizes = [4 if mask else 2 for mask in filter_mask]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=param_values,
            y=y_values,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=opacities,
                line=dict(width=0.5, color='gray')
            ),
            hovertemplate=f"<b>{param_name}</b><br>Value: %{{x:.4f}}<br>Point: %{{y}}<extra></extra>",
            showlegend=False
        )
    )

    fig.update_layout(
        height=80,
        margin=dict(l=10, r=10, t=5, b=20),
        xaxis=dict(title="", tickfont=dict(size=8)),
        yaxis=dict(showticklabels=False, showgrid=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def create_slider(title, slider_id, min_val, max_val, param_index=None, show_plot=True, filter_mask=None):
    """Create slider with optional mini parameter plot"""
    slider_div = html.Div([
        html.P(f"Filter by {title}:", style={'marginBottom': '5px', 'marginTop': '15px', 'fontSize': '14px'}),
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
    ], style={'marginBottom': '15px'})

    # Add mini plot for parameters if requested
    if param_index is not None and show_plot and pso_data['parameters'] is not None:
        mini_plot = dcc.Graph(
            id={'type': 'param-mini-plot', 'index': param_index},
            figure=create_parameter_mini_plot(pso_data['parameters'], title, param_index, filter_mask),
            style={'height': '80px', 'marginBottom': '5px'},
            config={'displayModeBar': False}
        )
        return html.Div([mini_plot, slider_div])

    return slider_div

def create_sliders():
    """Create sliders based on current data"""
    sliders = []

    try:
        # Add control buttons at the top
        control_buttons = html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Button("Toggle Param Plots", id='toggle-param-plots-btn',
                              color="info", size='sm', style={'width': '100%'})
                ], width=6),
                dbc.Col([
                    dbc.Button("Reset All Sliders", id='reset-sliders-btn',
                              color="secondary", size='sm', style={'width': '100%'})
                ], width=6)
            ], className="mb-3")
        ])
        sliders.append(control_buttons)

        # Create a filter mask for current filtering state
        # Note: The filter_mask here will be based on the last applied filters.
        # This function primarily rebuilds the slider UI, the actual filtering happens in update_visualization.
        filter_mask = np.ones(len(pso_data['objectives']) if pso_data['objectives'] is not None else 0, dtype=bool)

        # Parameter sliders - use just the parameter name
        if (pso_data['parameters'] is not None and
            len(pso_data['parameters']) > 0 and
            pso_data['parameters'].shape[1] > 0 and
            len(pso_data['param_names']) > 0):

            sliders.append(html.H6("Parameter Filters:", className="mt-3 mb-2"))
            for i, name in enumerate(pso_data['param_names']):
                if i < pso_data['parameters'].shape[1]:
                    param_min = float(np.min(pso_data['parameters'][:, i]))
                    param_max = float(np.max(pso_data['parameters'][:, i]))
                    if param_min != param_max:
                        sliders.append(create_slider(name,  # Just the name, no "Param 1" prefix
                                                   {'type': 'param-slider', 'index': i},
                                                   param_min, param_max,
                                                   param_index=i,
                                                   show_plot=pso_data.get('show_param_plots', True),
                                                   filter_mask=filter_mask))

        # Objective sliders - use just the objective name
        if (pso_data['objectives'] is not None and
            len(pso_data['objectives']) > 0 and
            len(pso_data['obj_names']) > 0):

            sliders.append(html.H6("Objective Filters:", className="mt-3 mb-2"))
            for i, name in enumerate(pso_data['obj_names']):
                if i < pso_data['objectives'].shape[1]:
                    obj_min = float(np.min(pso_data['objectives'][:, i]))
                    obj_max = float(np.max(pso_data['objectives'][:, i]))
                    if obj_min != obj_max:
                        sliders.append(create_slider(name,  # Just the name, no "Obj 1" prefix
                                                   {'type': 'obj-slider', 'index': i},
                                                   obj_min, obj_max))
    except Exception as e:
        sliders = [html.Div(f"Error creating sliders: {str(e)}", style={'color': 'red'})]

    return sliders

def filter_pareto_front(points):
    """Calculate Pareto front from points"""
    if len(points) == 0:
        return np.array([], dtype=bool)

    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i, c in enumerate(points):
        if is_pareto[i]:
            # is_dominated = np.all(points[is_pareto] <= c, axis=1) & np.any(points[is_pareto] < c, axis=1)
            # Corrected Pareto filtering: A point A dominates B if all objectives of A are <= B AND at least one objective of A is < B.
            # We want to find points that are *not* dominated by any other point.
            # So, for each point 'c', check if any other point 'p' in the set dominates 'c'.
            # A point 'p' dominates 'c' if (p <= c for all objectives) AND (p < c for at least one objective).
            dominated_by_any_other = np.any(np.all(points <= c, axis=1) & np.any(points < c, axis=1), axis=0)
            is_pareto[i] = not dominated_by_any_other # Mark as non-pareto if dominated

    # Re-calculate as the original logic can be inefficient and sometimes incorrect for large sets
    # This is a more robust N*M implementation (N points, M objectives)
    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i in range(points.shape[0]):
        for j in range(points.shape[0]):
            if i == j:
                continue
            # Check if point j dominates point i
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                is_pareto[i] = False
                break
    return is_pareto


def create_interactive_scatter_matrix(full_objectives, pareto_objectives, target_point_id=0, selected_indices=None, fixed_axis_ranges=None, filter_mask=None):
    """Create interactive scatter matrix with selection capabilities"""
    if selected_indices is None:
        selected_indices = set()

    if filter_mask is None:
        filter_mask = np.ones(len(full_objectives), dtype=bool)

    displayed_objectives, displayed_names = get_displayed_objectives()

    if displayed_objectives is None or len(displayed_objectives) == 0:
        fig = go.Figure()
        fig.update_layout(title="No data to display")
        return fig

    num_obj = displayed_objectives.shape[1]
    obj_names = displayed_names

    if target_point_id >= len(displayed_objectives):
        target_point_id = 0

    pareto_mask = filter_pareto_front(displayed_objectives)

    subplot_titles = []
    for i in range(num_obj):
        for j in range(num_obj):
            subplot_titles.append(f'{obj_names[j]} vs {obj_names[i]}' if i != j else obj_names[i])

    fig = make_subplots(
        rows=num_obj, cols=num_obj,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    for i in range(num_obj):
        for j in range(num_obj):
            row, col = i + 1, j + 1

            if i == j:
                fig.add_trace(
                    go.Scatter(x=[0.5], y=[0.5], mode='text',
                             text=[f'<b>{obj_names[i]}</b>'],
                             textfont=dict(size=16, color='#2E4057'),
                             showlegend=False, hoverinfo='none'),
                    row=row, col=col
                )
                fig.update_xaxes(range=[0, 1], showticklabels=False, showgrid=False, row=row, col=col)
                fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False, row=row, col=col)
            else:
                colors = []
                sizes = []
                symbols = []
                customdata = []
                opacities = []

                for idx in range(len(displayed_objectives)):
                    customdata.append(idx)

                    if not filter_mask[idx]:
                        colors.append('lightgray')
                        sizes.append(3)
                        symbols.append('circle')
                        opacities.append(0.2)  # Faded instead of very transparent
                        continue

                    if idx in selected_indices:
                        colors.append('red' if pareto_mask[idx] else 'orange')
                        sizes.append(12 if pareto_mask[idx] else 10)
                    else:
                        colors.append('blue' if pareto_mask[idx] else 'lightblue')
                        sizes.append(6 if pareto_mask[idx] else 4)

                    if idx == target_point_id:
                        colors[-1] = 'darkred'
                        sizes[-1] = 12
                        symbols.append('star')
                    else:
                        symbols.append('circle')

                    opacities.append(0.8)

                fig.add_trace(
                    go.Scatter(
                        x=displayed_objectives[:, j],
                        y=displayed_objectives[:, i],
                        mode='markers',
                        marker=dict(
                            size=sizes,
                            color=colors,
                            symbol=symbols,
                            opacity=opacities,
                            line=dict(width=1, color='gray')
                        ),
                        customdata=customdata,
                        hovertemplate=(
                            f"<b>Point #%{{customdata}}</b><br>"
                            f"{obj_names[j]}: %{{x:.3f}}<br>"
                            f"{obj_names[i]}: %{{y:.3f}}<br}"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                        selectedpoints=list(selected_indices) if selected_indices else None
                    ),
                    row=row, col=col
                )

                fig.update_xaxes(title_text=obj_names[j], title_font=dict(size=11), row=row, col=col)
                fig.update_yaxes(title_text=obj_names[i], title_font=dict(size=11), row=row, col=col)

    fig.update_layout(
        title=dict(
            text=f'<b>{num_obj}×{num_obj} Interactive Multi-Objective Optimization Matrix</b>',
            font=dict(size=20, color='#2E4057'),
            x=0.5
        ),
        height=num_obj * 250,
        showlegend=False,
        dragmode='select',
        selectdirection='d',
        margin=dict(l=60, r=60, t=100, b=60)
    )

    legend_text = (
        "<b>Legend:</b><br>"
        "Blue: Pareto Optimal<br>"
        "Light Blue: Regular<br>"
        "Red: Selected<br>"
        "Star: Target"
    )

    fig.add_annotation(
        text=legend_text,
        xref="paper", yref="paper",
        x=1.02, y=1.0,
        xanchor="left", yanchor="top",
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor="rgba(0, 0, 0, 0.2)",
        borderwidth=1,
        font=dict(size=10, color="#2E4057"),
        align="left"
    )

    return fig

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Interactive CSV PSO Visualizer", className="text-center mb-4"),

            dbc.Card([
                dbc.CardBody([
                    html.H5("Upload CSV File", className="card-title"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select CSV File', style={'textDecoration': 'underline'})
                        ]),
                        style={
                            'width': '100%', 'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'borderColor': '#007bff'
                        },
                        multiple=False,
                        accept='.csv'
                    ),
                    html.Div(id='file-info', className="mt-3")
                ])
            ], className="mb-4")
        ])
    ]),

    html.Div(id='control-panels', style={'display': 'none'}, children=[
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Interactive Controls", className="card-title"),

                        html.H6("Data Structure:", className="mt-3 mb-2"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Number of Objectives in CSV:", style={'fontSize': '12px'}),
                                dbc.Input(id='num-objectives', type='number', value=2, min=1, max=50, size='sm')
                            ], width=4),
                            dbc.Col([
                                dbc.Button("Apply Structure", id='apply-obj-selection-btn', color="primary", size='sm', style={'marginTop': '20px'})
                            ], width=3)
                        ], className="mb-3"),

                        html.H6("Actions:", className="mb-2"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Delete Selected", id='delete-selected-btn', color="danger", size='sm', style={'width': '100%'})
                            ], width=3),
                            dbc.Col([
                                dbc.Button("Keep Selected", id='keep-selected-btn', color="success", size='sm', style={'width': '100%'})
                            ], width=3),
                            dbc.Col([
                                dbc.Button("Reset Data", id='reset-data-btn', color="secondary", size='sm', style={'width': '100%'})
                            ], width=2),
                            dbc.Col([
                                dbc.Button("Clear Selection", id='clear-selection-btn', color="warning", size='sm', style={'width': '100%'})
                            ], width=2),
                            dbc.Col([
                                dbc.Input(id='target-input', type='number', value=0, min=0, size='sm', placeholder="Target Point")
                            ], width=2)
                        ]),

                        html.Div(id='status-display', className="mt-3 p-2 bg-light rounded")
                    ])
                ])
            ], width=12)
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dcc.Graph(id='main-plot', style={'height': '80vh'}, config={'displayModeBar': True})
            ], width=9),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Filters", className="card-title"),
                        html.Div(id='slider-container')
                    ])
                ])
            ], width=3)
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Point Information", className="card-title"),
                        html.Div(id='activity-panel', className="text-center")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Activity Log", className="card-title"),
                        html.Div(id='activity-log',
                               style={'height': '200px', 'overflowY': 'scroll', 'fontSize': '12px'})
                    ])
                ])
            ], width=6)
        ], className="mt-3")
    ])
], fluid=True)

app.layout.children.append(dcc.Store(id='selection-store', data={'selected_indices': []}))

@app.callback(
    [Output('file-info', 'children'),
     Output('slider-container', 'children'),
     Output('control-panels', 'style'),
     Output('target-input', 'max'),
     Output('num-objectives', 'max')],
    [Input('upload-data', 'contents'),
     Input('apply-obj-selection-btn', 'n_clicks'),
     Input('delete-selected-btn', 'n_clicks'),
     Input('keep-selected-btn', 'n_clicks'),
     Input('reset-data-btn', 'n_clicks')],
    [State('upload-data', 'filename'),
     State('num-objectives', 'value')],
    prevent_initial_call=False
)
def load_csv_and_process(contents, apply_clicks, delete_clicks, keep_clicks, reset_clicks, filename, num_objectives_input):
    if contents is None:
        return '', [], {'display': 'none'}, 0, 2

    try:
        ctx = callback_context
        if not ctx.triggered:
            trigger = 'upload-data'
        else:
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger in ['delete-selected-btn', 'keep-selected-btn', 'reset-data-btn']:
            if pso_data['objectives'] is not None:
                sliders = create_sliders()

                file_info = dbc.Alert([
                    html.H6(f"File: {pso_data['filename']}", className="alert-heading"),
                    html.P([
                        f"Rows: {len(pso_data['objectives'])} | ",
                        f"Total Columns: {len(pso_data['param_names']) + len(pso_data['obj_names'])} | ",
                        f"Parameters: {len(pso_data['param_names'])} | ",
                        f"Objectives: {len(pso_data['obj_names'])} | ",
                        f"Pareto Points: {np.sum(filter_pareto_front(pso_data['objectives'])) if len(pso_data['objectives']) > 0 else 0}"
                    ], className="mb-0")
                ], color="success")

                return file_info, sliders, {'display': 'block'}, len(pso_data['objectives']) - 1, len(pso_data['param_names']) + len(pso_data['obj_names'])
            else:
                return '', [], {'display': 'none'}, 0, 2

        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        all_columns = df.columns.tolist()
        total_columns = len(all_columns)

        if num_objectives_input is None or num_objectives_input < 1:
            num_objectives_input = min(2, total_columns)

        num_objectives = min(num_objectives_input, total_columns)
        num_parameters = total_columns - num_objectives

        if num_parameters < 0:
            raise ValueError(f"Cannot have {num_objectives} objectives with only {total_columns} columns")

        param_cols = all_columns[:num_parameters] if num_parameters > 0 else []
        obj_cols = all_columns[num_parameters:num_parameters + num_objectives]

        param_data = df[param_cols].values if param_cols else np.array([]).reshape(len(df), 0)
        obj_data = df[obj_cols].values

        pso_data['parameters'] = param_data
        pso_data['objectives'] = obj_data
        pso_data['original_objectives'] = obj_data.copy()
        pso_data['original_parameters'] = param_data.copy() if len(param_data) > 0 else None
        pso_data['param_names'] = param_cols
        pso_data['obj_names'] = obj_cols
        pso_data['filename'] = filename
        pso_data['selected_indices'] = set()
        pso_data['current_clicked_point'] = None
        pso_data['activity_log'] = []
        pso_data['max_objectives'] = total_columns
        pso_data['displayed_objectives'] = list(range(num_objectives))
        pso_data['show_param_plots'] = True  # Reset to show plots by default

        if len(param_data) > 0 and param_data.shape[1] > 0:
            pso_data['lb'] = np.min(param_data, axis=0)
            pso_data['ub'] = np.max(param_data, axis=0)
        else:
            pso_data['lb'] = np.array([])
            pso_data['ub'] = np.array([])

        pso_data['obj_mins'] = np.min(obj_data, axis=0)
        pso_data['obj_maxs'] = np.max(obj_data, axis=0)

        pareto_mask = filter_pareto_front(obj_data)
        pso_data['pareto_objectives'] = obj_data[pareto_mask]
        pso_data['pareto_positions'] = param_data[pareto_mask] if len(param_data) > 0 else np.array([])

        sliders = create_sliders()

        file_info = dbc.Alert([
            html.H6(f"File: {filename}", className="alert-heading"),
            html.P([
                f"Rows: {len(df)} | ",
                f"Total Columns: {total_columns} | ",
                f"Parameters: {len(param_cols)} | ",
                f"Objectives: {len(obj_cols)} | ",
                f"Pareto Points: {len(pso_data['pareto_objectives'])}"
            ], className="mb-0")
        ], color="success")

        log_activity(f"Loaded {filename}: {len(df)} points, {len(param_cols)} parameters, {len(obj_cols)} objectives")

        return file_info, sliders, {'display': 'block'}, len(obj_data) - 1, total_columns

    except Exception as e:
        error_msg = dbc.Alert(f"Error loading file: {str(e)}", color="danger")
        return error_msg, [], {'display': 'none'}, 0, 2

@app.callback(
    [Output('main-plot', 'figure'),
     Output('status-display', 'children'),
     Output('activity-panel', 'children'),
     Output('activity-log', 'children')],
    [Input('upload-data', 'contents'),
     Input({'type': 'param-slider', 'index': ALL}, 'value'),
     Input({'type': 'obj-slider', 'index': ALL}, 'value'),
     Input('target-input', 'value'),
     Input('selection-store', 'data'),
     Input('main-plot', 'selectedData'),
     Input('main-plot', 'clickData'),
     Input('clear-selection-btn', 'n_clicks'),
     Input('delete-selected-btn', 'n_clicks'),
     Input('keep-selected-btn', 'n_clicks'),
     Input('reset-data-btn', 'n_clicks'),
     Input('apply-obj-selection-btn', 'n_clicks')],
    [State('selection-store', 'data')],
    prevent_initial_call=False
)
def update_visualization(contents, param_slider_values, obj_slider_values, target_id,
                        selection_store, selected_data, click_data,
                        clear_clicks, delete_clicks, keep_clicks, reset_clicks, obj_selection_clicks,
                        current_selection_store):

    try:
        if pso_data['objectives'] is None or len(pso_data['objectives']) == 0:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="Upload a CSV file to begin")
            return empty_fig, "No data loaded", "Upload CSV file", []

        ctx = callback_context
        if not ctx.triggered:
            trigger = 'upload-data'
        else:
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]

        displayed_objectives, displayed_names = get_displayed_objectives()

        if displayed_objectives is None or len(displayed_objectives) == 0:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No data available")
            return empty_fig, "No data", "No data", []

        # --- SELECTION HANDLING LOGIC ---
        # Prioritize drag selection: if drag occurs, clear single click state
        if 'main-plot' in trigger and selected_data and selected_data.get('points'):
            new_selection = set()
            for point in selected_data['points']:
                if 'customdata' in point and isinstance(point['customdata'], int):
                    idx = point['customdata']
                    if 0 <= idx < len(pso_data['objectives']):
                        new_selection.add(idx)
            if new_selection != pso_data['selected_indices']:
                pso_data['selected_indices'] = new_selection
                pso_data['current_clicked_point'] = None # Clear single clicked point on drag select
                log_activity(f"Selected {len(new_selection)} points via drag selection")

        # Handle single click, ONLY if no drag selection happened in this trigger
        elif ('main-plot' in trigger and click_data and click_data.get('points') and
              not (selected_data and selected_data.get('points'))): # Ensure no drag selection
            clicked_point = click_data['points'][0]
            if 'customdata' in clicked_point and isinstance(clicked_point['customdata'], int):
                point_id = clicked_point['customdata']
                if 0 <= point_id < len(pso_data['objectives']):
                    # Clear previous selection if it was a multi-selection
                    if len(pso_data['selected_indices']) > 1:
                        pso_data['selected_indices'] = set()
                    
                    pso_data['current_clicked_point'] = point_id  # Store clicked point
                    if point_id in pso_data['selected_indices']:
                        pso_data['selected_indices'].remove(point_id)
                        log_activity(f"Deselected point #{point_id}")
                    else:
                        pso_data['selected_indices'].add(point_id)
                        log_activity(f"Selected point #{point_id}")
        # --- END SELECTION HANDLING ---

        valid_selected = {idx for idx in pso_data['selected_indices']
                         if isinstance(idx, int) and 0 <= idx < len(displayed_objectives)}
        pso_data['selected_indices'] = valid_selected

        # Handle button clicks
        if 'clear-selection-btn' in trigger and clear_clicks:
            pso_data['selected_indices'] = set()
            pso_data['current_clicked_point'] = None
            log_activity("Cleared selection")

        elif 'delete-selected-btn' in trigger and delete_clicks:
            if pso_data['selected_indices'] and len(displayed_objectives) > 0:
                count = len(pso_data['selected_indices'])
                keep_mask = np.ones(len(pso_data['objectives']), dtype=bool)
                for idx in pso_data['selected_indices']:
                    if 0 <= idx < len(pso_data['objectives']):
                        keep_mask[idx] = False

                if np.any(keep_mask):
                    pso_data['objectives'] = pso_data['objectives'][keep_mask]
                    if pso_data['parameters'] is not None and len(pso_data['parameters']) > 0:
                        pso_data['parameters'] = pso_data['parameters'][keep_mask]

                    pso_data['selected_indices'] = set()
                    pso_data['current_clicked_point'] = None

                    if len(pso_data['objectives']) > 0:
                        pso_data['obj_mins'] = np.min(pso_data['objectives'], axis=0)
                        pso_data['obj_maxs'] = np.max(pso_data['objectives'], axis=0)

                    if (pso_data['parameters'] is not None and
                        len(pso_data['parameters']) > 0 and
                        pso_data['parameters'].shape[1] > 0):
                        pso_data['lb'] = np.min(pso_data['parameters'], axis=0)
                        pso_data['ub'] = np.max(pso_data['parameters'], axis=0)

                    log_activity(f"Deleted {count} points, {len(pso_data['objectives'])} remaining")
                else:
                    log_activity("Cannot delete all points")

        elif 'keep-selected-btn' in trigger and keep_clicks:
            if pso_data['selected_indices'] and len(displayed_objectives) > 0:
                valid_indices = [idx for idx in pso_data['selected_indices']
                               if 0 <= idx < len(pso_data['objectives'])]

                if valid_indices:
                    pso_data['objectives'] = pso_data['objectives'][valid_indices]
                    if pso_data['parameters'] is not None and len(pso_data['parameters']) > 0:
                        pso_data['parameters'] = pso_data['parameters'][valid_indices]

                    pso_data['selected_indices'] = set()
                    pso_data['current_clicked_point'] = None

                    if len(pso_data['objectives']) > 0:
                        pso_data['obj_mins'] = np.min(pso_data['objectives'], axis=0)
                        pso_data['obj_maxs'] = np.max(pso_data['objectives'], axis=0)

                    if (pso_data['parameters'] is not None and
                        len(pso_data['parameters']) > 0 and
                        pso_data['parameters'].shape[1] > 0):
                        pso_data['lb'] = np.min(pso_data['parameters'], axis=0)
                        pso_data['ub'] = np.max(pso_data['parameters'], axis=0)

                    log_activity(f"Kept only {len(valid_indices)} selected points")
                else:
                    log_activity("No valid selected points to keep")

        elif 'reset-data-btn' in trigger and reset_clicks:
            if 'original_objectives' in pso_data and pso_data['original_objectives'] is not None:
                pso_data['objectives'] = pso_data['original_objectives'].copy()
                if 'original_parameters' in pso_data and pso_data['original_parameters'] is not None:
                    pso_data['parameters'] = pso_data['original_parameters'].copy()
                pso_data['selected_indices'] = set()
                pso_data['current_clicked_point'] = None

                if len(pso_data['objectives']) > 0:
                    pso_data['obj_mins'] = np.min(pso_data['objectives'], axis=0)
                    pso_data['obj_maxs'] = np.max(pso_data['objectives'], axis=0)

                if (pso_data['parameters'] is not None and
                    len(pso_data['parameters']) > 0 and
                    pso_data['parameters'].shape[1] > 0):
                    pso_data['lb'] = np.min(pso_data['parameters'], axis=0)
                    pso_data['ub'] = np.max(pso_data['parameters'], axis=0)

                log_activity("Reset data to original")
        
        # This part is now handled by the general selection logic above
        # elif 'main-plot' in trigger and selected_data and selected_data.get('points'):
        #     new_selection = set()
        #     for point in selected_data['points']:
        #         if 'customdata' in point:
        #             idx = point['customdata']
        #             if 0 <= idx < len(pso_data['objectives']):
        #                 new_selection.add(idx)
        #     if new_selection != pso_data['selected_indices']:
        #         pso_data['selected_indices'] = new_selection
        #         log_activity(f"Selected {len(new_selection)} points via drag selection")

        current_data_length = len(displayed_objectives)
        if target_id is None or target_id >= current_data_length or target_id < 0:
            target_id = 0

        current_objectives = displayed_objectives.copy()
        current_parameters = (pso_data['parameters'].copy()
                            if pso_data['parameters'] is not None and len(pso_data['parameters']) > 0
                            else None)

        filter_mask = np.ones(len(current_objectives), dtype=bool)

        try:
            if (current_parameters is not None and
                len(param_slider_values) > 0 and
                current_parameters.shape[1] > 0):

                if len(param_slider_values) == current_parameters.shape[1]:
                    for i, slider_range in enumerate(param_slider_values):
                        if (i < current_parameters.shape[1] and
                            len(slider_range) == 2 and
                            all(np.isfinite(slider_range))):
                            low, high = slider_range
                            if current_parameters[:, i].shape == filter_mask.shape:
                                filter_mask &= (current_parameters[:, i] >= low) & (current_parameters[:, i] <= high)
                else:
                    log_activity(f"Skipping parameter filters: {len(param_slider_values)} sliders vs {current_parameters.shape[1]} parameters")

            if (len(obj_slider_values) > 0 and
                current_objectives.shape[1] > 0):

                if len(obj_slider_values) == current_objectives.shape[1]:
                    for i, slider_range in enumerate(obj_slider_values):
                        if (i < current_objectives.shape[1] and
                            len(slider_range) == 2 and
                            all(np.isfinite(slider_range))):
                            low, high = slider_range
                            if current_objectives[:, i].shape == filter_mask.shape:
                                filter_mask &= (current_objectives[:, i] >= low) & (current_objectives[:, i] <= high)
                else:
                    log_activity(f"Skipping objective filters: {len(obj_slider_values)} sliders vs {current_objectives.shape[1]} objectives")

        except Exception as filter_error:
            filter_mask = np.ones(len(current_objectives), dtype=bool)
            log_activity(f"Filter error, using no filtering: {str(filter_error)}")

        try:
            fig = create_interactive_scatter_matrix(
                current_objectives,
                pso_data.get('pareto_objectives', np.array([])),
                target_id,
                pso_data['selected_indices'],
                (pso_data['obj_mins'], pso_data['obj_maxs']) if 'obj_mins' in pso_data else None,
                filter_mask
            )
        except Exception as plot_error:
            fig = go.Figure()
            fig.update_layout(title=f"Visualization Error: {str(plot_error)}")
            log_activity(f"Plot error: {str(plot_error)}")

        try:
            pareto_count = (np.sum(filter_pareto_front(current_objectives))
                          if len(current_objectives) > 0 else 0)

            stats = {
                'total': len(current_objectives),
                'pareto': pareto_count,
                'selected': len(pso_data['selected_indices']),
                'objectives': current_objectives.shape[1] if len(current_objectives) > 0 else 0
            }

            status_content = dbc.Row([
                dbc.Col(html.Strong(f"Total: {stats['total']}"), width=2),
                dbc.Col(html.Strong(f"Pareto: {stats['pareto']}", style={'color': 'blue'}), width=2),
                dbc.Col(html.Strong(f"Selected: {stats['selected']}", style={'color': 'red'}), width=2),
                dbc.Col(html.Strong(f"Target: #{target_id}", style={'color': 'darkred'}), width=3),
                dbc.Col(html.Strong(f"Objectives: {stats['objectives']}", style={'color': 'green'}), width=3)
            ])
        except Exception:
            status_content = "Status update error"

        # Updated point information display
        try:
            activity_content = "No point information available"

            # Priority: multiple selection > clicked point > single selected > target
            display_point_id = None
            display_type = "none"

            if len(pso_data['selected_indices']) > 1:
                display_type = "multiple_selected"
            elif (pso_data['current_clicked_point'] is not None and
                  pso_data['current_clicked_point'] < len(current_objectives) and
                  pso_data['current_clicked_point'] in pso_data['selected_indices']): # Only show clicked if still in selected
                display_point_id = pso_data['current_clicked_point']
                display_type = "clicked"
            elif len(pso_data['selected_indices']) == 1:
                display_point_id = list(pso_data['selected_indices'])[0]
                display_type = "single_selected"
            elif target_id < len(current_objectives):
                display_point_id = target_id
                display_type = "target"

            if display_type == "multiple_selected": # Changed from 'multiple' to 'multiple_selected' for clarity
                selected_indices_list = list(pso_data['selected_indices'])
                selected_objectives = current_objectives[selected_indices_list]

                num_selected = len(selected_indices_list)
                
                # Calculate aggregated objective values
                obj_names = pso_data.get('obj_names', [f'Obj_{i}' for i in range(selected_objectives.shape[1])])
                obj_avg_display = [html.P("Average Objective Values:", style={'fontWeight': 'bold', 'fontSize': '12px', 'margin': '2px 0'})]
                for i, name in enumerate(obj_names):
                    if selected_objectives.shape[0] > 0: # Avoid mean of empty slice
                        avg_val = np.mean(selected_objectives[:, i])
                        obj_avg_display.append(html.P(f"{name}: {avg_val:.4f}", style={'fontSize': '11px', 'margin': '2px 0'}))
                
                # Calculate aggregated parameter values (if available)
                param_avg_display = html.Div()
                if current_parameters is not None and len(current_parameters) > 0:
                    selected_parameters = current_parameters[selected_indices_list]
                    param_names = pso_data.get('param_names', [f'Param_{i}' for i in range(selected_parameters.shape[1])])
                    
                    param_avg_display_list = [html.P("Average Parameter Values:", style={'fontWeight': 'bold', 'fontSize': '12px', 'margin': '2px 0'})]
                    for i, name in enumerate(param_names):
                        if selected_parameters.shape[0] > 0: # Avoid mean of empty slice
                            avg_val = np.mean(selected_parameters[:, i])
                            param_avg_display_list.append(html.P(f"{name}: {avg_val:.4f}", style={'fontSize': '11px', 'margin': '2px 0'}))
                    param_avg_display = html.Div(param_avg_display_list)

                # Calculate Pareto count within selected points
                pareto_mask_selected = filter_pareto_front(selected_objectives)
                num_pareto_selected = np.sum(pareto_mask_selected)

                activity_content = html.Div([
                    html.P(f"{num_selected} Points Selected | {num_pareto_selected} Pareto Optimal (within selection)",
                           style={'fontWeight': 'bold', 'color': 'darkblue'}),
                    html.Hr(style={'margin': '5px 0'}),
                    html.Div(obj_avg_display),
                    param_avg_display # This will be an empty Div if no params
                ])

            elif display_point_id is not None:
                # Show detailed info for single point
                obj_values = current_objectives[display_point_id]
                param_values = (current_parameters[display_point_id]
                              if current_parameters is not None and len(current_parameters) > 0
                              else None)
                is_pareto = (filter_pareto_front(current_objectives)[display_point_id]
                           if len(current_objectives) > 0 else False)

                # Calculate distance to ideal point
                ideal_point = np.min(current_objectives, axis=0)
                ideal_distance = np.sqrt(np.sum((obj_values - ideal_point)**2))

                obj_names = pso_data.get('obj_names', [f'Obj_{i}' for i in range(len(obj_values))])
                param_names = pso_data.get('param_names', [f'Param_{i}' for i in range(len(param_values) if param_values is not None else 0)])

                if display_type == "clicked":
                    point_label = f"Clicked Point #{display_point_id}"
                    color = 'purple'
                elif display_type == "single_selected":
                    point_label = f"Selected Point #{display_point_id}"
                    color = 'red'
                else:  # target
                    point_label = f"Target Point #{display_point_id}"
                    color = 'darkred'

                # Create objective values display
                obj_display = html.Div([
                    html.P("Objective Values:", style={'fontWeight': 'bold', 'fontSize': '12px', 'margin': '2px 0'}),
                    html.P(" | ".join([f"{obj_names[i] if i < len(obj_names) else f'Obj_{i}'}: {val:.4f}"
                                     for i, val in enumerate(obj_values)]),
                           style={'fontSize': '11px', 'margin': '2px 0'})
                ])

                # Create parameter values display (if available)
                param_display = html.Div()
                if param_values is not None and len(param_values) > 0:
                    param_display = html.Div([
                        html.P("Parameter Values:", style={'fontWeight': 'bold', 'fontSize': '12px', 'margin': '2px 0'}),
                        html.P(" | ".join([f"{param_names[i] if i < len(param_names) else f'Param_{i}'}: {val:.4f}"
                                         for i, val in enumerate(param_values)]),
                               style={'fontSize': '11px', 'margin': '2px 0'})
                    ])

                activity_content = html.Div([
                    html.P(f"{point_label} | {'Pareto' if is_pareto else 'Non-Pareto'}",
                           style={'fontWeight': 'bold', 'color': color}),
                    html.Hr(style={'margin': '5px 0'}),
                    obj_display,
                    param_display,
                    html.P(f"Ideal Distribution: {ideal_distance:.4f}",
                           style={'fontSize': '11px', 'fontWeight': 'bold', 'color': 'black'})
                ])

        except Exception as activity_error:
            activity_content = f"Point information error: {str(activity_error)}"

        try:
            log_content = []
            if 'activity_log' in pso_data and pso_data['activity_log']:
                for log_msg in reversed(pso_data['activity_log'][-20:]):
                    log_content.append(html.Div(str(log_msg), style={'marginBottom': '2px'}))
        except Exception:
            log_content = [html.Div("Log error")]

        return fig, status_content, activity_content, log_content

    except Exception as e:
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Callback Error: {str(e)}")
        error_status = f"Error: {str(e)}"
        error_activity = f"Callback failed: {str(e)}"
        error_log = [html.Div(f"Critical error: {str(e)}")]

        try:
            log_activity(f"Critical callback error: {str(e)}")
        except:
            pass

        return error_fig, error_status, error_activity, error_log

# Add separate callback to handle slider updates when reset is clicked
@app.callback(
    Output('slider-container', 'children', allow_duplicate=True),
    [Input('reset-sliders-btn', 'n_clicks'),
     Input('toggle-param-plots-btn', 'n_clicks')],
    prevent_initial_call=True
)
def update_sliders_on_reset(reset_clicks, toggle_clicks):
    """Update sliders when reset button is clicked or plots are toggled"""
    try:
        ctx = callback_context
        if not ctx.triggered:
            return create_sliders()

        trigger = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger == 'toggle-param-plots-btn':
            pso_data['show_param_plots'] = not pso_data.get('show_param_plots', True)
            log_activity(f"Parameter plots {'shown' if pso_data['show_param_plots'] else 'hidden'}")
        elif trigger == 'reset-sliders-btn':
            log_activity("Reset all sliders to default ranges")

        return create_sliders()
    except Exception as e:
        return create_sliders()

# Add callback to update parameter mini plots when filters change
@app.callback(
    [Output({'type': 'param-mini-plot', 'index': ALL}, 'figure')],
    [Input({'type': 'param-slider', 'index': ALL}, 'value'),
     Input({'type': 'obj-slider', 'index': ALL}, 'value'),
     Input('upload-data', 'contents'),
     Input('delete-selected-btn', 'n_clicks'),
     Input('keep-selected-btn', 'n_clicks'),
     Input('reset-data-btn', 'n_clicks')],
    prevent_initial_call=True
)
def update_parameter_mini_plots(param_slider_values, obj_slider_values, contents,
                               delete_clicks, keep_clicks, reset_clicks):
    """Update parameter mini plots based on current filtering"""
    if (pso_data['parameters'] is None or
        len(pso_data['parameters']) == 0 or
        not pso_data.get('show_param_plots', True)):
        return [[]]

    try:
        # Calculate current filter mask
        filter_mask = np.ones(len(pso_data['objectives']), dtype=bool)

        # Apply parameter filters
        if (param_slider_values and
            len(param_slider_values) == pso_data['parameters'].shape[1]):
            for i, slider_range in enumerate(param_slider_values):
                if (len(slider_range) == 2 and
                    all(np.isfinite(slider_range)) and
                    i < pso_data['parameters'].shape[1]):
                    low, high = slider_range
                    filter_mask &= (pso_data['parameters'][:, i] >= low) & (pso_data['parameters'][:, i] <= high)

        # Apply objective filters
        if (obj_slider_values and
            len(obj_slider_values) == pso_data['objectives'].shape[1]):
            for i, slider_range in enumerate(obj_slider_values):
                if (len(slider_range) == 2 and
                    all(np.isfinite(slider_range)) and
                    i < pso_data['objectives'].shape[1]):
                    low, high = slider_range
                    filter_mask &= (pso_data['objectives'][:, i] >= low) & (pso_data['objectives'][:, i] <= high)

        # Create updated mini plots
        updated_figures = []
        for i, param_name in enumerate(pso_data['param_names']):
            if i < pso_data['parameters'].shape[1]:
                fig = create_parameter_mini_plot(pso_data['parameters'], param_name, i, filter_mask)
                updated_figures.append(fig)

        return [updated_figures]

    except Exception as e:
        # Return empty figures on error
        return [[go.Figure() for _ in range(len(pso_data.get('param_names', [])))]]

@app.callback(
    Output('selection-store', 'data'),
    Input('main-plot', 'selectedData'),
    State('selection-store', 'data')
)
def update_selection_store(selected_data, current_data):
    if selected_data and selected_data.get('points'):
        selected_indices = []
        for point in selected_data['points']:
            if 'customdata' in point:
                selected_indices.append(point['customdata'])
        return {'selected_indices': selected_indices}
    return current_data

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)