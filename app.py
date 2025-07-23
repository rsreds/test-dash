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

# Global storage for PSO data
pso_data = {
    'parameters': None,
    'objectives': None,
    'original_objectives': None,  # Keep original for reset
    'pareto_objectives': None,
    'pareto_positions': None,
    'param_names': [],
    'obj_names': [],
    'filename': None,
    'obj_mins': None,  
    'obj_maxs': None,
    'selected_indices': set(),
    'current_hover_point': None,
    'activity_log': [],
    'displayed_objectives': None,  # New: which objectives to display
    'max_objectives': 0  # New: maximum available objectives
}

def log_activity(message):
    """Add message to activity log with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    pso_data['activity_log'].append(f"[{timestamp}] {message}")
    # Keep only last 50 messages
    if len(pso_data['activity_log']) > 50:
        pso_data['activity_log'] = pso_data['activity_log'][-50:]

def get_displayed_objectives():
    """Get the currently displayed objectives - all objectives are displayed"""
    if pso_data['objectives'] is None:
        return None, []
    
    return pso_data['objectives'], pso_data['obj_names']

def create_slider(title, slider_id, min_val, max_val):
    return html.Div([
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

def create_sliders():
    """Create sliders based on current data"""
    sliders = []
    
    try:
        # Parameter sliders
        if (pso_data['parameters'] is not None and 
            len(pso_data['parameters']) > 0 and 
            pso_data['parameters'].shape[1] > 0 and
            len(pso_data['param_names']) > 0):
            
            sliders.append(html.H6("Parameter Filters:", className="mt-3 mb-2"))
            for i, name in enumerate(pso_data['param_names']):
                if i < pso_data['parameters'].shape[1]:
                    param_min = float(np.min(pso_data['parameters'][:, i]))
                    param_max = float(np.max(pso_data['parameters'][:, i]))
                    if param_min != param_max:  # Avoid sliders with same min/max
                        sliders.append(create_slider(f"Param {i+1} ({name})", 
                                                   {'type': 'param-slider', 'index': i}, 
                                                   param_min, param_max))

        # Objective sliders
        if (pso_data['objectives'] is not None and 
            len(pso_data['objectives']) > 0 and
            len(pso_data['obj_names']) > 0):
            
            sliders.append(html.H6("Objective Filters:", className="mt-3 mb-2"))
            for i, name in enumerate(pso_data['obj_names']):
                if i < pso_data['objectives'].shape[1]:
                    obj_min = float(np.min(pso_data['objectives'][:, i]))
                    obj_max = float(np.max(pso_data['objectives'][:, i]))
                    if obj_min != obj_max:  # Avoid sliders with same min/max
                        sliders.append(create_slider(f"Obj {i+1} ({name})", 
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
            is_dominated = np.all(points[is_pareto] <= c, axis=1) & np.any(points[is_pareto] < c, axis=1)
            is_pareto[is_pareto] = ~is_dominated
            is_pareto[i] = True
    return is_pareto

def create_interactive_scatter_matrix(full_objectives, pareto_objectives, target_point_id=0, selected_indices=None, fixed_axis_ranges=None, filter_mask=None):
    """Create interactive scatter matrix with selection capabilities"""
    if selected_indices is None:
        selected_indices = set()
    
    if filter_mask is None:
        filter_mask = np.ones(len(full_objectives), dtype=bool)
    
    # Get currently displayed objectives
    displayed_objectives, displayed_names = get_displayed_objectives()
    
    if displayed_objectives is None or len(displayed_objectives) == 0:
        fig = go.Figure()
        fig.update_layout(title="No data to display")
        return fig
    
    num_obj = displayed_objectives.shape[1]
    obj_names = displayed_names

    if target_point_id >= len(displayed_objectives):
        target_point_id = 0

    # Calculate Pareto front for displayed objectives only
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
                # Diagonal elements - show objective name
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
                # Create color and size arrays for interactive plotting
                colors = []
                sizes = []
                symbols = []
                customdata = []
                opacities = []
                
                for idx in range(len(displayed_objectives)):
                    customdata.append(idx)
                    
                    # Apply filtering - make filtered points nearly invisible
                    if not filter_mask[idx]:
                        colors.append('lightgray')
                        sizes.append(1)
                        symbols.append('circle')
                        opacities.append(0.1)
                        continue
                    
                    # Color logic: selected > pareto > regular
                    if idx in selected_indices:
                        colors.append('red' if pareto_mask[idx] else 'orange')
                        sizes.append(12 if pareto_mask[idx] else 10)
                    else:
                        colors.append('blue' if pareto_mask[idx] else 'lightblue')
                        sizes.append(6 if pareto_mask[idx] else 4)
                    
                    # Special handling for target point
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
                            f"{obj_names[i]}: %{{y:.3f}}<br>"
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
    
    # Add legend annotation
    legend_text = (
        "<b>Legend:</b><br>"
        "🔵 Pareto Optimal<br>"
        "🔷 Regular<br>"
        "🔴 Selected<br>"
        "⭐ Target"
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

# Enhanced layout with interactive controls
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Interactive CSV PSO Visualizer", className="text-center mb-4"),
            
            # File upload section
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
    
    # Control panels (hidden until file loaded)
    html.Div(id='control-panels', style={'display': 'none'}, children=[
        dbc.Row([
            # Interactive controls
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Interactive Controls", className="card-title"),
                        
                        # Objective selection controls
                        html.H6("Data Structure:", className="mt-3 mb-2"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Number of Objectives in CSV:", style={'fontSize': '12px'}),
                                dbc.Input(id='num-objectives', type='number', value=2, min=1, max=50, size='sm')
                            ], width=4),
                            dbc.Col([
                                dbc.Button("Apply Structure", id='apply-obj-selection-btn', color="primary", size='sm', style={'marginTop': '20px'})
                            ], width=3),
                            dbc.Col([
                                html.Div(id='objective-info', style={'fontSize': '11px', 'marginTop': '5px'})
                            ], width=5)
                        ], className="mb-3"),
                        
                        # Selection tools
                        html.H6("Selection Tools:", className="mt-3 mb-2"),
                        dbc.Row([
                            dbc.Col([
                                dbc.InputGroup([
                                    dbc.Input(id='random-count', type='number', value=10, min=1, max=100, size='sm'),
                                    dbc.Button("Random", id='select-random-btn', color="info", size='sm')
                                ])
                            ], width=4),
                            dbc.Col([
                                dbc.Button("Pareto", id='select-pareto-btn', color="info", size='sm', style={'width': '100%'})
                            ], width=2),
                            dbc.Col([
                                dbc.Button("Worst", id='select-worst-btn', color="info", size='sm', style={'width': '100%'})
                            ], width=2),
                            dbc.Col([
                                dbc.Button("Best", id='select-best-btn', color="info", size='sm', style={'width': '100%'})
                            ], width=2),
                            dbc.Col([
                                dbc.Button("Clear", id='clear-selection-btn', color="warning", size='sm', style={'width': '100%'})
                            ], width=2)
                        ], className="mb-3"),
                        
                        # Action buttons
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
                            ], width=3),
                            dbc.Col([
                                dbc.InputGroup([
                                    dbc.Input(id='target-input', type='number', value=0, min=0, size='sm', placeholder="Target ID"),
                                ])
                            ], width=3)
                        ]),
                        
                        # Status display
                        html.Div(id='status-display', className="mt-3 p-2 bg-light rounded")
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Main plot
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='main-plot', style={'height': '80vh'}, config={'displayModeBar': True})
            ], width=9),
            
            # Filters sidebar
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Filters", className="card-title"),
                        html.Div(id='slider-container')
                    ])
                ])
            ], width=3)
        ]),
        
        # Activity panel and log
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

# Store for maintaining selection state
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

        # Handle data modification triggers - regenerate sliders
        if trigger in ['delete-selected-btn', 'keep-selected-btn', 'reset-data-btn']:
            if pso_data['objectives'] is not None:
                sliders = create_sliders()
                
                file_info = dbc.Alert([
                    html.H6(f"📁 {pso_data['filename']}", className="alert-heading"),
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

        # Handle file upload
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        # Get all columns and total count
        all_columns = df.columns.tolist()
        total_columns = len(all_columns)
        
        # Default to 2 objectives if not specified
        if num_objectives_input is None or num_objectives_input < 1:
            num_objectives_input = min(2, total_columns)
        
        # Ensure we don't ask for more objectives than available columns
        num_objectives = min(num_objectives_input, total_columns)
        
        # Apply supervisor's logic: last N columns are objectives, first (total-N) are parameters
        num_parameters = total_columns - num_objectives
        
        if num_parameters < 0:
            raise ValueError(f"Cannot have {num_objectives} objectives with only {total_columns} columns")
        
        # Split columns based on position
        param_cols = all_columns[:num_parameters] if num_parameters > 0 else []
        obj_cols = all_columns[num_parameters:num_parameters + num_objectives]
        
        # Extract the data
        param_data = df[param_cols].values if param_cols else np.array([]).reshape(len(df), 0)
        obj_data = df[obj_cols].values
        
        # Store in global data
        pso_data['parameters'] = param_data
        pso_data['objectives'] = obj_data
        pso_data['original_objectives'] = obj_data.copy()
        pso_data['original_parameters'] = param_data.copy() if len(param_data) > 0 else None
        pso_data['param_names'] = param_cols
        pso_data['obj_names'] = obj_cols
        pso_data['filename'] = filename
        pso_data['selected_indices'] = set()
        pso_data['activity_log'] = []
        pso_data['max_objectives'] = total_columns  # Can't exceed total columns
        
        # All specified objectives are displayed
        pso_data['displayed_objectives'] = list(range(num_objectives))
        
        # Calculate bounds for parameters
        if len(param_data) > 0 and param_data.shape[1] > 0:
            pso_data['lb'] = np.min(param_data, axis=0)
            pso_data['ub'] = np.max(param_data, axis=0)
        else:
            pso_data['lb'] = np.array([])
            pso_data['ub'] = np.array([])
        
        # Store global min/max for each objective
        pso_data['obj_mins'] = np.min(obj_data, axis=0)
        pso_data['obj_maxs'] = np.max(obj_data, axis=0)

        # Calculate initial Pareto front
        pareto_mask = filter_pareto_front(obj_data)
        pso_data['pareto_objectives'] = obj_data[pareto_mask]
        pso_data['pareto_positions'] = param_data[pareto_mask] if len(param_data) > 0 else np.array([])

        # Create sliders using the new function
        sliders = create_sliders()

        file_info = dbc.Alert([
            html.H6(f"📁 {filename}", className="alert-heading"),
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
    [Output('objective-info', 'children')],
    [Input('apply-obj-selection-btn', 'n_clicks')],
    [State('num-objectives', 'value')],
    prevent_initial_call=True
)
def update_objective_info(n_clicks, num_obj):
    if not n_clicks or pso_data['objectives'] is None:
        return [""]
    
    total_columns = len(pso_data['param_names']) + len(pso_data['obj_names'])
    
    if num_obj is None:
        num_obj = 2
        
    num_obj = max(1, min(num_obj, total_columns))
    num_params = total_columns - num_obj
    
    obj_info = f"Structure: {num_params} parameters + {num_obj} objectives = {total_columns} total columns"
    log_activity(f"Updated data structure: {obj_info}")
    
    return [obj_info]

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
     Input('select-random-btn', 'n_clicks'),
     Input('select-pareto-btn', 'n_clicks'),
     Input('select-worst-btn', 'n_clicks'),
     Input('select-best-btn', 'n_clicks'),
     Input('clear-selection-btn', 'n_clicks'),
     Input('delete-selected-btn', 'n_clicks'),
     Input('keep-selected-btn', 'n_clicks'),
     Input('reset-data-btn', 'n_clicks'),
     Input('apply-obj-selection-btn', 'n_clicks')],
    [State('random-count', 'value'),
     State('selection-store', 'data')],
    prevent_initial_call=False
)
def update_visualization(contents, param_slider_values, obj_slider_values, target_id, 
                        selection_store, selected_data, click_data, 
                        random_clicks, pareto_clicks, worst_clicks, best_clicks,
                        clear_clicks, delete_clicks, keep_clicks, reset_clicks, obj_selection_clicks,
                        random_count, current_selection_store):
    
    try:
        # If no data loaded yet
        if pso_data['objectives'] is None or len(pso_data['objectives']) == 0:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="Upload a CSV file to begin")
            return empty_fig, "No data loaded", "Upload CSV file", []

        ctx = callback_context
        if not ctx.triggered:
            trigger = 'upload-data'
        else:
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]

        # Get currently displayed objectives
        displayed_objectives, displayed_names = get_displayed_objectives()
        
        if displayed_objectives is None or len(displayed_objectives) == 0:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No data available")
            return empty_fig, "No data", "No data", []

        # Validate and clean selected indices
        valid_selected = {idx for idx in pso_data['selected_indices'] 
                         if isinstance(idx, int) and 0 <= idx < len(displayed_objectives)}
        pso_data['selected_indices'] = valid_selected

        # Handle various button clicks and selection changes
        if 'select-random-btn' in trigger and random_clicks:
            n = min(random_count or 10, len(displayed_objectives))
            if len(displayed_objectives) > 0:
                pso_data['selected_indices'] = set(np.random.choice(len(displayed_objectives), n, replace=False))
                log_activity(f"Selected {n} random points")
            
        elif 'select-pareto-btn' in trigger and pareto_clicks:
            if len(displayed_objectives) > 0:
                pareto_mask = filter_pareto_front(displayed_objectives)
                pareto_indices = np.where(pareto_mask)[0]
                n = min(random_count or 10, len(pareto_indices))
                pso_data['selected_indices'] = set(pareto_indices[:n])
                log_activity(f"Selected {n} Pareto optimal points")
            
        elif 'select-worst-btn' in trigger and worst_clicks:
            if len(displayed_objectives) > 0:
                total_obj = np.sum(displayed_objectives, axis=1)
                n = min(random_count or 10, len(displayed_objectives))
                worst_indices = np.argsort(total_obj)[-n:]
                pso_data['selected_indices'] = set(worst_indices)
                log_activity(f"Selected {n} worst performing points")
            
        elif 'select-best-btn' in trigger and best_clicks:
            if len(displayed_objectives) > 0:
                total_obj = np.sum(displayed_objectives, axis=1)
                n = min(random_count or 10, len(displayed_objectives))
                best_indices = np.argsort(total_obj)[:n]
                pso_data['selected_indices'] = set(best_indices)
                log_activity(f"Selected {n} best performing points")
            
        elif 'clear-selection-btn' in trigger and clear_clicks:
            pso_data['selected_indices'] = set()
            log_activity("Cleared selection")
            
        elif 'delete-selected-btn' in trigger and delete_clicks:
            if pso_data['selected_indices'] and len(displayed_objectives) > 0:
                count = len(pso_data['selected_indices'])
                keep_mask = np.ones(len(pso_data['objectives']), dtype=bool)
                for idx in pso_data['selected_indices']:
                    if 0 <= idx < len(pso_data['objectives']):
                        keep_mask[idx] = False
                
                if np.any(keep_mask):  # Ensure we don't delete everything
                    pso_data['objectives'] = pso_data['objectives'][keep_mask]
                    if pso_data['parameters'] is not None and len(pso_data['parameters']) > 0:
                        pso_data['parameters'] = pso_data['parameters'][keep_mask]
                    
                    pso_data['selected_indices'] = set()
                    
                    # Recalculate bounds safely
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
                    
                    # Recalculate bounds safely
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
                
                # Recalculate bounds from original data
                if len(pso_data['objectives']) > 0:
                    pso_data['obj_mins'] = np.min(pso_data['objectives'], axis=0)
                    pso_data['obj_maxs'] = np.max(pso_data['objectives'], axis=0)
                
                if (pso_data['parameters'] is not None and 
                    len(pso_data['parameters']) > 0 and 
                    pso_data['parameters'].shape[1] > 0):
                    pso_data['lb'] = np.min(pso_data['parameters'], axis=0)
                    pso_data['ub'] = np.max(pso_data['parameters'], axis=0)
                
                log_activity("Reset data to original")
            
        # Handle plot interactions
        elif 'main-plot' in trigger and selected_data and selected_data.get('points'):
            new_selection = set()
            for point in selected_data['points']:
                if 'customdata' in point and isinstance(point['customdata'], int):
                    idx = point['customdata']
                    if 0 <= idx < len(pso_data['objectives']):
                        new_selection.add(idx)
            if new_selection != pso_data['selected_indices']:
                pso_data['selected_indices'] = new_selection
                log_activity(f"Selected {len(new_selection)} points via drag selection")
        
        elif ('main-plot' in trigger and click_data and click_data.get('points') and 
              not (selected_data and selected_data.get('points'))):
            clicked_point = click_data['points'][0]
            if 'customdata' in clicked_point and isinstance(clicked_point['customdata'], int):
                point_id = clicked_point['customdata']
                if 0 <= point_id < len(pso_data['objectives']):
                    if point_id in pso_data['selected_indices']:
                        pso_data['selected_indices'].remove(point_id)
                        log_activity(f"Deselected point #{point_id}")
                    else:
                        pso_data['selected_indices'].add(point_id)
                        log_activity(f"Selected point #{point_id}")

        # Validate target_id
        current_data_length = len(displayed_objectives)
        if target_id is None or target_id >= current_data_length or target_id < 0:
            target_id = 0

        # Get current objectives and parameters safely
        current_objectives = displayed_objectives.copy()
        current_parameters = (pso_data['parameters'].copy() 
                            if pso_data['parameters'] is not None and len(pso_data['parameters']) > 0 
                            else None)
        
        # Apply filters safely - skip filtering if sliders don't match data
        filter_mask = np.ones(len(current_objectives), dtype=bool)
        
        # DEFENSIVE FILTERING: Only apply filters if dimensions exactly match
        try:
            # Apply parameter filters only if everything matches perfectly
            if (current_parameters is not None and 
                len(param_slider_values) > 0 and 
                current_parameters.shape[1] > 0):
                
                # EXACT match required - if slider count doesn't match parameter count, skip filtering
                if len(param_slider_values) == current_parameters.shape[1]:
                    for i, slider_range in enumerate(param_slider_values):
                        if (i < current_parameters.shape[1] and 
                            len(slider_range) == 2 and
                            all(np.isfinite(slider_range))):
                            low, high = slider_range
                            # Additional safety check for array shapes
                            if current_parameters[:, i].shape == filter_mask.shape:
                                filter_mask &= (current_parameters[:, i] >= low) & (current_parameters[:, i] <= high)
                else:
                    # Slider count mismatch - skip parameter filtering
                    log_activity(f"Skipping parameter filters: {len(param_slider_values)} sliders vs {current_parameters.shape[1]} parameters")
            
            # Apply objective filters only if everything matches perfectly  
            if (len(obj_slider_values) > 0 and 
                current_objectives.shape[1] > 0):
                
                # EXACT match required - if slider count doesn't match objective count, skip filtering
                if len(obj_slider_values) == current_objectives.shape[1]:
                    for i, slider_range in enumerate(obj_slider_values):
                        if (i < current_objectives.shape[1] and 
                            len(slider_range) == 2 and
                            all(np.isfinite(slider_range))):
                            low, high = slider_range
                            # Additional safety check for array shapes
                            if current_objectives[:, i].shape == filter_mask.shape:
                                filter_mask &= (current_objectives[:, i] >= low) & (current_objectives[:, i] <= high)
                else:
                    # Slider count mismatch - skip objective filtering
                    log_activity(f"Skipping objective filters: {len(obj_slider_values)} sliders vs {current_objectives.shape[1]} objectives")
                    
        except Exception as filter_error:
            # If any filtering fails, just use no filtering
            filter_mask = np.ones(len(current_objectives), dtype=bool)
            log_activity(f"Filter error, using no filtering: {str(filter_error)}")

        # Create the plot safely
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
            # Fallback to empty plot if visualization fails
            fig = go.Figure()
            fig.update_layout(title=f"Visualization Error: {str(plot_error)}")
            log_activity(f"Plot error: {str(plot_error)}")

        # Update status safely
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

        # Update activity panel safely
        try:
            activity_content = "No point information available"
            if target_id < len(current_objectives):
                obj_values = current_objectives[target_id]
                is_pareto = (filter_pareto_front(current_objectives)[target_id] 
                           if len(current_objectives) > 0 else False)
                is_selected = target_id in pso_data['selected_indices']
                
                total_obj = np.sum(obj_values)
                all_totals = np.sum(current_objectives, axis=1) if len(current_objectives) > 0 else np.array([])
                rank = np.sum(all_totals < total_obj) + 1 if len(all_totals) > 0 else 1
                ideal_distance = np.sqrt(np.sum(obj_values**2))
                
                obj_names = pso_data.get('obj_names', [f'Obj_{i}' for i in range(len(obj_values))])
                obj_display = " | ".join([f"{obj_names[i] if i < len(obj_names) else f'Obj_{i}'}: {val:.4f}" 
                                        for i, val in enumerate(obj_values[:5])])
                
                activity_content = html.Div([
                    html.P(f"Point #{target_id} | {'Selected' if is_selected else 'Not Selected'} | {'Pareto' if is_pareto else 'Non-Pareto'}", 
                           style={'fontWeight': 'bold'}),
                    html.P(obj_display, style={'fontSize': '12px'}),
                    html.P(f"Total: {total_obj:.4f} | Rank: {rank}/{len(current_objectives)} | Ideal Dist: {ideal_distance:.4f}", 
                           style={'fontSize': '11px'})
                ])
        except Exception:
            activity_content = "Point information error"

        # Update activity log safely
        try:
            log_content = []
            if 'activity_log' in pso_data and pso_data['activity_log']:
                for log_msg in reversed(pso_data['activity_log'][-20:]):
                    log_content.append(html.Div(str(log_msg), style={'marginBottom': '2px'}))
        except Exception:
            log_content = [html.Div("Log error")]

        return fig, status_content, activity_content, log_content

    except Exception as e:
        # Ultimate fallback for any unhandled errors
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Callback Error: {str(e)}")
        error_status = f"Error: {str(e)}"
        error_activity = f"Callback failed: {str(e)}"
        error_log = [html.Div(f"Critical error: {str(e)}")]
        
        # Log the error
        try:
            log_activity(f"Critical callback error: {str(e)}")
        except:
            pass
            
        return error_fig, error_status, error_activity, error_log

# Store selection data
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