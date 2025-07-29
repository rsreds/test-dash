import base64
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, State, ALL, callback_context
from dash.exceptions import PreventUpdate
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
    'lb': np.array([]),
    'ub': np.array([]),
    'selected_indices': set(),
    'current_clicked_point': None,
    'activity_log': [],
    'displayed_objectives': None,
    'max_objectives': 0,
    'original_df': None,
    'original_parameters': None,
    'structure_id': 0  # Unique ID for each structure change
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

def calculate_plot_height(num_objectives):
    """Calculate dynamic plot height based on number of objectives"""
    if num_objectives <= 0:
        return 400
    return num_objectives * 250

def create_slider(title, slider_id, min_val, max_val):
    """Create slider with proper range handling"""
    epsilon = 1e-9
    if min_val == max_val:
        min_val_display = min_val - epsilon
        max_val_display = max_val + epsilon
        marks_display = {min_val: f'{min_val:.1f}'}
    else:
        min_val_display = min_val
        max_val_display = max_val
        marks_display = {min_val: f'{min_val:.1f}', max_val: f'{max_val:.1f}'}

    slider_div = html.Div([
        html.P(f"{title}:", style={'marginBottom': '5px', 'marginTop': '15px', 'fontSize': '14px'}),
        dcc.RangeSlider(
            id=slider_id,
            min=min_val_display,
            max=max_val_display,
            step=(max_val_display - min_val_display) / 100 if max_val_display > min_val_display else 0.01,
            value=[min_val, max_val],
            marks=marks_display,
            tooltip={"placement": "bottom", "always_visible": False},
            updatemode='drag'
        )
    ], style={'marginBottom': '15px'})

    return slider_div

def update_data_structure(df, num_objectives):
    """Update PSO data structure with new objectives/parameters split"""
    try:
        all_columns = df.columns.tolist()
        total_columns = len(all_columns)
        
        num_objectives = min(num_objectives, total_columns)
        num_parameters = total_columns - num_objectives
        
        if num_parameters < 0:
            raise ValueError(f"Cannot have {num_objectives} objectives with only {total_columns} columns")
        
        # Split columns
        param_cols = all_columns[:num_parameters] if num_parameters > 0 else []
        obj_cols = all_columns[num_parameters:num_parameters + num_objectives]
        
        # Extract data
        param_data = df[param_cols].values if param_cols else np.array([]).reshape(len(df), 0)
        obj_data = df[obj_cols].values
        
        # Update global data
        pso_data['parameters'] = param_data
        pso_data['objectives'] = obj_data
        pso_data['original_objectives'] = obj_data.copy()
        pso_data['original_parameters'] = param_data.copy() if len(param_data) > 0 else None
        pso_data['param_names'] = param_cols
        pso_data['obj_names'] = obj_cols
        pso_data['selected_indices'] = set()
        pso_data['current_clicked_point'] = None
        pso_data['max_objectives'] = total_columns
        pso_data['displayed_objectives'] = list(range(num_objectives))
        
        # Calculate bounds
        if len(param_data) > 0 and param_data.shape[1] > 0:
            pso_data['lb'] = np.min(param_data, axis=0)
            pso_data['ub'] = np.max(param_data, axis=0)
        else:
            pso_data['lb'] = np.array([])
            pso_data['ub'] = np.array([])
            
        if len(obj_data) > 0 and obj_data.shape[1] > 0:
            pso_data['obj_mins'] = np.min(obj_data, axis=0)
            pso_data['obj_maxs'] = np.max(obj_data, axis=0)
        else:
            pso_data['obj_mins'] = np.array([])
            pso_data['obj_maxs'] = np.array([])
        
        # Calculate Pareto front
        pareto_mask = filter_pareto_front(obj_data)
        pso_data['pareto_objectives'] = obj_data[pareto_mask]
        pso_data['pareto_positions'] = param_data[pareto_mask] if len(param_data) > 0 else np.array([])
        
        # Increment structure ID to force slider recreation
        pso_data['structure_id'] += 1
        
        log_activity(f"Structure updated: {len(param_cols)} params, {len(obj_cols)} objs (ID: {pso_data['structure_id']})")
        
        return True, None
        
    except Exception as e:
        log_activity(f"Structure update failed: {str(e)}")
        return False, str(e)

def create_sliders():
    """Create sliders based on current data structure"""
    sliders = []
    
    try:
        # Control buttons
        sliders.append(html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Button("Reset All Sliders", id='reset-sliders-btn',
                              color="secondary", size='sm', style={'width': '100%'})
                ], width=12)
            ], className="mb-3")
        ]))
        
        # Current structure info
        struct_info = f"Structure ID: {pso_data['structure_id']} | Params: {len(pso_data['param_names'])} | Objs: {len(pso_data['obj_names'])}"
        sliders.append(html.Div(struct_info, style={'fontSize': '10px', 'color': 'gray', 'marginBottom': '10px'}))
        
        # Parameter sliders
        if (pso_data['parameters'] is not None and 
            len(pso_data['parameters']) > 0 and 
            pso_data['parameters'].shape[1] > 0 and 
            len(pso_data['lb']) > 0):
            
            sliders.append(html.H6("Parameter Filters:", className="mt-3 mb-2"))
            
            for i in range(len(pso_data['param_names'])):
                param_name = pso_data['param_names'][i]
                param_min = float(pso_data['lb'][i])
                param_max = float(pso_data['ub'][i])
                
                # Unique slider ID based on structure
                slider_id = f"param-slider-{i}-struct-{pso_data['structure_id']}"
                slider_title = f"{param_name} [{param_min:.3f}, {param_max:.3f}]"
                
                sliders.append(create_slider(slider_title, slider_id, param_min, param_max))
        
        # Objective sliders  
        if (pso_data['objectives'] is not None and 
            len(pso_data['objectives']) > 0 and 
            pso_data['objectives'].shape[1] > 0 and 
            len(pso_data['obj_mins']) > 0):
            
            sliders.append(html.H6("Objective Filters:", className="mt-3 mb-2"))
            
            for i in range(len(pso_data['obj_names'])):
                obj_name = pso_data['obj_names'][i]
                obj_min = float(pso_data['obj_mins'][i])
                obj_max = float(pso_data['obj_maxs'][i])
                
                # Unique slider ID based on structure
                slider_id = f"obj-slider-{i}-struct-{pso_data['structure_id']}"
                slider_title = f"{obj_name} [{obj_min:.3f}, {obj_max:.3f}]"
                
                sliders.append(create_slider(slider_title, slider_id, obj_min, obj_max))
        
        if len(sliders) <= 2:
            sliders.append(html.Div("No data available", style={'color': 'gray', 'fontStyle': 'italic'}))
            
    except Exception as e:
        sliders = [html.Div(f"Slider creation error: {str(e)}", style={'color': 'red'})]
        log_activity(f"Slider creation error: {str(e)}")
    
    return sliders

def filter_pareto_front(points):
    """Calculate Pareto front from points"""
    if len(points) == 0:
        return np.array([], dtype=bool)

    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i in range(points.shape[0]):
        for j in range(points.shape[0]):
            if i == j:
                continue
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

    if target_point_id is None or target_point_id >= len(displayed_objectives) or target_point_id < 0:
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
                        opacities.append(0.2)
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
                            "<b>Point #%{customdata}</b><br>"
                            + obj_names[j] + ": %{x:.3f}<br>"
                            + obj_names[i] + ": %{y:.3f}<br>"
                            + "<extra></extra>"
                        ),
                        showlegend=False,
                        selectedpoints=list(selected_indices) if selected_indices else None
                    ),
                    row=row, col=col
                )

                if fixed_axis_ranges and fixed_axis_ranges[0] is not None and fixed_axis_ranges[1] is not None:
                    if len(fixed_axis_ranges[0]) > j and len(fixed_axis_ranges[1]) > j:
                        fig.update_xaxes(range=[fixed_axis_ranges[0][j], fixed_axis_ranges[1][j]],
                                         title_text=obj_names[j], title_font=dict(size=11), row=row, col=col)
                    else:
                        fig.update_xaxes(title_text=obj_names[j], title_font=dict(size=11), row=row, col=col)

                    if len(fixed_axis_ranges[0]) > i and len(fixed_axis_ranges[1]) > i:
                        fig.update_yaxes(range=[fixed_axis_ranges[0][i], fixed_axis_ranges[1][i]],
                                         title_text=obj_names[i], title_font=dict(size=11), row=row, col=col)
                    else:
                        fig.update_yaxes(title_text=obj_names[i], title_font=dict(size=11), row=row, col=col)
                else:
                    fig.update_xaxes(title_text=obj_names[j], title_font=dict(size=11), row=row, col=col)
                    fig.update_yaxes(title_text=obj_names[i], title_font=dict(size=11), row=row, col=col)

    dynamic_height = calculate_plot_height(num_obj)
    
    fig.update_layout(
        title=dict(
            text=f'<b>{num_obj}×{num_obj} Interactive Multi-Objective Optimization Matrix</b>',
            font=dict(size=20, color='#2E4057'),
            x=0.5
        ),
        height=dynamic_height,
        showlegend=False,
        dragmode='select',
        selectdirection='d',
        margin=dict(l=60, r=60, t=100, b=60)
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

        html.Div(id='main-content-row', children=[
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='main-plot', config={'displayModeBar': True})
                ], width=9),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Filters", className="card-title"),
                            html.Div(id='slider-container', style={'overflowY': 'scroll', 'padding': '10px', 'border': '1px solid lightgray', 'borderRadius': '5px'})
                        ])
                    ])
                ], width=3)
            ], className="mb-4")
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
     Output('num-objectives', 'max'),
     Output('target-input', 'value')],
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
        return '', [], {'display': 'none'}, 0, 2, 0

    try:
        ctx = callback_context
        if not ctx.triggered:
            trigger = 'upload-data'
        else:
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]

        # Handle data modification actions
        if trigger in ['delete-selected-btn', 'keep-selected-btn', 'reset-data-btn']:
            if pso_data['objectives'] is not None:
                if trigger == 'delete-selected-btn' and pso_data['selected_indices']:
                    # Delete selected points
                    keep_mask = np.ones(len(pso_data['objectives']), dtype=bool)
                    for idx in pso_data['selected_indices']:
                        if 0 <= idx < len(pso_data['objectives']):
                            keep_mask[idx] = False
                    
                    if np.any(keep_mask):
                        pso_data['objectives'] = pso_data['objectives'][keep_mask]
                        if pso_data['parameters'] is not None and len(pso_data['parameters']) > 0:
                            pso_data['parameters'] = pso_data['parameters'][keep_mask]
                        
                        # Recalculate bounds
                        if len(pso_data['parameters']) > 0:
                            pso_data['lb'] = np.min(pso_data['parameters'], axis=0)
                            pso_data['ub'] = np.max(pso_data['parameters'], axis=0)
                        if len(pso_data['objectives']) > 0:
                            pso_data['obj_mins'] = np.min(pso_data['objectives'], axis=0)
                            pso_data['obj_maxs'] = np.max(pso_data['objectives'], axis=0)
                        
                        pso_data['selected_indices'] = set()
                        pso_data['structure_id'] += 1  # Force slider update
                        log_activity(f"Deleted selected points, {len(pso_data['objectives'])} remaining")
                
                elif trigger == 'keep-selected-btn' and pso_data['selected_indices']:
                    # Keep only selected points
                    valid_indices = [idx for idx in pso_data['selected_indices']
                                   if 0 <= idx < len(pso_data['objectives'])]
                    
                    if valid_indices:
                        pso_data['objectives'] = pso_data['objectives'][valid_indices]
                        if pso_data['parameters'] is not None and len(pso_data['parameters']) > 0:
                            pso_data['parameters'] = pso_data['parameters'][valid_indices]
                        
                        # Recalculate bounds
                        if len(pso_data['parameters']) > 0:
                            pso_data['lb'] = np.min(pso_data['parameters'], axis=0)
                            pso_data['ub'] = np.max(pso_data['parameters'], axis=0)
                        if len(pso_data['objectives']) > 0:
                            pso_data['obj_mins'] = np.min(pso_data['objectives'], axis=0)
                            pso_data['obj_maxs'] = np.max(pso_data['objectives'], axis=0)
                        
                        pso_data['selected_indices'] = set()
                        pso_data['structure_id'] += 1  # Force slider update
                        log_activity(f"Kept only selected points: {len(valid_indices)} remaining")
                
                elif trigger == 'reset-data-btn':
                    # Reset to original data
                    if pso_data['original_objectives'] is not None:
                        pso_data['objectives'] = pso_data['original_objectives'].copy()
                        if pso_data['original_parameters'] is not None:
                            pso_data['parameters'] = pso_data['original_parameters'].copy()
                        
                        # Recalculate bounds
                        if len(pso_data['parameters']) > 0:
                            pso_data['lb'] = np.min(pso_data['parameters'], axis=0)
                            pso_data['ub'] = np.max(pso_data['parameters'], axis=0)
                        if len(pso_data['objectives']) > 0:
                            pso_data['obj_mins'] = np.min(pso_data['objectives'], axis=0)
                            pso_data['obj_maxs'] = np.max(pso_data['objectives'], axis=0)
                        
                        pso_data['selected_indices'] = set()
                        pso_data['structure_id'] += 1  # Force slider update
                        log_activity("Reset to original data")
                
                # Create updated sliders
                sliders = create_sliders()
                
                file_info = dbc.Alert([
                    html.H6(f"File: {pso_data['filename']}", className="alert-heading"),
                    html.P([
                        f"Rows: {len(pso_data['objectives'])} | ",
                        f"Parameters: {len(pso_data['param_names'])} | ",
                        f"Objectives: {len(pso_data['obj_names'])} | ",
                        f"Pareto Points: {np.sum(filter_pareto_front(pso_data['objectives'])) if len(pso_data['objectives']) > 0 else 0}"
                    ], className="mb-0")
                ], color="success")
                
                return file_info, sliders, {'display': 'block'}, max(0, len(pso_data['objectives']) - 1), total_columns, 0

    except Exception as e:
        error_msg = dbc.Alert(f"Error: {str(e)}", color="danger")
        log_activity(f"Error in load_csv_and_process: {str(e)}")
        return error_msg, [], {'display': 'none'}, 0, 2, 0

@app.callback(
    Output('slider-container', 'style'),
    [Input('main-plot', 'figure')],
    prevent_initial_call=True
)
def update_slider_height(figure):
    """Update slider container height to match plot height"""
    try:
        if figure and 'layout' in figure and 'height' in figure['layout']:
            plot_height = figure['layout']['height']
            return {
                'height': f'{plot_height}px',
                'overflowY': 'scroll', 
                'padding': '10px', 
                'border': '1px solid lightgray', 
                'borderRadius': '5px'
            }
        else:
            return {
                'height': '400px',
                'overflowY': 'scroll', 
                'padding': '10px', 
                'border': '1px solid lightgray', 
                'borderRadius': '5px'
            }
    except Exception:
        return {
            'height': '400px',
            'overflowY': 'scroll', 
            'padding': '10px', 
            'border': '1px solid lightgray', 
            'borderRadius': '5px'
        }

@app.callback(
    [Output('main-plot', 'figure'),
     Output('status-display', 'children'),
     Output('activity-panel', 'children'),
     Output('activity-log', 'children')],
    [Input('upload-data', 'contents'),
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
def update_visualization(contents, target_id, selection_store, selected_data, click_data,
                        clear_clicks, delete_clicks, keep_clicks, reset_clicks, obj_selection_clicks,
                        current_selection_store):

    try:
        if pso_data['objectives'] is None or len(pso_data['objectives']) == 0:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="Upload a CSV file to begin", height=400)
            return empty_fig, "No data loaded", "Upload CSV file", []

        ctx = callback_context
        if not ctx.triggered:
            trigger = 'upload-data'
            triggered_id = None
        else:
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]
            triggered_id = ctx.triggered[0]['prop_id']

        displayed_objectives, displayed_names = get_displayed_objectives()

        if displayed_objectives is None or len(displayed_objectives) == 0:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No data available", height=400)
            return empty_fig, "No data", "No data", []

        # Handle plot interactions
        if triggered_id == 'main-plot.selectedData' and selected_data and selected_data.get('points'):
            new_selection = set()
            for point in selected_data['points']:
                if 'customdata' in point and isinstance(point['customdata'], int):
                    idx = point['customdata']
                    if 0 <= idx < len(pso_data['objectives']):
                        new_selection.add(idx)
            if new_selection != pso_data['selected_indices']:
                pso_data['selected_indices'] = new_selection
                pso_data['current_clicked_point'] = None
                log_activity(f"Selected {len(new_selection)} points via drag selection")

        elif triggered_id == 'main-plot.clickData' and click_data and click_data.get('points'):
            clicked_point = click_data['points'][0]
            if 'customdata' in clicked_point and isinstance(clicked_point['customdata'], int):
                point_id = clicked_point['customdata']
                if 0 <= point_id < len(pso_data['objectives']):
                    if point_id in pso_data['selected_indices']:
                        pso_data['selected_indices'].remove(point_id)
                        log_activity(f"Deselected point #{point_id}")
                    else:
                        if len(pso_data['selected_indices']) > 1:
                            pso_data['selected_indices'] = set()
                            log_activity("Cleared multi-selection on single click")
                        pso_data['selected_indices'].add(point_id)
                        log_activity(f"Selected point #{point_id}")
                    pso_data['current_clicked_point'] = point_id

        # Validate selected indices
        valid_selected = {idx for idx in pso_data['selected_indices']
                         if isinstance(idx, int) and 0 <= idx < len(displayed_objectives)}
        pso_data['selected_indices'] = valid_selected

        # Handle button clicks
        if 'clear-selection-btn' in trigger and clear_clicks:
            pso_data['selected_indices'] = set()
            pso_data['current_clicked_point'] = None
            log_activity("Cleared selection")

        # Validate target ID
        current_data_length = len(displayed_objectives)
        if target_id is None or target_id >= current_data_length or target_id < 0:
            target_id = 0

        current_objectives = displayed_objectives.copy()
        current_parameters = (pso_data['parameters'].copy()
                            if pso_data['parameters'] is not None and len(pso_data['parameters']) > 0
                            else None)

        # Create filter mask (no slider filtering for now since we use simple string IDs)
        filter_mask = np.ones(len(current_objectives), dtype=bool)

        # Create visualization
        try:
            fig = create_interactive_scatter_matrix(
                current_objectives,
                pso_data.get('pareto_objectives', np.array([])),
                target_id,
                pso_data['selected_indices'],
                (pso_data['obj_mins'], pso_data['obj_maxs']) if 'obj_mins' in pso_data and pso_data['obj_mins'].size > 0 else None,
                filter_mask
            )
        except Exception as plot_error:
            fig = go.Figure()
            fig.update_layout(title=f"Visualization Error: {str(plot_error)}", height=400)
            log_activity(f"Plot error: {str(plot_error)}")

        # Create status display
        try:
            pareto_count = (np.sum(filter_pareto_front(current_objectives))
                          if len(current_objectives) > 0 else 0)

            status_content = dbc.Row([
                dbc.Col(html.Strong(f"Total: {len(current_objectives)}"), width=2),
                dbc.Col(html.Strong(f"Pareto: {pareto_count}", style={'color': 'blue'}), width=2),
                dbc.Col(html.Strong(f"Selected: {len(pso_data['selected_indices'])}", style={'color': 'red'}), width=2),
                dbc.Col(html.Strong(f"Target: #{target_id}", style={'color': 'darkred'}), width=3),
                dbc.Col(html.Strong(f"Objectives: {current_objectives.shape[1]}", style={'color': 'green'}), width=3)
            ])
        except Exception:
            status_content = "Status update error"

        # Create point information display
        try:
            activity_content = "No point information available"

            display_point_id = None
            display_type = "none"

            if len(pso_data['selected_indices']) > 1:
                display_type = "multiple_selected"
            elif (pso_data['current_clicked_point'] is not None and
                  pso_data['current_clicked_point'] < len(current_objectives) and
                  pso_data['current_clicked_point'] in pso_data['selected_indices']):
                display_point_id = pso_data['current_clicked_point']
                display_type = "clicked"
            elif len(pso_data['selected_indices']) == 1:
                display_point_id = list(pso_data['selected_indices'])[0]
                display_type = "single_selected"
            elif target_id < len(current_objectives):
                display_point_id = target_id
                display_type = "target"

            if display_type == "multiple_selected":
                selected_indices_list = list(pso_data['selected_indices'])
                selected_objectives = current_objectives[selected_indices_list]
                num_selected = len(selected_indices_list)

                pareto_mask_selected = filter_pareto_front(selected_objectives)
                num_pareto_selected = np.sum(pareto_mask_selected)

                ideal_point = np.min(current_objectives, axis=0)
                avg_objectives = np.mean(selected_objectives, axis=0)
                ideal_distance = np.sqrt(np.sum((avg_objectives - ideal_point)**2))

                obj_names = pso_data.get('obj_names', [f'Obj_{i}' for i in range(selected_objectives.shape[1])])
                
                obj_display = html.Div([
                    html.P("Average Objective Values:", style={'fontWeight': 'bold', 'fontSize': '12px', 'margin': '2px 0'}),
                    html.P(" | ".join([f"{obj_names[i] if i < len(obj_names) else f'Obj_{i}'}: {avg_val:.4f}"
                                     for i, avg_val in enumerate(avg_objectives)]),
                           style={'fontSize': '11px', 'margin': '2px 0'})
                ])

                param_display = html.Div()
                if current_parameters is not None and len(current_parameters) > 0:
                    selected_parameters = current_parameters[selected_indices_list]
                    param_names = pso_data.get('param_names', [f'Param_{i}' for i in range(selected_parameters.shape[1])])
                    avg_parameters = np.mean(selected_parameters, axis=0)
                    
                    param_display = html.Div([
                        html.P("Average Parameter Values:", style={'fontWeight': 'bold', 'fontSize': '12px', 'margin': '2px 0'}),
                        html.P(" | ".join([f"{param_names[i] if i < len(param_names) else f'Param_{i}'}: {avg_val:.4f}"
                                         for i, avg_val in enumerate(avg_parameters)]),
                               style={'fontSize': '11px', 'margin': '2px 0'})
                    ])

                activity_content = html.Div([
                    html.P(f"{num_selected} Points Selected | {num_pareto_selected} Pareto Optimal",
                           style={'fontWeight': 'bold', 'color': 'darkblue'}),
                    html.Hr(style={'margin': '5px 0'}),
                    obj_display,
                    param_display,
                    html.P(f"Ideal Distance: {ideal_distance:.4f}",
                           style={'fontSize': '11px', 'fontWeight': 'bold', 'color': 'black'})
                ])

            elif display_point_id is not None:
                obj_values = current_objectives[display_point_id]
                param_values = (current_parameters[display_point_id]
                              if current_parameters is not None and len(current_parameters) > 0
                              else None)
                is_pareto = (filter_pareto_front(current_objectives)[display_point_id]
                           if len(current_objectives) > 0 else False)

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
                else:
                    point_label = f"Target Point #{display_point_id}"
                    color = 'darkred'

                obj_display = html.Div([
                    html.P("Objective Values:", style={'fontWeight': 'bold', 'fontSize': '12px', 'margin': '2px 0'}),
                    html.P(" | ".join([f"{obj_names[i] if i < len(obj_names) else f'Obj_{i}'}: {val:.4f}"
                                     for i, val in enumerate(obj_values)]),
                           style={'fontSize': '11px', 'margin': '2px 0'})
                ])

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
                    html.P(f"Ideal Distance: {ideal_distance:.4f}",
                           style={'fontSize': '11px', 'fontWeight': 'bold', 'color': 'black'})
                ])

        except Exception as activity_error:
            activity_content = f"Point information error: {str(activity_error)}"

        # Create activity log
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
        error_fig.update_layout(title=f"Callback Error: {str(e)}", height=400)
        error_status = f"Error: {str(e)}"
        error_activity = f"Callback failed: {str(e)}"
        error_log = [html.Div(f"Critical error: {str(e)}")]

        try:
            log_activity(f"Critical callback error: {str(e)}")
        except:
            pass

        return error_fig, error_status, error_activity, error_log

@app.callback(
    Output('slider-container', 'children', allow_duplicate=True),
    [Input('reset-sliders-btn', 'n_clicks')],
    prevent_initial_call=True
)
def update_sliders_on_reset(reset_clicks):
    """Update sliders when reset button is clicked"""
    try:
        ctx = callback_context
        if not ctx.triggered:
            return create_sliders()

        trigger = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger == 'reset-sliders-btn':
            log_activity("Reset all sliders to default ranges")
            pso_data['structure_id'] += 1  # Force recreation

        return create_sliders()
    except Exception as e:
        return create_sliders()

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
<<<<<<< HEAD
    app.run(host="localhost", port=8080, debug=True) 1), len(pso_data['param_names']) + len(pso_data['obj_names']), 0
            else:
                return '', [], {'display': 'none'}, 0, 2, 0

        # Handle file upload or structure change
        if trigger == 'upload-data' or trigger == 'apply-obj-selection-btn':
            if trigger == 'upload-data':
                # Decode CSV
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                pso_data['original_df'] = df.copy()
                pso_data['filename'] = filename
                log_activity(f"Uploaded: {filename}")
            else:
                # Use stored dataframe
                if 'original_df' not in pso_data or pso_data['original_df'] is None:
                    raise ValueError("No data to restructure")
                df = pso_data['original_df'].copy()
                log_activity(f"Restructuring with {num_objectives_input} objectives")

            all_columns = df.columns.tolist()
            total_columns = len(all_columns)

            if num_objectives_input is None or num_objectives_input < 1:
                num_objectives_input = min(2, total_columns)

            # Update data structure
            success, error = update_data_structure(df, num_objectives_input)
            if not success:
                raise ValueError(error)

            # Create sliders with new structure
            sliders = create_sliders()

            file_info = dbc.Alert([
                html.H6(f"File: {pso_data['filename']}", className="alert-heading"),
                html.P([
                    f"Rows: {len(df)} | ",
                    f"Total Columns: {total_columns} | ",
                    f"Parameters: {len(pso_data['param_names'])} ({pso_data['param_names']}) | ",
                    f"Objectives: {len(pso_data['obj_names'])} ({pso_data['obj_names']}) | ",
                    f"Pareto Points: {len(pso_data['pareto_objectives'])}"
                ], className="mb-0")
            ], color="success")

            return file_info, sliders, {'display': 'block'}, max(0, len(pso_data['objectives']) -
=======
    app.run(host="0.0.0.0", port=8080, debug=True)
>>>>>>> ef2ea61b77d43f5ad27ecc664b478092413f31df
