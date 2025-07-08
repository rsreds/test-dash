from dash import Dash, html, dcc, Input, Output, State, ctx, dash_table
import pandas as pd
import os

app = Dash(__name__)

BASE_DIR = "/eos/home-o/"

app.layout = html.Div([
    html.H1("EOS Browser & CSV Uploader"),

    html.Div([
        html.H2("📁 Browse EOS"),
        html.P(f"Base directory: {BASE_DIR}"),
        dcc.Dropdown(id="folder-dropdown", placeholder="Select a folder"),
        html.Button("Go", id="go-button"),
        html.Div(id="eos-file-list"),
    ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),

    html.Div([
        html.H2("⬆️ Upload CSV"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=False,
        ),
        html.Div(id="upload-output"),
    ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
])

@app.callback(
    Output("folder-dropdown", "options"),
    Output("folder-dropdown", "value"),
    Input("go-button", "n_clicks"),
    State("folder-dropdown", "value"),
    prevent_initial_call=True
)
def update_dropdown(n_clicks, selected_folder):
    current_dir = selected_folder if selected_folder else BASE_DIR
    try:
        entries = os.listdir(current_dir)
        dirs = []
        for entry in sorted(entries):
            full_path = os.path.join(current_dir, entry)
            if os.path.isdir(full_path):
                try:
                    os.listdir(full_path)
                    dirs.append({"label": entry, "value": full_path})
                except PermissionError:
                    dirs.append({"label": f"{entry} (no permission)", "value": None, "disabled": True})
        return dirs, current_dir
    except PermissionError:
        return [], BASE_DIR
    except Exception:
        return [], BASE_DIR

@app.callback(
    Output("eos-file-list", "children"),
    Input("folder-dropdown", "value")
)
def show_eos_files(selected_folder):
    if selected_folder:
        try:
            entries = os.listdir(selected_folder)
            items = []
            for entry in sorted(entries):
                path = os.path.join(selected_folder, entry)
                if os.path.isfile(path) and entry.endswith(".csv"):
                    items.append(html.Button(entry, id={"type": "eos-file", "name": path}))
                elif os.path.isdir(path):
                    items.append(html.Div(f"📁 {entry}/"))
            if not items:
                return html.P("This folder is empty or has no CSV files.")
            return html.Div(items)
        except PermissionError:
            return html.P("❌ Permission denied for this folder")
        except Exception as e:
            return html.P(f"⚠️ Error: {e}")
    return html.P("📁 Select a folder to browse")

from dash.dependencies import ALL

@app.callback(
    Output("upload-output", "children"),
    Input({"type": "eos-file", "name": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def load_eos_csv(n_clicks):
    triggered_id = ctx.triggered_id
    if triggered_id and "name" in triggered_id:
        file_path = triggered_id["name"]
        try:
            df = pd.read_csv(file_path)
            return dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df.columns],
                page_size=10,
                style_table={"overflowX": "auto"},
            )
        except Exception as e:
            return html.P(f"⚠️ Failed to load CSV: {e}")
    return html.P("No file selected.")

@app.callback(
    Output("upload-output", "children", allow_duplicate=True),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True
)
def upload_csv(contents, filename):
    if contents:
        try:
            content_type, content_string = contents.split(",")
            import base64
            import io
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df.columns],
                page_size=10,
                style_table={"overflowX": "auto"},
            )
        except Exception as e:
            return html.P(f"⚠️ Error reading uploaded file: {e}")
    return html.P("No file uploaded.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
