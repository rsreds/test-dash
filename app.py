from dash import Dash, html

app = Dash(__name__)
app.layout = html.Div("Hello from Dash deployed on OpenShift!")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
