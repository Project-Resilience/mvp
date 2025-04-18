"""
Main entrypoint to run the app. Contains the layout of the app and registers all the callbacks of each component.
"""
from dash import Dash, html
import dash_bootstrap_components as dbc
import pandas as pd

import app.constants as app_constants
from app.components.intro import IntroComponent
from app.components.context.context import ContextComponent
from app.components.filter import FilterComponent
from app.components.dms.dms import DMSComponent
from app.components.references import ReferencesComponent
from app.utils import EvolutionHandler


app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
           prevent_initial_callbacks="initial_duplicate")
server = app.server

app.title = 'Land Use Optimization'

app_df = pd.read_csv("app/data/app_data.csv", index_col=app_constants.INDEX_COLS)

evolution_handler = EvolutionHandler()

intro_component = IntroComponent()
context_component = ContextComponent(app_df, evolution_handler)
filter_component = FilterComponent(evolution_handler)
dms_component = DMSComponent(app_df, evolution_handler)
references_component = ReferencesComponent()

context_component.register_callbacks(app)
filter_component.register_callbacks(app)
dms_component.register_callbacks(app)

app.layout = html.Div(
    children=[
        intro_component.get_div(),
        context_component.get_div(),
        filter_component.get_div(),
        dms_component.get_div(),
        references_component.get_references_div()
    ]
)

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=False, port=4057, use_reloader=False, threaded=True)
