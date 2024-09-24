from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd

import app.constants as app_constants
from app.newcomponents.context import ContextComponent
from app.newcomponents.filter import FilterComponent
from app.utils import EvolutionHandler
from data import constants


app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
           prevent_initial_callbacks="initial_duplicate")
server = app.server

app.title = 'Land Use Optimization'

app_df = pd.read_csv(app_constants.DATA_FILE_PATH, index_col=app_constants.INDEX_COLS)
evolution_handler = EvolutionHandler()

context_component = ContextComponent(app_df, evolution_handler)
filter_component = FilterComponent(evolution_handler)

context_component.register_callbacks(app)
filter_component.register_callbacks(app)

app.layout = html.Div([context_component.get_div(), filter_component.get_div()])

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=False, port=4057, use_reloader=True, threaded=False)
