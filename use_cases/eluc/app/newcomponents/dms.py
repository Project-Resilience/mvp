
import pandas as pd


from app.components.chart import ChartComponent
from app.components.lock import LockComponent
from app.components.prediction import PredictionComponent
from app.components.sliders import SlidersComponent

class DMSComponent():
    def __init__(self, app_df: pd.DataFrame):
        self.chart_component = ChartComponent(app_df)
        self.lock_component = LockComponent()
        self.prediction_component = PredictionComponent(app_df)
        self.sliders_component = SlidersComponent(app_df)