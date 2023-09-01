import base64
import os

# TODO: Have this generated somewhere else/loaded from a file
fields = {
    "ELUC":{"data_type":"FLOAT","has_nan":False,"mean":0.08431588113307953,"range":[-88.90611267089844,116.95401763916016],"std_dev":0.7141819000244141,"sum":1244851,"valued":"CONTINUOUS"},
    "c3ann":{"data_type":"FLOAT","has_nan":False,"mean":0.05719335377216339,"range":[0,0.9272090196609497],"std_dev":0.13004545867443085,"sum":844410.375,"valued":"CONTINUOUS"},
    "c3ann_diff":{"data_type":"FLOAT","has_nan":False,"mean":0.0001039158960338682,"range":[-1,1],"std_dev":0.003100739326328039,"sum":1534.228271484375,"valued":"CONTINUOUS"},
    "c3nfx":{"data_type":"FLOAT","has_nan":False,"mean":0.01243751309812069,"range":[0,0.6590129733085632],"std_dev":0.04110949859023094,"sum":183629.125,"valued":"CONTINUOUS"},
    "c3nfx_diff":{"data_type":"FLOAT","has_nan":False,"mean":0.000047470879508182406,"range":[-1,1],"std_dev":0.0009231835720129311,"sum":700.866455078125,"valued":"CONTINUOUS"},
    "c3per":{"data_type":"FLOAT","has_nan":False,"mean":0.0064199501648545265,"range":[0,0.6860707998275757],"std_dev":0.025114575400948524,"sum":94785.0078125,"valued":"CONTINUOUS"},
    "c3per_diff":{"data_type":"FLOAT","has_nan":False,"mean":0.000037744077417301014,"range":[-1,1],"std_dev":0.0007734607788734138,"sum":557.2586669921875,"valued":"CONTINUOUS"},
    "c4ann":{"data_type":"FLOAT","has_nan":False,"mean":0.01571018248796463,"range":[0,0.9358039498329163],"std_dev":0.04956522956490517,"sum":231947.265625,"valued":"CONTINUOUS"},
    "c4ann_diff":{"data_type":"FLOAT","has_nan":False,"mean":0.0000682401514495723,"range":[-1,1],"std_dev":0.0016709286719560623,"sum":1007.5067749023438,"valued":"CONTINUOUS"},
    "c4per":{"data_type":"FLOAT","has_nan":False,"mean":0.0009445593459531665,"range":[0,0.7032631039619446],"std_dev":0.008503105491399765,"sum":13945.6015625,"valued":"CONTINUOUS"},
    "c4per_diff":{"data_type":"FLOAT","has_nan":False,"mean":0.000008784978490439244,"range":[-1,1],"std_dev":0.0002543762675486505,"sum":129.70260620117188,"valued":"CONTINUOUS"},
    "cell_area":{"data_type":"FLOAT","has_nan":False,"mean":54771.609375,"range":[8915.4794921875,77276.703125],"std_dev":18437.73046875,"sum":808655454208,"valued":"CONTINUOUS"},
    "change":{"data_type":"FLOAT","has_nan":False,"mean":0.5,"range":[0,1],"std_dev":0.1,"sum":7382067,"valued":"CONTINUOUS"},
    "pastr":{"data_type":"FLOAT","has_nan":False,"mean":0.04077955335378647,"range":[0,1],"std_dev":0.10672948509454727,"sum":602074.8125,"valued":"CONTINUOUS"},
    "pastr_diff":{"data_type":"FLOAT","has_nan":False,"mean":0.00026207335758954287,"range":[-1,1],"std_dev":0.005082341376692057,"sum":3869.28662109375,"valued":"CONTINUOUS"},
    "primf":{"data_type":"FLOAT","has_nan":False,"mean":0.19610066711902618,"range":[0,1],"std_dev":0.35063520073890686,"sum":2895256.75,"valued":"CONTINUOUS"},
    "primf_diff":{"data_type":"FLOAT","has_nan":False,"mean":-0.0009334315545856953,"range":[-0.850843608379364,0],"std_dev":0.004068289417773485,"sum":-13781.3095703125,"valued":"CONTINUOUS"},
    "primn":{"data_type":"FLOAT","has_nan":False,"mean":0.2566087543964386,"range":[0,1],"std_dev":0.3646445870399475,"sum":3788606.5,"valued":"CONTINUOUS"},
    "primn_diff":{"data_type":"FLOAT","has_nan":False,"mean":-0.001117548905313015,"range":[-0.936556875705719,0],"std_dev":0.005212769843637943,"sum":-16499.642578125,"valued":"CONTINUOUS"},
    "range":{"data_type":"FLOAT","has_nan":False,"mean":0.15799088776111603,"range":[0,1],"std_dev":0.28534045815467834,"sum":2332598.75,"valued":"CONTINUOUS"},
    "range_diff":{"data_type":"FLOAT","has_nan":False,"mean":0.00040018701110966504,"range":[-1,1],"std_dev":0.011220048181712627,"sum":5908.4150390625,"valued":"CONTINUOUS"},
    "secdf":{"data_type":"FLOAT","has_nan":False,"mean":0.10117984563112259,"range":[0,1],"std_dev":0.2359693944454193,"sum":1493832.875,"valued":"CONTINUOUS"},
    "secdf_diff":{"data_type":"FLOAT","has_nan":False,"mean":0.0006275310879573226,"range":[-1,1],"std_dev":0.004725911188870668,"sum":9264.9541015625,"valued":"CONTINUOUS"},
    "secdn":{"data_type":"FLOAT","has_nan":False,"mean":0.08007288724184036,"range":[0,1],"std_dev":0.18958471715450287,"sum":1182206.875,"valued":"CONTINUOUS"},
    "secdn_diff":{"data_type":"FLOAT","has_nan":False,"mean":0.0004467177495826036,"range":[-1,1],"std_dev":0.009596005082130432,"sum":6595.4013671875,"valued":"CONTINUOUS"},
    "urban":{"data_type":"FLOAT","has_nan":False,"mean":0.0025856471620500088,"range":[0,1],"std_dev":0.021832581609487534,"sum":38174.84375,"valued":"CONTINUOUS"},
    "urban_diff":{"data_type":"FLOAT","has_nan":False,"mean":0.00004831538171856664,"range":[-0.15093612670898438,0.1676577627658844],"std_dev":0.0006846132455393672,"sum":713.3348388671875,"valued":"CONTINUOUS"}
}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(ROOT_DIR, "../data/processed/eluc_1982.csv")

GRID_STEP = 0.25

INDEX_COLS = ["time", "lat", "lon"]

LAND_USE_COLS = ['c3ann', 'c3nfx', 'c3per', 'c4ann', 'c4per', 'pastr', 'primf', 'primn', 'range', 'secdf', 'secdn', 'urban']
CONTEXT_COLUMNS = LAND_USE_COLS + ['cell_area']
DIFF_LAND_USE_COLS = [f"{col}_diff" for col in LAND_USE_COLS]
COLS_MAP = dict(zip(LAND_USE_COLS, DIFF_LAND_USE_COLS))

# Prescriptor outputs
RECO_COLS = ['c3ann', 'c3nfx', 'c3per','c4ann', 'c4per', 'pastr', 'range', 'secdf', 'secdn']
DIFF_RECO_COLS = [f"{col}_diff" for col in RECO_COLS]
RECO_MAP = dict(zip(RECO_COLS, DIFF_RECO_COLS))

NO_CHANGE_COLS = ["primf", "primn", "urban"]
CHART_COLS = LAND_USE_COLS + ["nonland"]

SLIDER_PRECISION = 1e-5

# Tonnes of CO2 per person for a flight from JFK to Geneva
CO2_JFK_GVA = 2.2
CO2_PERSON = 4

# For creating treemap
C3 = ['c3ann', 'c3nfx', 'c3per']
C4 = ['c4ann', 'c4per']
PRIMARY = ['primf', 'primn']
SECONDARY = ['secdf', 'secdn']
FIELDS = ['pastr', 'range']

CHART_TYPES = ["Treemap", "Pie Chart"]

# Pareto front
PARETO_CSV_PATH = os.path.join(ROOT_DIR, "../prescriptors/pareto.csv")
PARETO_FRONT_PATH = os.path.join(ROOT_DIR, "../prescriptors/pareto_front.png")
PARETO_FRONT = base64.b64encode(open(PARETO_FRONT_PATH, 'rb').read()).decode('ascii')

PREDICTOR_PATH = os.path.join(ROOT_DIR, "../predictors/")
PRESCRIPTOR_PATH = os.path.join(ROOT_DIR, "../prescriptors/")
DEFAULT_PRESCRIPTOR_IDX = 3  # By default we select the fourth prescriptor that minimizes change
