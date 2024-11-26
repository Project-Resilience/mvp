from sklearn.metrics import mean_absolute_error

import data.constants as constants
from data.min_eluc_data import MinimalELUCData

from prsdk.data.torch_data import TorchDataset

from prsdk.persistence.persistors.hf_persistor import HuggingFacePersistor
from prsdk.persistence.serializers.neural_network_serializer import NeuralNetSerializer

if __name__ =="__main__":
    pred_persistor = HuggingFacePersistor(NeuralNetSerializer())
    nnp = pred_persistor.from_pretrained("danyoung/eluc-global-nn",
                                         local_dir="predictors/trained_models/danyoung--eluc-global-nn")
    
    dataset = MinimalELUCData.from_hf()

    test_df = dataset.test_df.sample(frac=0.1, random_state=42)

    preds = nnp.predict(test_df)
    mae = mean_absolute_error(test_df["ELUC"], preds)
    print(mae)
