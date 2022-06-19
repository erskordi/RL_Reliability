from ray import serve

import os
import json
import tempfile
import numpy as np
import pandas as pd
import requests

from config import Config

##### Serve saved models
@serve.deployment(route_prefix="/saved_models")
class TFEncoderDecoderModel:
    def __init__(self, model_path):
        import tensorflow as tf

        self.model_path = model_path
        self.encoder_model = tf.keras.models.load_model(model_path[0], compile=False)
        self.decoder_model = tf.keras.models.load_model(model_path[1], compile=False)
        print(self.encoder_model.summary(), self.decoder_model.summary())

    async def __call__(self, starlette_request):
        const = Config()
        # Step 1: transform HTTP request -> tensorflow input
        # Here we define the request schema to be a json array.
        input_array_encoder = np.array((await starlette_request.json())["array"][0])
        input_array_decoder = np.array((await starlette_request.json())["array"][1])
        #print(f'Input array dec: {input_array_decoder}')

        # Step 2: tensorflow input -> tensorflow output
        encoder_prediction = self.encoder_model(input_array_encoder)
        decoder_prediction = self.decoder_model(input_array_decoder)

        # Step 3: tensorflow output -> web output
        return {"predictions": [encoder_prediction.numpy().tolist(), decoder_prediction.numpy().tolist()], "file": self.model_path}
    

if __name__ == "__main__":

    from data_prep import DataPrep

    serve.start()
    TFEncoderDecoderModel.deploy(['./saved_models/encoder','./saved_models/decoder'])

    const = Config()
    neurons = const.VAE_neurons

    # Data prep
    data = DataPrep(file = const.file_path,
                    num_settings = const.num_settings,
                    num_sensors = const.num_sensors,
                    num_units = const.num_units[1],
                    prev_step_units = const.prev_step_units[1],
                    step = const.step[1],
                    normalization_type="01")

    df = data.ReadData()

   
    for t in range(len(df)):
        resp = requests.get(
            "http://localhost:8000/saved_models", 
            json={"array": [[df.iloc[t,1:].tolist()], [np.random.uniform(0,1,2).tolist()]]
                } #np.random.uniform(0,1,1).tolist()
        )
    
    #res = pd.DataFrame(resp.json()['prediction'])
    print(resp.json()['predictions'][0][0], resp.json()['predictions'][1][0])