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
        self.encoder_model = tf.keras.models.load_model(model_path[0])
        self.decoder_model = tf.keras.models.load_model(model_path[1])

    async def __call__(self, starlette_request):
        const = Config()
        # Step 1: transform HTTP request -> tensorflow input
        # Here we define the request schema to be a json array.
        input_array_encoder = np.array((await starlette_request.json())["array"][0])
        input_array_decoder = np.array((await starlette_request.json())["array"][1])
        #print(f'Input array: {input_array}')

        # Step 2: tensorflow input -> tensorflow output
        encoder_prediction = self.encoder_model(input_array_encoder)
        decoder_prediction = self.decoder_model(input_array_decoder)

        # Step 3: tensorflow output -> web output
        return {"predictions": [encoder_prediction.numpy().tolist(), decoder_prediction.numpy().tolist()], "file": self.model_path}
    

if __name__ == "__main__":
    serve.start()
    TFEncoderDecoderModel.deploy(['./saved_models/encoder','./saved_models/decoder'])
   
    
    resp = requests.get(
        "http://localhost:8000/saved_models", 
        json={"array": [[[0.22, 1.0, 1.0, 1.0, 0.0,\
            0.12, 0.28, 0.24, 7.0e-05, 0.0, 0.00, 0.57, \
            0.26, 0.23, 0.45, 0.0, 0.98, 0.54, 0.36, \
            6.33e-11, 0.28, 0.57, 1.0, 0.01, 0.01]], [.5]]
            } #np.random.uniform(0,1,1).tolist()
    )
    
    #res = pd.DataFrame(resp.json()['prediction'])
    print(resp.json()['predictions'][0][0], resp.json()['predictions'][1][0])