import pickle
import pandas as pd

with open("random_forest_model.pkl", "rb") as model_file:
    loaded_rf_model = pickle.load(model_file)


user_input = {
    "trans_hour": int(input("Enter trans_hour: ")),
    "trans_day": int(input("Enter trans_day: ")),
    "trans_month": int(input("Enter trans_month: ")),
    "trans_year": int(input("Enter trans_year: ")),
    "category": int(input("Enter category: ")),  # 4
    "upi_number": int(input("Enter upi_number: ")),
    "age": int(input("Enter age: ")),
    "trans_amount": float(input("Enter trans_amount: ")),
}

user_df = pd.DataFrame([user_input])
prediction = loaded_rf_model.predict(user_df)


randomf_prediction = prediction
print(f"Predicted Fraud Risk By Random Forest (10000): {prediction[0]}")


# CNN model

import pickle
from keras.models import load_model
import pandas as pd

loaded_model = load_model("CNN_Model.h5")
with open("scaler.pkl", "rb") as scaler_file:
    loaded_scaler = pickle.load(scaler_file)
user_df = pd.DataFrame([user_input])
user_input_scaled = loaded_scaler.transform(user_df)
loaded_prediction_prob = loaded_model.predict(user_input_scaled)
loaded_prediction_binary = (loaded_prediction_prob > 0.5).astype(int)

print(f"Predicted Fraud Probability (CNN Model): {loaded_prediction_prob[0]}")
print(f"Predicted Fraud Risk (Binary) (CNN Model): {loaded_prediction_binary[0]}")
cnn_predction = loaded_prediction_prob[0]


###RNN
import pickle
from keras.models import load_model
import pandas as pd

loaded_model = load_model("RNN_Model.h5")
with open("scaler_rnn.pkl", "rb") as scaler_file:
    loaded_scaler_rnn = pickle.load(scaler_file)

user_df = pd.DataFrame([user_input])
user_input_scaled = loaded_scaler_rnn.transform(user_df)
user_input_reshaped = user_input_scaled.reshape(
    (user_input_scaled.shape[0], 1, user_input_scaled.shape[1])
)
loaded_prediction_prob = loaded_model.predict(user_input_reshaped)
loaded_prediction_binary = (loaded_prediction_prob > 0.5).astype(int)
print(f"Predicted Fraud Probability (RNN Model): {loaded_prediction_prob[0]}")
print(f"Predicted Fraud Risk (Binary) (RNN Model): {loaded_prediction_binary[0]}")
rnn_predction = loaded_prediction_prob[0]
print("-------------------------------------------------------------------")
valid = []
invalid = []
if randomf_prediction == 1:
    invalid.append(1)
else:
    valid.append(0)

if cnn_predction >= 0.5:
    invalid.append(cnn_predction)
else:
    valid.append(cnn_predction)

if rnn_predction >= 0.5:
    invalid.append(rnn_predction)
else:
    valid.append(rnn_predction)

print(valid, invalid)
if len(valid) > len(invalid):
    print("The probabiity of the Transaction being Fradu is", sum(valid) / len(valid))
    print(" Possibility Genuine UPI Tranaction ")
else:
    print(
        "The probabiity of the Transaction being Fradu is", sum(invalid) / len(invalid)
    )
    print(" Possibility Fake UPI Tranaction ")
