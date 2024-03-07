from flask import Flask, render_template, url_for, request
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import pickle
import pandas as pd
from keras.models import load_model
import smtplib

app = Flask(__name__, static_url_path="/static")

# print(app)


# index page
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    return render_template("predict.html")


@app.route("/result", methods=["POST", "GET"])
def result():
    print(request.method)

    if request.method == "POST":
        with open("random_forest_model.pkl", "rb") as model_file:
            loaded_rf_model = pickle.load(model_file)

        prob = 0.0000
        ans = False
        print(request.form)
        # print(request.args.get("email"))
        # email = request.form.get(str("email"))
        # print(email)
        email = request.form["email"]
        trans = request.form["trans_id"]
        amount = request.form["amount"]
        date = request.form["date"]
        time = request.form["time"]
        send_upi = request.form["send_upi"]
        rec_upi = request.form["rec_upi"]
        s_num = request.form["s_num"]
        r_num = request.form["r_num"]
        age = request.form["age"]
        cat = 4

        print(
            "---------------------------------------------------------------------------------------"
        )
        print(
            email, trans, amount, date, time, send_upi, rec_upi, s_num, r_num, age, cat
        )
        # trans = int(trans)  # pt
        amount = int(amount)
        from datetime import datetime

        date_string = date
        date_object = datetime.strptime(date_string, "%Y-%m-%d")
        day = date_object.day
        month = date_object.month
        year = date_object.year

        # time_string_2 = time
        # time_object_1 = datetime.strptime(time_string_2, "%H:%M")
        # hour = time_object_1.hour
        time_string_2 = time
        time_object_24hr = datetime.strptime(time_string_2, "%H:%M")
        time_12hr_format = time_object_24hr.strftime("%I %p")
        hour = int(time_12hr_format.split()[0])
        print(hour)
        sender_num = int(s_num)
        age = int(age)
        cat = int(cat)

        print("The Amont is ", amount)
        print("The day month  year", day, month, year)
        print("Time :", hour)
        print("Sendberds num", sender_num, age, cat)

        user_input = {
            "trans_hour": hour,
            "trans_day": day,
            "trans_month": month,
            "trans_year": year,
            "category": cat,
            "upi_number": sender_num,
            "age": age,
            "trans_amount": amount,
        }
        user_df = pd.DataFrame([user_input])
        prediction = loaded_rf_model.predict(user_df)
        randomf_prediction = prediction
        print(f"Predicted Fraud Risk By Random Forest (10000): {prediction[0]}")

        # cnn model l
        loaded_model = load_model("CNN_Model.h5")
        with open("scaler.pkl", "rb") as scaler_file:
            loaded_scaler = pickle.load(scaler_file)

        user_df = pd.DataFrame([user_input])
        user_input_scaled = loaded_scaler.transform(user_df)
        loaded_prediction_prob = loaded_model.predict(user_input_scaled)
        loaded_prediction_binary = (loaded_prediction_prob > 0.5).astype(int)

        print(f"Predicted Fraud Probability (CNN Model): {loaded_prediction_prob[0]}")
        print(
            f"Predicted Fraud Risk (Binary) (CNN Model): {loaded_prediction_binary[0]}"
        )
        cnn_predction = loaded_prediction_prob[0]

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
        print(
            f"Predicted Fraud Risk (Binary) (RNN Model): {loaded_prediction_binary[0]}"
        )
        rnn_predction = loaded_prediction_prob[0]
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

        user_input = {
            "sender_Mobile": sender_num,
            "transaction_id": trans,
            "transaction_time": time,
            "transaction_amount": amount,
            "receiver_mobile": r_num,
        }

        print(valid, invalid)
        if len(valid) > len(invalid):
            print(
                "The Probability of the Transaction being Fraud is",
                sum(valid) / len(valid),
            )
            prob = sum(valid) / len(valid)
            print(" Possibility Genuine UPI Tranaction ")
            p1 = prob[0] * 100
            mail_send(
                email,
                "Transaction Alert !!! - Possibility of Valid Transaction",
                "The Probability of the Transaction being Fraud is " + str(p1) + "%",
                user_input,
            )

            return render_template("result.html", ans=True, prob=prob[0] * 100)
        else:
            print(
                "The Probability of the Transaction being Fraud is",
                sum(invalid) / len(invalid),
            )
            prob = sum(invalid) / len(invalid)
            p1 = prob[0] * 100
            print(" Possibility Fake UPI Tranaction ")
            mail_send(
                email,
                "Transaction Alert !!! -  Possibility of Fraud Transaction",
                "The Probability of the Transaction being Fraud is " + str(p1) + "%",
                user_input,
            )

            return render_template("result.html", ans=False, prob=prob[0] * 100)


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/contactus")
def contactUs():
    return render_template("contactUs.html")


def mail_send(receive_mail, subject, body, user_input):

    email = "fraudfighters2024@gmail.com"
    receiver_email = receive_mail
    subject = subject
    message = body
    text1 = f"Subject: {subject}\n\n{message}"
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(email, "tmbjjhmsnnvltspc")
    server.sendmail(
        email, receiver_email, text1 + ".  Thanks For Using Our Application "
    )


# if __name__ == "__main__":
#     app.run(debug=True)
