from flask import Flask, request, jsonify
import pickle

app = Flask(__name__) #Initialize Flask app

with open("./models/mlops-sample-file.pkl", "rb") as fileobj:
    iris_model = pickle.load(fileobj) # Load the pre-trained model


@app.route( "/", methods=["GET"])
def home():
    return"Welcome to the Isha's Flask App!"





@app.route("/get_square", methods=["POST"])
def get_square():
    data = request.get_json()
    number = data.get("number")
    return jsonify({"square": number ** 2}) 

@app.route("/predict", methods=["POST"]) #<-- this is the controller
def iris_prediction(): # <-- this is view function
    data = request.get_json()
    sepal_lenght = data.get("sl")
    petal_lenght = data.get("pl")
    sepal_width = data.get("sw")
    petal_width = data.get("pw")
    flower_type = iris_model.predict([[sepal_lenght, sepal_width, petal_lenght, petal_width]])
    return jsonify({"predcited_flower_type": flower_type[0]})  

if __name__ == "__main__":
    app.run(port=5000) #Run the app in debug mode 