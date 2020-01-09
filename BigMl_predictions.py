from bigml.api import BigML
from bigml.model import Model

api = BigML("friendlycoconut","936583948d0c870ccb5cb004afcf6c13f086c900")

source = api.create_source('https://static.bigml.com/csv/diabetes.csv')
api.ok(source)
dataset = api.create_dataset(source)

api.ok(dataset)
model = api.create_model(dataset)
api.ok(model)


local_model = Model (model)
input_data ={"age": 65, "bmi": 36, "plasma glucose": 180, "pregnancies": 3}
local_model.predict(input_data, add_confidence=True)