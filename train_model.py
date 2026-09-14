import pandas as pd

data = pd.read_csv("student_data.csv")


print(data)

x = data.drop("final_score",axis=1)
y = data["final_score"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
print("model training completed!")
predictions = model.predict(x_test)

print("actual scores:")
print(y_test.values)

print("predicted scores:")
print(predictions)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test,predictions)
print("mean absolute error:",mae)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, predictions)
print("R2 score:",r2)

print("\n student performance prediction")

hours = float(input("enter study hours: "))
attendance = float(input("enter attendence percentage: "))
previous_score = float(input("enter previous score"))
assignment = float(input("enter completed assignments: "))

new_student =[[hours, attendance,previous_score,assignment]]
result = model.predict(new_student)

print("predicted final score:",round(result[0],2))