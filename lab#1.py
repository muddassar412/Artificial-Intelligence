import sklearn
import numpy as np
import pandas as pd
from sklearn import linear_model
x = np.array([[5],[15],[25],[35],[45],[55]])#.reshape((-1,1))
y = np.array([5,20,14,32,22,38])
modle = linear_model.LinearRegression()
modle.fit(x,y)
x_new = np.array([150]).reshape((-1,1))
y-new = modle.predict(x_new)
print(y-new)


"""
from google.colab import drive
import sklearn
import numpy as np
import pandas as pd
from sklearn import linear_model

drive.mount('/content/drive')

# Assuming your excel file is named 'your_excel_file.xlsx' and located in your Google Drive
# Replace 'your_excel_file.xlsx' and the path if necessary
file_path = '/content/drive/MyDrive/your_excel_file.xlsx'

try:
  # Read the excel file into a pandas DataFrame
  df = pd.read_excel(file_path)

  # Convert the relevant columns to numpy arrays
  # Replace 'column_x' and 'column_y' with the actual column names from your excel file
  x = df['column_x'].values.reshape((-1, 1))
  y = df['column_y'].values

  # Create and train the linear regression model
  model = linear_model.LinearRegression()
  model.fit(x, y)

  # Example prediction (replace with your desired input)
  x_new = np.array([150]).reshape((-1, 1))
  y_new = model.predict(x_new)
  print(y_new)

except FileNotFoundError:
  print(f"Error: File not found at {file_path}")
except KeyError as e:
  print(f"Error: Column not found in the excel file: {e}")
except Exception as e:
  print(f"An unexpected error occurred: {e}")
"""
