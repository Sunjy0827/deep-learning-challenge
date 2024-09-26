# Deep-Learning-Challenge

<h2>Alphabet Soup Charity</h2>

<h3>Overview</h3>

<p> 
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively
</p>

<hr/>

<h3>Goal</h3>

<ul>
<li>Preprocess the Data</li>
<li>Compile, Train, and Evaluate the Model</li>
<li>Optimize the Model</li>
<li>Write a Report on the Neural Network Model</li>
</ul>

</hr>

<h3>Tools and Techniques</h3>

<ul>
<li>Python</li>
<li>Pandas</li>
<li>Tensorflow</li>
<li>Scikit-learn</li>
<li>Google Colab</li>
</ul>

</hr>

<h3>Project Structure</h3>
<hr/>
<h4>Part I: Preprocessing Data</h4>

<p>
To begin with the machine learning process, the raw charity_data.csv file was first dissected to identify the target variable (y, which in this case represents whether the outcome was successful or not) and the feature variables (all other variables except for EIN and NAME, which were dropped). Next, the number of unique values for each column was determined, and for columns with more than 10 unique values, the number of data points per unique value was evaluated. This information was used to consolidate rare categorical variables into a new "Other" category, ensuring that the replacement was successful. Specifically, the APPLICATION_TYPE and CLASSIFICATION columns were identified as having more than 10 unique values. Afterward, pd.get_dummies() was applied to encode the categorical variables. The processed data was then split into feature (X) and target (y) arrays, followed by splitting into training and testing datasets using train_test_split. Finally, the features in both datasets were scaled using a StandardScaler, which was fitted to the training data and used to transform both the training and testing features.
</p>


```python
# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf

#  Import and read the charity_data.csv.
import pandas as pd
application_df = pd.read_csv("https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv")
application_df.head()

# Drop the non-beneficial ID columns, 'EIN' and 'NAME'.
application_df = application_df.drop(columns = ['EIN', 'NAME'])

# Determine the number of unique values in each column.
print(application_df.nunique())

# Look at APPLICATION_TYPE value counts to identify and replace with "Other"
print(application_df["APPLICATION_TYPE"].value_counts())

# Choose a cutoff value and create a list of application types to be replaced
# use the variable name `application_types_to_replace`
application_types_to_replace = ["T9", "T13", "T12", "T2", "T25", "T14", "T29", "T15", "T17"]

# Replace in dataframe
for app in application_types_to_replace:
    application_df['APPLICATION_TYPE'] = application_df['APPLICATION_TYPE'].replace(app,"Other")

# Check to make sure replacement was successful
print(application_df['APPLICATION_TYPE'].value_counts())

# Look at CLASSIFICATION value counts to identify and replace with "Other"
print(application_df["CLASSIFICATION"].value_counts())

# You may find it helpful to look at CLASSIFICATION value counts >1
print(application_df["CLASSIFICATION"].value_counts()[application_df["CLASSIFICATION"].value_counts()>1])

# Choose a cutoff value and create a list of classifications to be replaced
# use the variable name `classifications_to_replace`
classifications_to_replace = application_df["CLASSIFICATION"].value_counts()[application_df["CLASSIFICATION"].value_counts()<1000].index.tolist()

# Replace in dataframe
for cls in classifications_to_replace:
    application_df['CLASSIFICATION'] = application_df['CLASSIFICATION'].replace(cls,"Other")

# Check to make sure replacement was successful
print(application_df['CLASSIFICATION'].value_counts())

# Convert categorical data to numeric with `pd.get_dummies`
application_dummies=pd.get_dummies(application_df).astype(int)
application_dummies.head()

# Split our preprocessed data into our features and target arrays
y = application_dummies["IS_SUCCESSFUL"]
X = application_dummies.drop(columns="IS_SUCCESSFUL")

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

```
<hr/>

<h4>Part II: Compile, Train, and Evaluate the Model</h4>
</br>

<p>
I executed the steps to build a neural network model using TensorFlow and Keras by first assigning the number of input features and determining the number of nodes for each layer. I then created the first hidden layer with an appropriate activation function, and since necessary, I added a second hidden layer with its own activation function. I followed this by constructing the output layer using the appropriate activation function. After checking the model's structure, I compiled and trained the model. I also implemented a callback to save the model's weights every five epochs. Once training was complete, I evaluated the model using the test data to determine its loss and accuracy. Finally, I saved and exported the model results in an HDF5 file named AlphabetSoupCharity.h5.

<b>Model Summary:</b>

<ul>
<li>2 hidden layers (8 and 3 nodes)</li>
<li>50 EPOCH</li>
<li>Activation Functions: ReLU and sigmoid (Output)</li>
<li>Result: 72.82%</li>
</ul>

</p>


```python

# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_input_features = len(X_train_scaled[0])
hidden_nodes_layer1 =  8
hidden_nodes_layer2 = 5

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu")
)

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()

# Compile the model
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

###
# This part was to have checkpoint every 5 epoch, but it did not work
import os
print(os.getcwd())

# Create a directory for checkpoints
checkpoint_dir = '/content/checkpoints/'
os.makedirs(checkpoint_dir, exist_ok=True)

# Define the callback to save the model's weights every 5 epochs
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir + "model_weights_epoch_{epoch:02d}.weights.h5",
    save_weights_only=True,
    save_freq=5 * (len(X_train_scaled) // 32),
    verbose=1
)
###

# Train the model
fit_model = nn.fit(X_train_scaled,y_train,epochs=50)

# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# Export our model to HDF5 file
from google.colab import files

nn.save('/content/AlphabetSoupCharity_Optimization.h5')
files.download('/content/AlphabetSoupCharity_Optimization.h5')


```
<hr/>

<h4>Part III: Optimize the Model</h4>

<p>
After looking into the dataset, I found that there were roughly 56% (19206 organizations) of organizations had venture one or two times only. Therefore, instead of using only two variables (application_type and classification), I decided to bucketing the categorical variable "NAME" in the optimization process.
</p>

```python

# Look at NAME value counts to identify and replace with "Other"
print(application_df["NAME"].value_counts())

# Choose a cutoff value and create a list of NAMEs to be replaced
# use the variable name `name_to_replace`
name_to_replace = application_df["NAME"].value_counts()[application_df["NAME"].value_counts()<3].index.tolist()

# Replace in dataframe
for name in name_to_replace:
    application_df['NAME'] = application_df['NAME'].replace(name,"Other")
# Check to make sure replacement was successful
print(application_df['NAME'].value_counts())

```

<p>
After that, I also changed activation function, and the number of hidden layers to get better result below. As you see below, the last optimization model 3 got the best result.

<b>Optimization Model 1 Summary:</b>

<ul>
<li>2 hidden layers (8 and 3 nodes)</li>
<li>50 EPOCH</li>
<li>Activation Functions: ReLU, <b>ELU</b> and Sigmoid (Output)</li>
<li>Result: 79.44%</li>
</ul>

<b>Optimization Model 2 Summary:</b>

<ul>
<li>3 hidden layers (8, 5, 3 nodes)</li>
<li>50 EPOCH</li>
<li>Activation Functions: ReLU and sigmoid (Output)</li>
<li>Result: 79.46%</li>
</ul>

<b>Optimization Model 3 Summary:</b>

<ul>
<li>3 hidden layers (8, 5, 3 nodes)</li>
<li>100 EPOCH</li>
<li>Activation Functions: ReLU, ELU and Sigmoid (Output)</li>
<li style="color: green;"><b>Result: 79.52%</b></li>
</ul>

</p>
<hr/>

<h4>Part IV: Report</h4>

<b>Overview:</b>

The primary objective of this project is to develop a model that assists the nonprofit organization Alphabet Soup in selecting applicants for funding who have the greatest potential for success in their endeavors.

<b>Report</b>

Data Preprocessing

<ul>
<li>
What variable(s) are the target(s) for your model?</br>
The target variable is the last column "Is_SUCCESSFUL" which has the boolean value can be utilized for logistic regression.
</li>
<li>
What variable(s) are the features for your model?</br>
All columns except for the 'EIN' column are features based on the optimization model.
</li>
<li>What variable(s) should be removed from the input data because they are neither targets nor features?</li>
'EIN' column has to be removed because it just identifies the applicant's individual charity funding case.
</ul>


Compiling, Training, and Evaluating the Model

<ul>
<li>
How many neurons, layers, and activation functions did you select for your neural network model, and why?</br>
For my neural network model, I selected the following architecture:

<ul>
    <li>Layers: I started with 2 hidden layers in the initial model. As I optimized the model, I increased it to 3 hidden layers in subsequent versions.</li>
    <li>Neurons: The first model had 8 neurons in the first hidden layer and 3 neurons in the second hidden layer. In the optimized models, I kept the first hidden layer with 8 neurons, increased the second hidden layer to 5 neurons, and retained 3 neurons in the third layer.</li>
    <li>Activation Functions: I initially used the ReLU activation function for the hidden layers and sigmoid for the output layer. In my optimization attempts, I also experimented with the ELU activation function in some layers to see if it would improve performance.</li>
</ul>
I chose these configurations based on common practices in deep learning, where starting with a smaller number of layers and neurons allows for easier debugging and gradual tuning of the model's complexity. The ReLU activation function is known for its effectiveness in hidden layers, while the sigmoid function is suitable for binary classification tasks in the output layer.

</li></br>
<li>
Were you able to achieve the target model performance?</br>
My initial attempt achieved an accuracy of 72.82%. However, through iterative optimization, I successfully enhanced the model's performance, reaching an accuracy of 79.52%, which exceeded the 75% target.
</li><br/>
<li>What steps did you take in your attempts to increase model performance?</br>
<ul>
    <li><strong>Layer and Neuron Adjustments:</strong> I experimented with different numbers of hidden layers and neurons. By adding an additional layer and adjusting the number of neurons, I could capture more complex patterns in the data.</li>
    <li><strong>Activation Function Experiments:</strong> I tested different activation functions like <strong>ReLU</strong> and <strong>ELU</strong> to see which combinations provided better performance. This helped in addressing potential issues like vanishing gradients.</li>
    <li><strong>Increased Epochs:</strong> In my final optimization model, I increased the number of training epochs from <strong>50 to 100</strong> to allow the model to learn more from the data and improve its generalization capabilities.</li>
    <li><strong>Monitoring Performance:</strong> I consistently monitored the model&apos;s performance metrics, such as accuracy, after each run, enabling me to assess which changes yielded positive results and adjust my strategy accordingly.</li>
</ul>
</li>
</ul>

<hr/>

<h3>Summary</h3>

<p>
The deep learning model initially achieved an accuracy of 72.82%, which fell short of the target of over 75%. However, through iterative optimization, I enhanced the model's performance to 79.52%, surpassing the target. As a recommendation for further improvement, I suggest exploring ensemble methods such as Random Forests or Gradient Boosting. These methods combine multiple models to improve accuracy and robustness, provide feature importance for better interpretability, are often more efficient with smaller datasets, and tend to be less prone to overfitting. Implementing an ensemble approach could lead to even higher accuracy and reliability in solving this classification problem.
</p>