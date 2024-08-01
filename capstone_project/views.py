from django.shortcuts import render, redirect
from .forms import CSVUploadForm
from .models import CSVFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def upload_csv(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = form.save()
            # Run your prediction code here
            csv_path = csv_file.file.path
            ontarioPublic = pd.read_csv(csv_path)
            # Assume a function called 'make_predictions' exists
            prediction_results = make_predictions(ontarioPublic)
            return render(request, 'csv_detail.html', {
                'csv_file': csv_file,
                **prediction_results  # Unpack the dictionary returned from make_predictions
            })
    else:
        form = CSVUploadForm()
    return render(request, 'upload_csv.html', {'form': form})

def csv_detail(request, pk):
    csv_file = CSVFile.objects.get(pk=pk)
    return render(request, 'csv_detail.html', {'csv_file': csv_file})

def make_predictions(ontarioPublic):
    # Select targets and predictors
    # Target 1: Total Operating Revenues
    # Target 2: Total Operating Expenditures
    target_1 = 'Soma de B2.9  Total Operating Revenues'
    target_2 = 'Soma de B5.0  Total Operating Expenditures'

    # Select potential predictors (excluding non-numeric columns and the targets)
    predictors = ontarioPublic.select_dtypes(include=[np.number]).drop(columns=[target_1, target_2])


    # In[25]:


    # Handle missing values by filling with the mean (for simplicity)
    predictors = predictors.fillna(predictors.mean())
    ontarioPublic[target_1] = ontarioPublic[target_1].fillna(ontarioPublic[target_1].mean())
    ontarioPublic[target_2] = ontarioPublic[target_2].fillna(ontarioPublic[target_2].mean())


    # In[26]:


    # Normalize the predictors
    scaler = StandardScaler()
    X = scaler.fit_transform(predictors)


    # In[28]:


    # Perform PCA for both targets
    pca = PCA(n_components=5)  # Number of components can be adjusted
    X_pca = pca.fit_transform(X)



    # Predicting Total Operating Revenues
    print("Model for Total Operating Revenues")
    operating_revenues = train_and_evaluate_model(X_pca, ontarioPublic[target_1])

    # Predicting Total Operating Expenditures
    print("Model for Total Operating Expenditures")
    operating_expenditures = train_and_evaluate_model(X_pca, ontarioPublic[target_2])

    return{
        'operating_revenues': operating_revenues,
        'operating_expenditures': operating_expenditures
    }

# Develop predictive models

# Function to train and evaluate the model
def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        'mse': mse,
        'r2': r2,
    }
