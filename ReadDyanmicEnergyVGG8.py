import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def predict_all_layers(mem_cell_type, subarray_size):
    # Load the data
    data = pd.read_csv('/content/ReadDynamicEnergyVGG8.csv', delimiter=',')
    print('-' * 85)
    print("Predictions for All Layers")
    print('-' * 85)
    # Separate features (X) and target variable (y)
    X = data.iloc[:, :2]  # First two columns: MemCellType and numRowSubArray/numColSubArray
    y = data.iloc[:, 2:]  # Columns from position 2 onwards: Layer1, Layer2, ...

    # Create and fit the model
    model = RandomForestRegressor(n_estimators=100, random_state=35)
    model.fit(X, y)

    # Now you can use this model to predict read dynamic energy for new data
    # For example, if you want to predict read dynamic energy for MemCellType=1, numRowSubArray/numColSubArray=64, and Layer=10
    new_data = pd.DataFrame({
        'MemCellType': [mem_cell_type],
        'numRowSubArray/numColSubArray': [subarray_size]
    })
    training_accuracy = model.score(X, y)
    print(f"Training Accuracy: {training_accuracy}")

    # Predicting read dynamic energy for new data
    predicted_read_dynamic_energy = model.predict(new_data)
    print("Predicted Read Dynamic Energy:", predicted_read_dynamic_energy)

    print('-' * 85)
    print("Plot for Read Dynamic Energy")
    print('-' * 85)
    # Choose MemCellType and numRowSubArray/numColSubArray
    mem_cell_type = mem_cell_type
    subarray_size = subarray_size

    # Filter the training data for the chosen MemCellType and numRowSubArray/numColSubArray
    filtered_data = data[(data['MemCellType'] == mem_cell_type) & (data['numRowSubArray/numColSubArray'] == subarray_size)]

    # Extract the actual read dynamic energy values
    actual_values = filtered_data.iloc[:, 2:]

    # Plot the predicted read dynamic energy and the actual read dynamic energy values
    plt.figure(figsize=(10, 6))
    plt.plot(actual_values.values[0], label='Actual')
    plt.plot(predicted_read_dynamic_energy[0], label='Predicted')
    plt.xlabel('Layer')
    plt.ylabel('Read Dynamic Energy')
    plt.title(f'Predicted vs Actual Read Dynamic Energy for MemCellType={mem_cell_type}, Subarray Size={subarray_size}')
    plt.legend()
    plt.grid(True)
    plt.show()

    print('-' * 85)
    print("Trained with splitted Data")
    print('-' * 85)
    # Load the data
    data = pd.read_csv('/content/ReadDynamicEnergyVGG8.csv', delimiter=',')

    # Separate features (X) and target variable (y)
    X = data.iloc[:, :2]  # First two columns: MemCellType and numRowSubArray/numColSubArray
    y = data.iloc[:, 2:]  # Columns from position 2 onwards: Layer1, Layer2, ...

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the model
    model = RandomForestRegressor(n_estimators=100, random_state=35)
    model.fit(X_train, y_train)

    # Predicting read dynamic energy for the testing data
    predicted_read_dynamic_energy_test = model.predict(X_test)

    # Plot a graph comparing the predicted and actual read dynamic energy values for a specific MemCellType and numRowSubArray/numColSubArray
    mem_cell_type = 1
    subarray_size = 128
    training_accuracy = model.score(X, y)
    print(f"Training Accuracy: {training_accuracy}")

    # Filter the testing data for the chosen MemCellType and numRowSubArray/numColSubArray
    filtered_data_test = X_test[(X_test['MemCellType'] == mem_cell_type) & (X_test['numRowSubArray/numColSubArray'] == subarray_size)]

    if not filtered_data_test.empty:
        index_test = filtered_data_test.index[0]

        # Extract the actual read dynamic energy values for the chosen MemCellType and numRowSubArray/numColSubArray
        actual_values_test = y_test.iloc[index_test]

        # Plot the predicted and actual read dynamic energy values
        plt.figure(figsize=(10, 6))
        plt.plot(actual_values_test.values, label='Actual')
        plt.plot(predicted_read_dynamic_energy_test[index_test], label='Predicted')
        plt.xlabel('Layer')
        plt.ylabel('Read Dynamic Energy')
        plt.title(f'Predicted vs Actual Read Dynamic Energy for MemCellType={mem_cell_type}, Subarray Size={subarray_size}')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No data found for the specified MemCellType and Subarray Size.")


