import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt



def predict_all_parameters(technode_size,subarray_size):
    # Load the data
    data = pd.read_csv('/content/technode.csv')

    # Select features and target
    X = data[['Technode', 'numRowSubArray/numColSubArray']]
    y_energy_efficiency = data['Energy Efficiency TOPS/W (Pipelined Process)']
    y_throughput_tops = data['Throughput TOPS (Pipelined Process)']
    y_throughput_fps = data['Throughput FPS (Pipelined Process)']
    y_compute_efficiency = data['Compute efficiency TOPS/mm^2 (Pipelined Process)']
    print('-' * 85)
    print("Accuracy for All Parmeters")
    print('-' * 85)
    # Define the model for Energy Efficiency TOPS/W
    model_energy_efficiency = RandomForestRegressor()
    model_energy_efficiency.fit(X, y_energy_efficiency)
    predicted_energy_efficiency = model_energy_efficiency.predict(X)
    r2_energy_efficiency = r2_score(y_energy_efficiency, predicted_energy_efficiency)
    print("R-squared score for Energy Efficiency TOPS/W (Pipelined Process):", r2_energy_efficiency*100)

    # Define the model for Throughput TOPS
    model_throughput_tops = RandomForestRegressor()
    model_throughput_tops.fit(X, y_throughput_tops)
    predicted_throughput_tops = model_throughput_tops.predict(X)
    r2_throughput_tops = r2_score(y_throughput_tops, predicted_throughput_tops)
    print("R-squared score for Throughput TOPS (Pipelined Process):", r2_throughput_tops*100)

    # Define the model for Throughput FPS
    model_throughput_fps = RandomForestRegressor()
    model_throughput_fps.fit(X, y_throughput_fps)
    predicted_throughput_fps = model_throughput_fps.predict(X)
    r2_throughput_fps = r2_score(y_throughput_fps, predicted_throughput_fps)
    print("R-squared score for Throughput FPS (Pipelined Process):", r2_throughput_fps*100)

    # Define the model for Compute efficiency
    model_compute_efficiency = RandomForestRegressor()
    model_compute_efficiency.fit(X, y_compute_efficiency)
    predicted_compute_efficiency = model_compute_efficiency.predict(X)
    r2_compute_efficiency = r2_score(y_compute_efficiency, predicted_compute_efficiency)
    print("R-squared score for Compute efficiency TOPS/mm^2 (Pipelined Process):", r2_compute_efficiency*100)
    # Selecting data for a specific subarray size
    subarray_size = subarray_size
    data_subarray = data[data['numRowSubArray/numColSubArray'] == subarray_size]

    # Select features and target
    X = data_subarray[['Technode']]
    y_energy_efficiency = data_subarray['Energy Efficiency TOPS/W (Pipelined Process)']
    y_throughput_tops = data_subarray['Throughput TOPS (Pipelined Process)']
    y_throughput_fps = data_subarray['Throughput FPS (Pipelined Process)']
    y_compute_efficiency = data_subarray['Compute efficiency TOPS/mm^2 (Pipelined Process)']
    print('-' * 85)
    print("Accuracy for All Parmeters")
    print('-' * 85)
    # Define the model for Energy Efficiency TOPS/W
    model_energy_efficiency = RandomForestRegressor()
    model_energy_efficiency.fit(X, y_energy_efficiency)
    predicted_energy_efficiency = model_energy_efficiency.predict(X)
    r2_energy_efficiency = r2_score(y_energy_efficiency, predicted_energy_efficiency)
    print("R-squared score for Energy Efficiency TOPS/W (Pipelined Process):", r2_energy_efficiency)

    # Define the model for Throughput TOPS
    model_throughput_tops = RandomForestRegressor()
    model_throughput_tops.fit(X, y_throughput_tops)
    predicted_throughput_tops = model_throughput_tops.predict(X)
    r2_throughput_tops = r2_score(y_throughput_tops, predicted_throughput_tops)
    print("R-squared score for Throughput TOPS (Pipelined Process):", r2_throughput_tops)

    # Define the model for Throughput FPS
    model_throughput_fps = RandomForestRegressor()
    model_throughput_fps.fit(X, y_throughput_fps)
    predicted_throughput_fps = model_throughput_fps.predict(X)
    r2_throughput_fps = r2_score(y_throughput_fps, predicted_throughput_fps)
    print("R-squared score for Throughput FPS (Pipelined Process):", r2_throughput_fps)

    # Define the model for Compute efficiency
    model_compute_efficiency = RandomForestRegressor()
    model_compute_efficiency.fit(X, y_compute_efficiency)
    predicted_compute_efficiency = model_compute_efficiency.predict(X)
    r2_compute_efficiency = r2_score(y_compute_efficiency, predicted_compute_efficiency)
    print("R-squared score for Compute efficiency TOPS/mm^2 (Pipelined Process):", r2_compute_efficiency)

    print('-' * 85)
    print(" Plotting for Energy Efficiency TOPS/W for Subarray Size {subarray_size}")
    print('-' * 85)
    # Plotting for Energy Efficiency TOPS/W
    plt.figure(figsize=(10, 5))
    plt.plot(data_subarray['Technode'], y_energy_efficiency, color='blue', label='Training Value')
    plt.plot(data_subarray['Technode'], predicted_energy_efficiency, color='red', label='Predicted Value')
    plt.xlabel('Technode')
    plt.ylabel('Energy Efficiency TOPS/W (Pipelined Process)')
    plt.title(f'Energy Efficiency TOPS/W Prediction for Subarray Size {subarray_size}')
    plt.legend()
    plt.show()

    print('-' * 85)
    print("Plotting for Throughput TOPS for Subarray Size {subarray_size}")
    print('-' * 85)
    # Plotting for Throughput TOPS
    plt.figure(figsize=(10, 5))
    plt.plot(data_subarray['Technode'], y_throughput_tops, color='blue', label='Training Value')
    plt.plot(data_subarray['Technode'], predicted_throughput_tops, color='red', label='Predicted Value')
    plt.xlabel('Technode')
    plt.ylabel('Throughput TOPS (Pipelined Process)')
    plt.title(f'Throughput TOPS Prediction for Subarray Size {subarray_size}')
    plt.legend()
    plt.show()
    print('-' * 85)
    print("Plotting for Throughput FPS for Subarray Size {subarray_size}")
    print('-' * 85)
    # Plotting for Throughput FPS
    plt.figure(figsize=(10, 5))
    plt.plot(data_subarray['Technode'], y_throughput_fps, color='blue', label='Training Value')
    plt.plot(data_subarray['Technode'], predicted_throughput_fps, color='red', label='Predicted Value')
    plt.xlabel('Technode')
    plt.ylabel('Throughput FPS (Pipelined Process)')
    plt.title(f'Throughput FPS Prediction for Subarray Size {subarray_size}')
    plt.legend()
    plt.show()

    print('-' * 85)
    print("Plotting for Compute efficiency for Subarray Size {subarray_size}")
    print('-' * 85)
    # Plotting for Compute efficiency
    plt.figure(figsize=(10, 5))
    plt.plot(data_subarray['Technode'], y_compute_efficiency, color='blue', label='Training Value')
    plt.plot(data_subarray['Technode'], predicted_compute_efficiency, color='red', label='Predicted Value')
    plt.xlabel('Technode')
    plt.ylabel('Compute efficiency TOPS/mm^2 (Pipelined Process)')
    plt.title(f'Compute efficiency Prediction for Subarray Size {subarray_size}')
    plt.legend()
    plt.show()

    print('\n' * 5)

    

    # Define a function to predict values for a given Technode and subarray size
    def predict_values(technode, subarray_size):
        # Predict Energy Efficiency TOPS/W
        energy_efficiency_prediction = model_energy_efficiency.predict([[technode]])
        # Predict Throughput TOPS
        throughput_tops_prediction = model_throughput_tops.predict([[technode]])
        # Predict Throughput FPS
        throughput_fps_prediction = model_throughput_fps.predict([[technode]])
        # Predict Compute efficiency
        compute_efficiency_prediction = model_compute_efficiency.predict([[technode]])

        # Print predicted values
        print("Predicted values for Technode =", technode, "and Subarray Size =", subarray_size)
        print("Energy Efficiency TOPS/W (Pipelined Process):", energy_efficiency_prediction[0])
        print("Throughput TOPS (Pipelined Process):", throughput_tops_prediction[0])
        print("Throughput FPS (Pipelined Process):", throughput_fps_prediction[0])
        print("Compute efficiency TOPS/mm^2 (Pipelined Process):", compute_efficiency_prediction[0])
        print('\n' * 5)

    # Call the function with a specific Technode and subarray size
    technode_input = technode_size
    subarray_size_input = subarray_size
    print('-' * 85)
    print(f"Predicted Values For <- Technode -> {technode_input}nm and  <- SubArray -> {subarray_size_input} ")
    print('-' * 85)
    predict_values(technode_input, subarray_size_input)

    # Load the data
    data = pd.read_csv('/content/technode.csv')

    # Selecting data for a specific Technode
    technode = technode_size
    data_technode = data[data['Technode'] == technode]

    # Define fixed Technode
    X = data_technode[['numRowSubArray/numColSubArray']]

    # Select features and target
    y_energy_efficiency = data_technode['Energy Efficiency TOPS/W (Pipelined Process)']
    y_throughput_tops = data_technode['Throughput TOPS (Pipelined Process)']
    y_throughput_fps = data_technode['Throughput FPS (Pipelined Process)']
    y_compute_efficiency = data_technode['Compute efficiency TOPS/mm^2 (Pipelined Process)']

    print('-' * 85)
    print("Accuracy for All Parmeters")
    print('-' * 85)
    # Define the model for Energy Efficiency TOPS/W
    model_energy_efficiency = RandomForestRegressor()
    model_energy_efficiency.fit(X, y_energy_efficiency)
    predicted_energy_efficiency = model_energy_efficiency.predict(X)
    r2_energy_efficiency = r2_score(y_energy_efficiency, predicted_energy_efficiency)
    print("R-squared score for Energy Efficiency TOPS/W (Pipelined Process):", r2_energy_efficiency)

    # Define the model for Throughput TOPS
    model_throughput_tops = RandomForestRegressor()
    model_throughput_tops.fit(X, y_throughput_tops)
    predicted_throughput_tops = model_throughput_tops.predict(X)
    r2_throughput_tops = r2_score(y_throughput_tops, predicted_throughput_tops)
    print("R-squared score for Throughput TOPS (Pipelined Process):", r2_throughput_tops)

    # Define the model for Throughput FPS
    model_throughput_fps = RandomForestRegressor()
    model_throughput_fps.fit(X, y_throughput_fps)
    predicted_throughput_fps = model_throughput_fps.predict(X)
    r2_throughput_fps = r2_score(y_throughput_fps, predicted_throughput_fps)
    print("R-squared score for Throughput FPS (Pipelined Process):", r2_throughput_fps)

    # Define the model for Compute efficiency
    model_compute_efficiency = RandomForestRegressor()
    model_compute_efficiency.fit(X, y_compute_efficiency)
    predicted_compute_efficiency = model_compute_efficiency.predict(X)
    r2_compute_efficiency = r2_score(y_compute_efficiency, predicted_compute_efficiency)
    print("R-squared score for Compute efficiency TOPS/mm^2 (Pipelined Process):", r2_compute_efficiency)

    print('-' * 85)
    print(f'Energy Efficiency TOPS/W Prediction for Technode {technode}')
    print('-' * 85)
    # Plotting for Energy Efficiency TOPS/W
    plt.figure(figsize=(10, 5))
    plt.plot(data_technode['numRowSubArray/numColSubArray'], y_energy_efficiency, color='blue', label='Training Value')
    plt.plot(data_technode['numRowSubArray/numColSubArray'], predicted_energy_efficiency, color='red', label='Predicted Value')
    plt.xlabel('Subarray Size')
    plt.ylabel('Energy Efficiency TOPS/W (Pipelined Process)')
    plt.title(f'Energy Efficiency TOPS/W Prediction for Technode {technode}')
    plt.legend()
    plt.show()

    print('-' * 85)
    print(f'Throughput TOPS Prediction for Technode {technode}')
    print('-' * 85)
    # Plotting for Throughput TOPS
    plt.figure(figsize=(10, 5))
    plt.plot(data_technode['numRowSubArray/numColSubArray'], y_throughput_tops, color='blue', label='Training Value')
    plt.plot(data_technode['numRowSubArray/numColSubArray'], predicted_throughput_tops, color='red', label='Predicted Value')
    plt.xlabel('Subarray Size')
    plt.ylabel('Throughput TOPS (Pipelined Process)')
    plt.title(f'Throughput TOPS Prediction for Technode {technode}')
    plt.legend()
    plt.show()

    print('-' * 85)
    print(f'Throughput FPS Prediction for Technode {technode}')
    print('-' * 85)
    # Plotting for Throughput FPS
    plt.figure(figsize=(10, 5))
    plt.plot(data_technode['numRowSubArray/numColSubArray'], y_throughput_fps, color='blue', label='Training Value')
    plt.plot(data_technode['numRowSubArray/numColSubArray'], predicted_throughput_fps, color='red', label='Predicted Value')
    plt.xlabel('Subarray Size')
    plt.ylabel('Throughput FPS (Pipelined Process)')
    plt.title(f'Throughput FPS Prediction for Technode {technode}')
    plt.legend()
    plt.show()

    print('-' * 85)
    print(f'Compute efficiency Prediction for Technode {technode}')
    print('-' * 85)
    # Plotting for Compute efficiency
    plt.figure(figsize=(10, 5))
    plt.plot(data_technode['numRowSubArray/numColSubArray'], y_compute_efficiency, color='blue', label='Training Value')
    plt.plot(data_technode['numRowSubArray/numColSubArray'], predicted_compute_efficiency, color='red', label='Predicted Value')
    plt.xlabel('Subarray Size')
    plt.ylabel('Compute efficiency TOPS/mm^2 (Pipelined Process)')
    plt.title(f'Compute efficiency Prediction for Technode {technode}')
    plt.legend()
    plt.show()

