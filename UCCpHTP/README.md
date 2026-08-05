# Data-Driven Aerodynamic Modeling of Cp Distribution on the HTP <img src="images/model.png" alt="Aircraft HTP" align="right" width="200">

This case study demonstrates the application of the **Surrogate Factory (SF)** framework to predict and analyze the **pressure coefficient (Cp) distribution** over the **Horizontal Tail Plane (HTP)** of an aircraft.  
The objective is to reproduce the aerodynamic behavior obtained from CFD simulations through a **data-driven surrogate model** based on a **Multi-Layer Perceptron (MLP)** implemented with **PyLOM** and validated using **NGSValidation**.

---

## Objective

The aim of this study is to develop a surrogate model capable of predicting the **Cp distribution across the HTP surface** for a given set of **flight conditions** (Mach number and angle of attack) at **Flight Level 310**.

The model learns the relationship between the geometric and flow parameters and the resulting surface pressure coefficient at each mesh point.

<p align="center">
  <img src="../../images/mesh.png" alt="HTP Mesh Geometry" width="200">
</p>

---

## Dataset Description

The input dataset originates from CFD simulations performed on a **structured 3D mesh** of the HTP region.  
Each point in the mesh is defined by its spatial coordinates and associated local flow condition.

- **Number of points:** ≈ 8.2 million  
- **Region:** Horizontal Tail Plane (HTP)  
- **Flight Level:** FL310  
- **Input variables:** spatial coordinates (x, y, z), angle of attack (α), Mach number (M)  
- **Output variable:** pressure coefficient (Cp)

| Variable | Description | Units |
|-----------|--------------|--------|
| x, y, z | Cartesian coordinates of each surface node | m |
| α | Angle of Attack | deg |
| Mach | Mach number | – |
| Cp | Pressure coefficient at each node | – |

---

## Model Definition

The surrogate model is defined as a **Multi-Layer Perceptron (MLP)** using **PyLOM**, trained to map the relationship:

**Model mapping function:**

`f : (x, y, z, α, M) → Cp`


### Architecture Overview

<p align="center">
  <img src="../../images/mlp.png" alt="MLP Architecture for Cp Model" align="right" width="500">
</p>


| Parameter | Description | Example Value |
|------------|--------------|----------------|
| `input_size` | Number of input features | 5 |
| `output_size` | Target variable dimension | 1 |
| `n_layers` | Number of hidden layers | 4 |
| `n_neurons` | Neurons per layer | 252 |
| `dropout` | Dropout probability | 0.2 |
| `optimizer` | Optimization algorithm | Adam |
| `loss_fn` | Loss function | MSELoss |
| `learning_rate` | Learning rate | 8.1915e-4 |

All parameters are defined in the `SF_Model_training` file, while `flow_variables.json` contains the configuration of inputs and outputs.


---

## Training Procedure

Training follows the Surrogate Factory pipeline:

1. **Dataset preparation:** loading, scaling, and tensor conversion using PyLOM utilities.  
2. **Model initialization:** parameters imported from configuration JSON.  
3. **Training phase:** conducted on GPU using the Adam optimizer and a learning rate scheduler.  
4. **Validation:** statistical and visual comparison with ground truth CFD data.  

The model is trained for multiple combinations of **Mach** and **Angle of Attack**, enabling generalization across different aerodynamic regimes.

---

## Validation and Performance Assessment

Model validation is performed using **NGSValidation**, focusing on pointwise errors between predicted and CFD-derived Cp values.  

### Statistical Evaluation
- **Metrics:** mean, median, standard deviation, IQR, skewness, kurtosis.  
- **Percentiles:** 1%, 5%, 10%, 50%, 90%, 95%, 99%.  
- **Residual and absolute error distributions** are computed for statistical robustness.

### Visual Assessment
- Comparison of predicted and reference Cp distributions over the HTP.  
- Cumulative and histogram plots of the prediction errors.  
- Residual maps illustrating spatial deviations on the HTP surface.

<p align="center">
  <img src="../../images/cp_example.png" alt="Cp Distribution Example" width="45%">
  <img src="../../images/results.png" alt="CP prediction" width="40%">
</p>


---

## Results Summary

The trained MLP model achieves strong agreement with CFD data, providing accurate Cp predictions across the entire HTP mesh.

<p align="center">
  <img src="../../images/table.png" alt="HTP Cp Error Distribution" width="80%">
</p>

- Average prediction error below **2%** of the CFD Cp magnitude.  
- The model correctly captures the aerodynamic gradients induced by Mach and α variations.  
- Validation shows consistent accuracy across multiple flight conditions.

<p align="center">
  <img src="../../images/summary.png" alt="HTP Cp Error Distribution" width="80%">
</p>

---

## Conclusions

The Cp surrogate developed through the Surrogate Factory framework successfully reproduces the aerodynamic behavior of the HTP with high fidelity and computational efficiency.  
This workflow establishes a reliable foundation for future integration of surrogate models into **Flight Physics digital environments**, including optimization and design exploration loops.

---

## Author

**Marta A. Martín**  
Flight Physics – Technology Integration  
Airbus Defence and Space (Getafe, Spain)
