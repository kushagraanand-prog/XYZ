import streamlit as st
import numpy as np
import joblib
import lightgbm as lgb
from scipy.optimize import minimize

# -----------------------------
# Load models
# -----------------------------
model = lgb.Booster(model_file="lgb_model.txt")


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Smelting Process Target Optimizer (LightGBM)")
st.write("Fix some inputs, leave others empty, and the app will suggest values to achieve target Cu in CL slag.")

# Desired target
target_cu = float(st.text_input("Desired Cu in CL Slag (%)", '0.69'))

# Input variables
input_names = [
    "Cu","Fe","S","SiO2","Al2O3","CaO","MgO",
    "S_Cu_ratio","CONC_FEED_RATE","COAL_FEED_RATE","C_SLAG_FEED_RATE",
    "S_FURNACE_AIR","S_FURNACE_OXYGEN","S_MELT_TEMPERATURE","CL_SLAG_TEMPERATURE",
    "Fe_in_CL_slag","SiO2_in_CL_slag","CaO_in_CL_slag","Al2O3_in_CL_slag","MgO_in_CL_slag",
    "Fe_SiO2_RATIO","CLS_Fe3O4","S_outlet_Fe3O4","Cu_in_CL_matte","Cu_in_C_slag",
    "Fe_in_C_slag","CaO_in_C_slag","Fe3O4_in_C_slag","SILICA_FEED_RATE",'slag_ratio','O2_per_material','coal_per_material'
]

# Default guesses
default_values = [
    25,27,28,0.6,0.02,1.3,1.7,1.13,100,2,9,
    18000,170000,1200,1230,44,35,2.2,4.5,1.4,
    1.15,1.7,7,69,14,45,16,28,13
]

# Realistic bounds for each variable
variable_bounds = {
    "Cu": (20, 35),
    "Fe": (20, 35),
    "S": (20, 35),
    "SiO2": (4, 12),
    "Al2O3": (1, 3),
    "CaO": (0.15, 1.5),
    "MgO": (0.05, 1),
    "S_Cu_ratio": (0.5, 2.0),
    "CONC_FEED_RATE": (90, 112),
    "COAL_FEED_RATE": (0, 7),
    "C_SLAG_FEED_RATE": (1, 14),
    "S_FURNACE_AIR": (15000, 23000),
    "S_FURNACE_OXYGEN": (15000, 23000),
    "S_MELT_TEMPERATURE": (1190, 1230),
    "CL_SLAG_TEMPERATURE": (1180, 1260),
    "Fe_in_CL_slag": (34, 50),
    "SiO2_in_CL_slag": (34, 40),
    "CaO_in_CL_slag": (1, 6),
    "Al2O3_in_CL_slag": (3, 7),
    "MgO_in_CL_slag": (0.4, 2.5),
    "Fe_SiO2_RATIO": (0.9, 1.5),
    "CLS_Fe3O4": (1.2, 2.4),
    "S_outlet_Fe3O4": (3.5, 10.1),
    "Cu_in_CL_matte": (64, 72.5),
    "Cu_in_C_slag": (9, 28),
    "Fe_in_C_slag": (37, 49),
    "CaO_in_C_slag": (8, 26.1),
    "Fe3O4_in_C_slag": (23, 38),
    "SILICA_FEED_RATE": (1, 27.5),
    "slag_ratio": (0.4, 0.75),
    "O2_per_material": (182, 266),
    "coal_per_material": (0.08)
}

# Collect user inputs
user_inputs = []
free_indices = []
for i, name in enumerate(input_names):
    val = st.text_input(name, "")
    try:
        val_float = float(val)
        user_inputs.append(val_float)
    except:
        user_inputs.append(default_values[i])
        free_indices.append(i)

# -----------------------------
# Optimization
# -----------------------------
def objective(x_free):
    X_full = user_inputs.copy()
    for idx, val in zip(free_indices, x_free):
        X_full[idx] = val
    X_array = np.array([X_full])
    X_scaled = scaler.transform(X_array)
    y_pred = model.predict(X_scaled)[0]
    return (y_pred - target_cu)**2

if free_indices:
    x0 = [user_inputs[i] for i in free_indices]
    bounds = [variable_bounds[input_names[i]] for i in free_indices]
    res = minimize(objective, x0, bounds=bounds)
    for idx, val in zip(free_indices, res.x):
        user_inputs[idx] = val

# -----------------------------
# Display results
# -----------------------------
st.subheader("Fixed Inputs")
for i, name in enumerate(input_names):
    if i not in free_indices:
        st.write(f"{name}: {user_inputs[i]:.2f} (fixed)")

st.subheader("Optimized (Free) Inputs")
for i in free_indices:
    st.write(f"{input_names[i]}: {user_inputs[i]:.2f} (optimized)")

X_final = np.array([user_inputs])
X_scaled = scaler.transform(X_final)
cu_slag_pred = model.predict(X_scaled)[0]

st.subheader("Predicted Cu in Slag with optimized inputs")
st.write(f"🔹 Cu in Slag: {cu_slag_pred:.2f}%")
