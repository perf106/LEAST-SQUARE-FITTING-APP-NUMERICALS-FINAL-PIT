import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Least Squares Fitting", layout="centered")
st.title("📉 Least Squares Fitting App")

st.markdown("Choose a fitting method, input x and y values, and press **Fit Curve**.")

# --- Method Selection ---
method = st.selectbox("Choose the fitting method:", ["Linear", "Quadratic", "Cubic"])

# --- Input Fields ---
x_input = st.text_area("Enter x values (comma-separated)", "1, 2, 3, 4, 5")
y_input = st.text_area("Enter y values (comma-separated)", "2, 4, 5, 4, 5")

# --- Fit Button ---
fit_button = st.button("📈 Fit Curve")

if fit_button:
    try:
        # --- Parse Inputs ---
        x = np.array([float(i.strip()) for i in x_input.split(',')])
        y = np.array([float(i.strip()) for i in y_input.split(',')])

        if len(x) != len(y):
            st.error("❗ x and y must have the same number of values.")
        else:
            # --- Perform Fitting ---
            if method == "Linear":
                degree = 1
            elif method == "Quadratic":
                degree = 2
            elif method == "Cubic":
                degree = 3

            coeffs = np.polyfit(x, y, degree)
            poly = np.poly1d(coeffs)
            y_pred = poly(x)

            # --- Show Equation ---
            st.subheader(f"📈 {method} Regression Equation")
            equation_latex = " + ".join([f"{round(c, 4)}x^{deg}" if deg > 1 else
                                         (f"{round(c, 4)}x" if deg == 1 else f"{round(c, 4)}")
                                         for deg, c in zip(range(degree, -1, -1), coeffs)])
            st.latex(f"\\hat{{y}} = {equation_latex}")

            # --- Results Table ---
            df = pd.DataFrame({
                "x": x,
                "Observed y": y,
                "Predicted ŷ": y_pred,
                "Error (y - ŷ)": y - y_pred
            })
            st.subheader("📊 Results Table")
            st.dataframe(df.style.format(precision=4), height=250)

            # --- Plot ---
            x_range = np.linspace(min(x), max(x), 300)
            y_range = poly(x_range)

            fig, ax = plt.subplots()
            ax.scatter(x, y, color='deepskyblue', label='Observed y')
            ax.plot(x_range, y_range, color='orange', label='Fitted Curve')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(f"{method} Least Squares Fitting")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
