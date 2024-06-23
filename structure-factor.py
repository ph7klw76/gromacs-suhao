import numpy as np
import matplotlib.pyplot as plt

# Load the RDF data
file_path = '/mnt/data/rdf_hhpa.xvg'
rdf_data = np.loadtxt(file_path, comments=['#', '@'])

# Extract r and g(r) from the data
r = rdf_data[:, 0]
g_r = rdf_data[:, 1]

# Define the function to compute the structure factor with correct handling of r=0
def compute_structure_factor_corrected(r, g_r, rho):
    q_values = np.linspace(0.1, 15, 500)  # scattering vector range
    S_q = np.zeros_like(q_values)

    for i, q in enumerate(q_values):
        integrand = np.where(r != 0, r**2 * (g_r - 1) * np.sin(q * r) / (q * r), 0)
        S_q[i] = 1 + 4 * np.pi * rho * np.trapz(integrand, r)
    
    return q_values, S_q

# Set the number density rho (assuming it is provided or we have a reasonable estimate)
rho = 1.0  # Example value, this should be adjusted according to the specific system

# Compute the structure factor S(q)
q_values, S_q = compute_structure_factor_corrected(r, g_r, rho)

# Compute the XRD intensity I(q)
I_q = np.abs(S_q)**2

# Plotting the RDF, Structure Factor, and XRD pattern
plt.figure(figsize=(18, 6))

# Plot RDF
plt.subplot(1, 3, 1)
plt.plot(r, g_r, label='g(r)')
plt.xlabel('r (nm)')
plt.ylabel('g(r)')
plt.title('Radial Distribution Function')
plt.legend()

# Plot Structure Factor
plt.subplot(1, 3, 2)
plt.plot(q_values, S_q, label='S(q)')
plt.xlabel('q (nm^-1)')
plt.ylabel('S(q)')
plt.title('Structure Factor')
plt.legend()

# Plot XRD Intensity
plt.subplot(1, 3, 3)
plt.plot(q_values, I_q, label='I(q)')
plt.xlabel('q (nm^-1)')
plt.ylabel('I(q)')
plt.title('XRD Intensity')
plt.legend()

plt.tight_layout()
plt.show()

# Display the results in a DataFrame
import pandas as pd
import ace_tools as tools

data = {
    "q_values (nm^-1)": q_values,
    "S_q": S_q,
    "I_q": I_q
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the DataFrame to the user
tools.display_dataframe_to_user(name="Computed Structure Factor and XRD Intensity", dataframe=df)
