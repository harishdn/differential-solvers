import numpy as np
import matplotlib.pyplot as plt
import time

# Define the potential function V(x) and its derivative dV(x)
def V(x):
    return 0.5 * mass * x**2  # Example potential function V(x) = 0.5 * x^2

def dV(x):
    return mass * x  # Derivative of V(x) = x

# Define the system of differential equations
def system(t, state):
    x, y, h = state
    
    # dx/dt = y
    dxdt = y
    
    # dy/dt = -(3 -0.5*y**2 ) * y + (1 / h^2) * dV(x)
    dydt = -(3 - 0.5 * y**2) * y - (dV(x) / h**2) 
    
    # dh/dt = -0.5 * h * y^2
    dhdt = -0.5 * h * y**2
    
    # Check for overflow or extreme values
    if np.any(np.isnan([dxdt, dydt, dhdt])) or np.any(np.isinf([dxdt, dydt, dhdt])):
        print(f"Warning: Invalid value encountered at t={t}")
        return np.array([0, 0, 0])  # Return a zero vector to stop further calculations
    
    return np.array([dxdt, dydt, dhdt])

# Implement the RK2 method
def rk2_method(system, t0, t_end, dt, initial_state):
    # Initialize the time array and state array
    t_values = np.arange(t0, t_end, dt)
    state = initial_state
    states = []
    
    for t in t_values:
        states.append(state)
        
        # Compute k1 and k2 for the RK2 method
        k1 = system(t, state)
        k2 = system(t + dt, state + dt * k1)
        
        # Update the state
        state = state + dt * 0.5 * (k1 + k2)
        
        # Check if the state goes to NaN or infinity
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            print(f"Warning: Integration stopped due to instability in rk-2 at t={t}")
            break  # Stop integration if instability is detected
    
    return t_values, np.array(states)

# Implement the RK4 method
def rk4_method(system, t0, t_end, dt, initial_state):
    # Initialize the time array and state array
    t_values = np.arange(t0, t_end, dt)
    state = initial_state
    states = []
    
    for t in t_values:
        states.append(state)
        
        # Compute k1, k2, k3, and k4 for the RK4 method
        k1 = system(t, state)
        k2 = system(t + 0.5 * dt, state + 0.5 * dt * k1)
        k3 = system(t + 0.5 * dt, state + 0.5 * dt * k2)
        k4 = system(t + dt, state + dt * k3)
        
        # Update the state
        state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # Check if the state goes to NaN or infinity
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            print(f"Integration stopped due to instability in rk-4 at t={t}")
            break  # Stop integration if instability is detected
    
    return t_values, np.array(states)


# Parameters
mass = 1
t0 = 0
t_end = 70
dt = 0.01  # Use a smaller time step to improve stability
phi0 = 16.5
y0 = -dV(phi0) / V(phi0)
h0 = np.sqrt(V(phi0) / (3 - 0.5 * (y0)**2))
initial_state = np.array([phi0, y0, h0])  # Initial conditions

#Start the timer before running the RK2 method
ti = time.time()

# Solve using RK2 method
t_values_rk2, states_rk2 = rk2_method(system, t0, t_end, dt, initial_state)

# End the timer after the method execution
tf = time.time()

# Print the execution time
print("Code execution time for rk-2 " + str(t_end) + " e-folds with step size of " + str(dt) + ": " + str(tf - ti) + " seconds.")

#Start the timer before running the RK2 method
ti = time.time()

# Solve using RK2 method
t_values_rk4, states_rk4 = rk4_method(system, t0, t_end, dt, initial_state)

# End the timer after the method execution
tf = time.time()

# Print the execution time
print("Code execution time for rk-4 " + str(t_end) + " e-folds with step size of " + str(dt) + ": " + str(tf - ti) + " seconds.")

# Set up the figure with the correct size
plt.figure(figsize=(8, 14))

# Plot x, y, and h
# First subplot
plt.subplot(3, 1, 1)  # Specify rows=2, columns=1, and subplot index=1
plt.plot(t_values_rk2, states_rk2[:, 0], label='$\phi(N)$ (RK2)', color='blue')
plt.plot(t_values_rk4, states_rk4[:, 0], label='$\phi(N)$ (RK4)', color='red', linestyle='--')
plt.ylabel('$\phi$')
plt.legend()

# Second subplot
plt.subplot(3, 1, 2)  # Specify rows=2, columns=1, and subplot index=2
plt.plot(t_values_rk2, states_rk2[:, 2], label='H (RK2)', color='green')
plt.plot(t_values_rk4, states_rk4[:, 2], label='H (RK4)', color='black', linestyle='--')
# Add labels and legend
plt.xlabel('N')
plt.ylabel('H')
plt.legend()

plt.subplot(3, 1, 3)  # Specify rows=2, columns=1, and subplot index=2
plt.plot(t_values_rk2, 0.5*(states_rk2[:, 1])**2, label='$\epsilon_1$ (RK2)', color='red')
plt.plot(t_values_rk4, 0.5*(states_rk4[:, 1])**2, label='$\epsilon_1$ (RK4)', color='blue', linestyle='--')
plt.plot(t_values_rk2, np.gradient(0.5*(states_rk2[:, 1])**2,t_values_rk2), label='$\epsilon_2$ (RK2)', color='orange')
plt.plot(t_values_rk4, np.gradient(0.5*(states_rk4[:, 1])**2,t_values_rk4), label='$\epsilon_2$ (RK4)', color='purple', linestyle='--')
plt.axhline(1, color='black', linewidth=1, linestyle='--')
plt.axvline(68.5, color='Black', linewidth=1, linestyle='--')
plt.ylabel('$\epsilon_1, \epsilon_2$')
#plt.xlim(60,70)

# plt.xlim(0, 70)  # Uncomment if you need to limit x-axis range
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.legend()

# Show the plot
plt.tight_layout()  # Adjust subplots to fit nicely
plt.show()

# Show the plot
plt.tight_layout()  # Adjust subplots to fit nicely
plt.show()
