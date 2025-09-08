# importing Librarirs--------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time 


# TIME MEASURING
start_time = time.time()

# Defining Variables--------------------------------------------------------------------------------------------------------------------------


    # Computational Constants
grid_size = 10 
Iteration_Number = 1000
Temperature_Steps = 100
T_i = 1
T_f = 10





    # Physical Constants
K_b = 1 #boltzman constant 
J = 1

J_1 = 1 * K_b
J_2 = 0.5 * K_b
J_3 = 0.2 * K_b
 

    # initial conditions

    # Global Variables

S = np.zeros([grid_size,grid_size])
T = np.zeros([Temperature_Steps])

E_T = np.zeros([Temperature_Steps])
E_t = np.zeros([Iteration_Number+1])

M_T = np.zeros([Temperature_Steps])
M_t = np.zeros([Iteration_Number+1])



# Defining Functions---------------------------------------------------------------------------------------------------------------------------------------




def Initial_Randoms():
    for i in range(grid_size):
        for j in range(grid_size):
            S[i][j] = (2*random.randint(0,1)) - 1

def Initial_Uniform(p):
        for i in range(grid_size):
            for j in range(grid_size):
                S[i][j] = p

def Element_Energy_Calculator(n,m):
    E = 0
    # definig vicinities

        #J_1-----------------------------------
    r = m + 1
    l = m - 1 
    u = n + 1
    d = n - 1 
    if r >= grid_size:
        r = 0
    if u >= grid_size:
        u = 0
    if l < 0:
        l = grid_size-1
    if d < 0:
        d = grid_size-1

    vicinity_1 = [[n,r], [m, u], [n, l], [m, d]]

            # Adding J_1 energies

    for v in vicinity_1:
        E += (-1) * (J_1) * S[v[0], v[1]] * S[n,m]



        #J_2----------------------------------
    rr = m + 2
    ll = m - 2
    uu = n + 2
    dd = n - 2
    if rr >= grid_size:
        rr = 0
    if uu >= grid_size:
        uu = 0
    if ll < 0:
        ll = grid_size-1
    if dd < 0:
        dd = grid_size-1


    vicinity_2 = [[n,rr], [m, uu], [n, ll], [m, dd]]

            # Adding J_2 energies
    for v in vicinity_2:
        E += (-1) * (J_2) * S[v[0], v[1]] * S[n,m]


        #J_3-----------------------


        vicinity_3 = [[u,r], [u, l], [d, l], [d, r]]
            # Adding J_3 energies
    for v in vicinity_3:
        E += (-1) * (J_3) * S[v[0], v[1]] * S[n,m]




    return E

def check_flip_condition(n,m, T):
    delta_E = (-2)*Element_Energy_Calculator(n,m)


    

    if (delta_E <= 0):
        return 1

    else:
        booltzman_factor = np.exp(-delta_E / (K_b * T))
        if booltzman_factor > 1:
            booltzman_factor = 1
        if random.random() < booltzman_factor :
            return 1
        else:
            return 0

def flip(i,j):
        S[i][j] = -S[i][j]


def Monte_Carlo(T):
    for i in range(grid_size):
        for j in range(grid_size):
            if check_flip_condition(i, j, T) == 1:
                flip(i, j)
            
def Magnetization_calculator():
    m = 0
    for i in range(grid_size):
        for j in range(grid_size):
            m += S[i][j]
    return m/(grid_size**2)
    

def Final_Mean_Magnetisation(): #Mean over last MonteCarlos in a specific temp.
    # M_bar = 0
    # for i in range(Iteration_Number - 10, Iteration_Number):
    #     M_bar += abs(M_t[i])
    # M_bar = M_bar/(10)
    # Last_Mean = M_t[Iteration_Number]

    return np.mean(M_t[(Iteration_Number-10):Iteration_Number])

# def Final_Mean_Energy(): #Mean over last MonteCarlos in a specific temp.
#     E_bar = 0
#     for i in range(int((9/10)*Iteration_Number), Iteration_Number):
#         E_bar += E_t[i]
#     E_bar = E_bar/((1/10)*Iteration_Number)
#     return E_bar


def Total_Energy_Calculator():
    E_total = 0
    for i in range(grid_size):
        for j in range(grid_size):
            E_total += Element_Energy_Calculator(i, j)
    return (E_total / 2)


def Variate_Temperature():

    for i in range (Temperature_Steps):
        T[i] = (T_i) + i*((T_f - T_i)/(Temperature_Steps))
        Constant_Temperatures(T[i])
        M_T[i] = Final_Mean_Magnetisation()
        E_T[i] = Total_Energy_Calculator()
        M_t.fill(0)
        E_t.fill(0)

    
def Constant_Temperatures(T):
        # print("I'm fired")
        M_t[0] = Magnetization_calculator()
        for i in range(1, Iteration_Number+1):
            Monte_Carlo(T)
            M_t[i] = Magnetization_calculator()
            E_t[i] = Total_Energy_Calculator()




# Fire Chain of Commands--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Initial_Randoms()
Initial_Uniform(1)


Variate_Temperature()






# Show Outputs
# ////////////////////////////////////////////////////
# Create a figure and axes
fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

# First plot: Magnetization over Temperature
axs[0].plot(T, M_T, linestyle='-', linewidth=1, marker='o',markersize=2, label='M(T)')  # Replace with your data
axs[0].set_title('Magnetization over Temperature')  # Set your title
axs[0].set_xlabel('T')  # Set your x-axis label
axs[0].set_ylabel('M')  # Set your y-axis label
axs[0].axhline(y=0, color='red', linestyle='--', linewidth=1, label='M=0')  # Add red line
axs[0].legend()  # Add legend for clarity






# Second plot: Energy over Temperature
axs[1].plot(T, E_T, linestyle='-', linewidth=1, marker='o',markersize=2, label='E(T)')  # Replace with your data
axs[1].set_title('Energy over Temperature')  # Set your title
axs[1].set_xlabel('T')  # Set your x-axis label
axs[1].set_ylabel('E')  # Set your y-axis label
axs[1].axhline(y=0, color='red', linestyle='--', linewidth=1, label='E=0')  # Add red line
axs[1].legend()  # Add legend for clarity


#Explanations
plt.subplots_adjust(right=1000)  # Adjust the right margin

#Numerical Constants Explained
textstr = f"""
    GridSize          =          {grid_size}*{grid_size} [#]
    d_T (Temperature Step Size)=          {(T_f-T_i)/Temperature_Steps} [Kelvin]
    N (Number of Montecarlos for a specifc Temperature)            =          {Iteration_Number} [#]

"""


props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# Place the text box in the lower right corner
# fig.text(0.5, 0.1, textstr, fontsize=8, bbox=props, ha='left')

plt.subplots_adjust(bottom=0.8)

fig.text(0.4, -0.02, textstr, 
          fontsize=5, bbox=dict(facecolor='lightgray', alpha=0.5))




# # All M-t in each temperture plot-------------------------------------------------
# fig, axs = plt.subplots(5, 2, figsize=(12, 6))  # 5 row, 2 columns
# tempo = 0
# for i in range(5):
#     for j in range(2):
#         Initial_Uniform(+1)
#         Constant_Temperatures(T[tempo])


#         axs[i][j].plot(M_t, linestyle='-', marker='o', markersize=0.1, linewidth=1)  # Replace with your data
#         axs[i][j].set_title(f'Magnetisation over time in Temperature: {T[tempo]}')  # Set your title
#         axs[i][j].set_xlabel('t')  # Set your x-axis label
#         axs[i][j].set_ylabel('M')  # Set your y-axis label
#         axs[i][j].set_ylim(-1.1, +1.1)

#         tempo += int(Temperature_Steps/10)

# # All E-t in each temperture plot-------------------------------------------------
# fig, axs = plt.subplots(5, 2, figsize=(12, 6))  # 5 row, 2 columns
# tempo = 0
# for i in range(5):
#     for j in range(2):
#         Initial_Uniform(+1)
#         Constant_Temperatures(T[tempo])



#         axs[i][j].plot(E_t, linestyle='-', marker='o', markersize=0.1, linewidth=1)  # Replace with your data
#         axs[i][j].set_title(f'Energy over time in Temperature: {T[tempo]}')  # Set your title
#         axs[i][j].set_xlabel('t')  # Set your x-axis label
#         axs[i][j].set_ylabel('E')  # Set your y-axis label
#         # axs[i][j].set_ylim(-1.1, +1.1)

#         tempo += int(Temperature_Steps/10)


# Different Sizes
# fig, axs = plt.subplots(5, 2, figsize=(12, 6))  # 5 row, 2 columns
# grid_size = 5
# for i in range(5):


#     Variate_Temperature()
#     # First plot: Magnetization over Temperature
#     axs[i][0].plot(T, M_T, linestyle='-', linewidth=1, marker='o',markersize=2, label='M(T)')  # Replace with your data
#     axs[i][0].set_title(f'M over Temperature in GridSize: {grid_size}*{grid_size}')  # Set your title
#     axs[i][0].set_xlabel('T')  # Set your x-axis label
#     axs[i][0].set_ylabel('M')  # Set your y-axis label
#     # Second plot: Energy Over temperature
#     axs[i][1].plot(T, E_T, linestyle='-', linewidth=1, marker='o',markersize=2, label='E(T)')  # Replace with your data
#     axs[i][1].set_title(f'E over Temperature in GridSize: {grid_size}*{grid_size}')  # Set your title
#     axs[i][1].set_xlabel('T')  # Set your x-axis label
#     axs[i][1].set_ylabel('E')  # Set your y-axis label
#     grid_size += 1





# Different J's
# fig, axs = plt.subplots(5, 2, figsize=(12, 6))  # 5 row, 2 columns

# for i in range(5):
#     J_2 -= 0.1
#     J_3 -= 0.04

#     Variate_Temperature()
#     # First plot: Magnetization over Temperature
#     axs[i][0].plot(T, M_T, linestyle='-', linewidth=1, marker='o',markersize=2, label='M(T)')  # Replace with your data
#     axs[i][0].set_title(f'M over Temperature in Couplings: J_2={J_2} & J_3={J_3}')  # Set your title
#     axs[i][0].set_xlabel('T')  # Set your x-axis label
#     axs[i][0].set_ylabel('M')  # Set your y-axis label
#     # Second plot: Energy Over temperature
#     axs[i][1].plot(T, E_T, linestyle='-', linewidth=1, marker='o',markersize=2, label='E(T)')  # Replace with your data
#     axs[i][1].set_title(f'E over Temperature in Couplings: J_2={J_2} & J_3={J_3}')  # Set your title
#     axs[i][1].set_xlabel('T')  # Set your x-axis label
#     axs[i][1].set_ylabel('E')  # Set your y-axis label










# Adjust layout
plt.tight_layout()

# /////////////////////////////////////////////////////////

    # Magnetisim over time
# plt.plot(M_t)




    # Magnetisim over Temperature
# plt.plot(T, M_T)




    #Energy over ??
# Max = max(M_t)
# Min = min(M_t)
# Interval = Max-Min
# Start = Min - Interval/10
# End = Max + Interval/10
# plt.ylim(-1.1, +1.1)
# plt.show()



# TIME MEASURING------------------------------------------------
end_time = time.time()
execution_time = end_time - start_time
minutes = int(execution_time // 60)
seconds = execution_time % 60
print(f"Total execution time: {minutes} minutes and {seconds:.2f} seconds")
#============================================================
plt.show()
