# ENTS656 Cellular Project
# Project Title: Downlink Behavior of a 3-sectored base-station
# Author: Pranav H. Deo
# Start Date: 11/12/20
# End Date: 12/02/20
# 2020, the year of COVID and American Election.

import numpy as np                                              # Without this God knows what would've happened
import matplotlib.pyplot as plt                                 # A plot saves lives

################################################### GIVEN PARAMETERS ###################################################

# Properties of a Base-station (Given):
Road_Length = 6000                                              # Length of the Road is 6km = 6000m
Height_BS = 50                                                  # Height of the Base-station = 50m
BS_Distance_from_Road = 20                                      # 20m West (Right) of the Road
BS_Location = Road_Length / 2                                   # Location: Road Length / 2
Power_TX = 43                                                   # P_TX = 43dBm
Line_Loss = 2                                                   # Loss = 2dB
Gain_TX = 14.8                                                  # G_TX = 14.8 dBi
Total_Sectors = 3                                               # Given 3 sectored Base-station
Traffic_Ch_per_Sector = 15                                      # Ch/sector = 15
Traffic_Ch_Sector_Alpha = 15                                    # Channels in Sector Alpha = Ch/Sector
Traffic_Ch_Sector_Beta = 15                                     # Channels in Sector Beta = Ch/Sector
Traffic_Ch_Sector_Alpha_counter = 0                             # Counter that keeps track of channels in Sector Alpha under usage
Traffic_Ch_Sector_Beta_counter = 0                              # Counter that keeps track of channels in Sector Beta under usage
Total_Traffic_Ch = Total_Sectors * Traffic_Ch_per_Sector        # Total Channels = 3 * 15 = 45
Available_Traffic_Ch = 2 * Traffic_Ch_per_Sector                # Since we are only going to consider alpha and beta sectors
F_MHZ_alpha = 860                                               # Frequency Alpha Sector = 860 MHz
F_MHZ_beta = 870                                                # Frequency Beta Sector = 870 MHz

# Properties of a Mobile (Given):
Height_Mobile = 1.5                                             # Height of a mobile = 1.5m
HOm = 0                                                         # HandOff Margin = 3dB (initially set)
RSL_Thresh = -102                                               # Mobile RSL Threshold = -102dBm

# User Properties (Given):
Total_Users = 320                                               # Total Users = 160 (Initial value)
Lambda = 2                                                      # 2 calls/hr
Delta_T = 1                                                     # Simulation Step-Size in secs
H = 3/60                                                        # Call Duration: 3min/call = 0.05hr/call = 180s/call
User_Speed_V = 15                                               # Speed of the User on Road: 15m/s
User_Moving_Direction = ['NORTH', 'SOUTH']                      # Direction the User is moving


################################################# FUNCTION DEFINITIONS #################################################


#   *************************** 1. Propagation Loss ***************************
## Function Parameters: frequency 'f'(MHz), Base-station Height 'h_te'(m), Receiver Height 'h_re'(m) and distance 'd'(m)
def PL_Okamura_Hata_Model(f, h_te, h_re, d):
    a_h_re = (1.1*np.log10(f)-0.7)*h_re - (1.56*np.log10(f)-0.8)                                # Correction Factor a(h_re) in dB for small city
    Okamura_Hata_Urban_PL = 69.55 + 26.16 * np.log10(f) - 13.82 * np.log10(h_te) \
                            - a_h_re + ((44.9 - 6.55 * np.log10(h_te)) * np.log10(d/1000))      # Computing Okamura-Hata Path_Loss for Urban Areas
    return Okamura_Hata_Urban_PL



#   *************************** 2. Shadowing Loss ***************************
# List which stores the shadowing value using the log-normal distribution
Shadowing_LookUp_Table = []
def Compute_Shadowing():                                # This function computes shadowing beforehand with step-size of 10m until Road-length
    global Shadowing_LookUp_Table
    N = int(Road_Length / 10)
    Shadowing_LookUp_Table = np.random.normal(0, 2, N)  # Given: MU=0 and Sigma=2. Computing log-normal distribution

def Shadowing(d):                                       # Function that looks-up the value of Shadowing from the Shadowing Look-Up Table
    dist = int(d) - d                                   # Getting the integer deviation from the real value
    if dist <= -0.5:                                    # If distance difference <= -0.5 then we ceil else we floor
        dist = np.ceil(d)                               # Using the ceil function to get upper bound
        dist = int(dist / 10)                           # Values within the range of 10 should produce the same values. For distance: 1940 and 1947 the value from the shadowing table should be same
    else:
        dist = np.floor(d)                              # Using the floor function to get the lower bound
        dist = int(dist / 10)                           # Values within the range of 10 should produce the same values. For distance: 1940 and 1947 the value from the shadowing table should be same
    return Shadowing_LookUp_Table[dist]                 # Extract Fading value for that distance



#   *************************** 3. Fading Loss ***************************
def Fading():
    Rayleigh = []                               # Rayleigh list to store all the Rayleigh Distribution
    real_vals = np.random.normal(0, 1, 10)      # Real-part
    imag_vals = np.random.normal(0, 1, 10)      # Imaginary-part
    Equation = real_vals + imag_vals * 1j       # Equation = real + imaginary
    Eqn_mag = np.abs(Equation)                  # Getting the magnitude of the equation
    Rayleigh.append(20 * np.log10(Eqn_mag))     # Add magnitude value to Rayleigh list in dB
    Rayleigh = np.sort(Rayleigh)                # Sorting the Rayleigh list
    Second_Deepest_Fade = Rayleigh[0][1]        # Pulling out the 2nd deepest fade
    return Second_Deepest_Fade                  # Return the 2nd Deepest Fade



# *************************** 4. Total Path Loss ***************************
## Function Parameters: frequency 'f'(MHz), Base-station Height 'H_Te'(m), Receiver Height 'H_Re'(m) and distance 'Dist_Mobile_to_BS'(m)
def Total_Path_Loss(f, H_Te, H_Re, Dist_Mobile_to_BS):
    propagation_Loss = PL_Okamura_Hata_Model(f, H_Te, H_Re, Dist_Mobile_to_BS)         # Calling the Okamura-Hata Propagation_Loss Function
    # print('Propagation Loss:', Propagation_Loss)
    shadowing = Shadowing(Dist_Mobile_to_BS)                                           # Calling the Shadow Loss function with distance as parameter
    # print('Shadow Loss:', Shadow_Loss)
    fading = Fading()                                                                  # Calling the Fading Loss function
    # print('Fading Loss:', Loss_Fading)
    Total_PL = propagation_Loss - shadowing - fading                                   # Total Path Loss = Propagation - Shadowing - Fading
    return Total_PL



# *************************** 5. Boresight Angle ***************************
def Get_BoreSight_Angle(Loc, BS_sector):                                            # Function to compute Bore-sight angle
    U = [BS_Distance_from_Road, Loc - BS_Location]                                  # Generating the U vector = [20, User Location-Base-station Location]
    if BS_sector == 'alpha':                                                        # If the Base-station sector is 'Alpha',
        V = [0, 1]                                                                  # then V = [0, 1]
    else:                                                                           # If the Base-station sector is 'Beta',
        V = [(np.sqrt(3))/2, -(1/2)]                                                # then, V = [root(3)/2, -(1/2)]
    theta = np.dot(U, V) / ((np.linalg.norm(U))*(np.linalg.norm(V)))                # Computing the dot product (U.V) / (||U||*||V||)
    theta_degrees = int(np.degrees(np.arccos(theta)))                               # Converting radians to degrees and then to integer format
    return theta_degrees                                                            # return theta computed in degrees



# *************************** 6. EIRP and RSL ***************************
EIRP_BoreSight = Power_TX + Gain_TX - Line_Loss                                                         # EIRP at Bore-sight
def Compute_RSL(User_Loc, Dist_Mobile_to_BS, Sector):                                                   # Function which computes the RSL for Sector 'Alpha' and 'Beta'
    global Antenna_Pattern                                                                              # Getting a Global Dictionary for reference
    if Sector == 'alpha':                                                                               # For Sector Alpha
        #print('ALPHA:')
        Path_Loss_Alpha = Total_Path_Loss(F_MHZ_alpha, Height_BS, Height_Mobile, Dist_Mobile_to_BS)     # Call made to the function: Total_Path_Loss
        Angle_Alpha = Get_BoreSight_Angle(User_Loc, 'alpha')                                            # Call made to the function: Get_BoreSight_Angle to get the angle between the Antenna and Mobile
        Alpha_Antenna_Discrimination = float(Antenna_Pattern[format(float(Angle_Alpha), '.2f')])        # Finding Angle (in degrees) from the Antenna_Pattern Dictionary
        RSL = EIRP_BoreSight - Alpha_Antenna_Discrimination - Path_Loss_Alpha                           # Computing the Alpha_RSL = EIRP_BoreSight - Discrimination_Sector_Alpha - Path_Loss_Sector_Alpha
    else:                                                                                               # For Sector Beta
        #print('BETA:')
        Path_Loss_Beta = Total_Path_Loss(F_MHZ_beta, Height_BS, Height_Mobile, Dist_Mobile_to_BS)       # Call made to the function: Total_Path_Loss
        Angle_Beta = Get_BoreSight_Angle(User_Loc, 'beta')                                              # Call made to the function: Get_BoreSight_Angle to get the angle between the Antenna and Mobile
        Beta_Antenna_Discrimination = float(Antenna_Pattern[format(float(Angle_Beta), '.2f')])          # Finding Angle (in degrees) from the Antenna_Pattern Dictionary
        RSL = EIRP_BoreSight - Beta_Antenna_Discrimination - Path_Loss_Beta                             # Computing the Beta_RSL = EIRP_BoreSight - Discrimination_Sector_Beta - Path_Loss_Sector_Beta
    return RSL



# *************************** 7. Reading Antenna Pattern File ***************************
def Read_AntennaPattern_File(fpath):        # We want to create a Dictionary from the Antenna-Pattern.txt
    Dct = {}                                # Initialize a dictionary
    f = open(fpath, 'r')                    # Open the file in read mode
    for line in f:
        (key, value) = line.split()         # For every line in the file we split it into [Key, Value] pair
        Dct[key] = value                    # Assign the Keys their values
    return Dct                              # Returning the Dictionary to be used



# *************************** 8. PLACING NEW CALL ***************************
# When a new user with no active call, enters the range, he places a call
def New_Call_Placing(Users):
    global All_User_Records
    global Traffic_Ch_Sector_Alpha_counter
    global Traffic_Ch_Sector_Beta_counter
    global Total_Active_Users
    global Active_Users_Alpha
    global Active_Users_Beta
    global FOS_BS_Alpha
    global FOS_BS_Beta
    global Call_Attempts_Alpha
    global Call_Attempts_Beta
    global Alpha_Success_Call_Conn_counter
    global Beta_Success_Call_Conn_counter
    global Alpha_Call_Drop_counter
    global Beta_Call_Drop_counter
    global Alpha_Low_Capacity_counter
    global Beta_Low_Capacity_counter

    for User in range(1, Users+1):                                                                  # For every User from the given no. of Users
        if User in Total_Active_Users:                                                              # If the User has an Active Call,
            continue                                                                                # Go onto the next User
        else:                                                                                       # If the User does not have a call, he may initiate a call
            random_probability = np.random.random_sample()                                          # Selecting a random sample to compare with the User Call Placing Probability
            user_probability_to_make_call = (Lambda/3600) * Delta_T                                 # Prob(User Places a Call) = 2/3600 * Delta_T

            if user_probability_to_make_call >= random_probability:                                 # Call will be made if the User_probability >= Random sample
                #print('> User '+str(User)+' has Initiated a Call...')
                Sector = ''                                                                         # Assign a null Sector initially to that User
                RSL = 0                                                                             # Initialize the RSL as 0 for that User
                Call_Duration = 0                                                                   # Initialize Call Duration as 0 for that User
                Call_Status = ''                                                                    # Token to keep track of a successful, failed or a blocked call
                Call_Record_Type = ''                                                               # For Future use for marking Call as 'SUCCESS' or 'FAILURE'
                HandOff_Status = 'HANDOFF:NONE'                                                     # For Future use in stating if the Handoff was 'HANDOFF-SUCCESS' or 'HANDOFF-FAILURE'
                User_Features = []                                                                  # Stores User Location, Direction, Distance etc
                User_Location = np.random.uniform(0, Road_Length)                                   # Choosing a random value for distance from range 0-6000 'm'
                User_Direction = np.random.choice(User_Moving_Direction)                            # Choosing a random direction from the List User_Moving_Direction

                if (User_Location < (BS_Location)) or (User_Location > (BS_Location)):              # If the User is not directly infront of the BS
                    User_Distance = np.sqrt(np.square(BS_Distance_from_Road)
                                    + np.square(BS_Location - User_Location))                       # Computing the Distance from the Mobile to BS
                else:
                    User_Distance = BS_Distance_from_Road                                           # When the User is directly infront of the BS

                SectorA_RSL = Compute_RSL(User_Location, User_Distance, 'alpha')                    # Compute the RSL for Sector 'Alpha'
                SectorB_RSL = Compute_RSL(User_Location, User_Distance, 'beta')                     # Compute the RSL for Sector 'Beta'
                #print('RSL:', SectorA_RSL, ' ', SectorB_RSL)

                if SectorA_RSL > SectorB_RSL:                                                       # Condition to check if Sector 'Alpha' has highest RSL
                    #print('Chosen: Sector A')
                    Call_Attempts_Alpha += 1                                                        # Increment the Call Attempt counter for Sector 'Alpha'
                    Sector = 'Alpha'                                                                # Allot the Sector as the Serving Sector
                    RSL = SectorA_RSL                                                               # Allot that particular sector's RSL
                    if SectorA_RSL < RSL_Thresh:                                                    # Checking if RSL of the Serving Sector 'Alpha' < Required Threshold to place a call
                        #print('Call Dropped....', SectorA_RSL, '<', RSL_Thresh)
                        Call_Status = 'DROPPED:Low_RSL'                                             # Status changed to Dropped, since we cannot start/serve a call with low signal strength
                        Alpha_Call_Drop_counter += 1                                                # If RSL < RSL_Thresh then mark that as a failure for that sector
                    else:                                                                           # Checking if RSL from serving sector 'Alpha' >= Required Threshold to place a call
                        #print('Call Successful....', SectorA_RSL, '>', RSL_Thresh)
                        if 0 <= Traffic_Ch_Sector_Alpha_counter < 15:                               # If channel on Sector 'Alpha' is available then establish the call
                            #print('Channels Available...Allocated')
                            Call_Duration = int(np.random.exponential(180))                         # The duration of the call being made
                            Call_Status = 'ACTIVE:Established'                                      # Change Call Status to 'Call Established'
                            Traffic_Ch_Sector_Alpha_counter += 1                                    # Increment the Channel under Usage counter on Sector 'Alpha'
                            Alpha_Success_Call_Conn_counter += 1                                    # Count the 'Alpha' successful Call Connections
                        else:
                            #print('Channels Unavailable on Sector A...Checking with Sector B')      # In case Sector 'Alpha' has no channels, we check with Sector 'Beta'
                            if SectorB_RSL > RSL_Thresh:                                            # Only if Sector 'Beta' RSL > Threshold
                                Call_Attempts_Beta += 1                                             # Increment the Call Attempt counter for Sector 'Beta'
                                #print('Sector B helping...')
                                if 0 <= Traffic_Ch_Sector_Beta_counter < 15:                        # Channel counter checks if channels are available for allocation
                                    Sector = 'Beta'                                                 # If available, the serving sector is 'Beta'
                                    RSL = SectorB_RSL                                               # Sector 'Beta' RSL is recorded
                                    #print('Channels Available...Allocated')
                                    Call_Duration = int(np.random.exponential(180))                 # The duration of the call being made
                                    Call_Status = 'ACTIVE:Established'                              # Call Status Updated to ACTIVE:Established
                                    Traffic_Ch_Sector_Beta_counter += 1                             # Increment the Channel counter on Sector 'Beta'
                                    Beta_Success_Call_Conn_counter += 1                             # Count the 'Beta' successful Call Connections
                                else:
                                    #print('No Channels available on Sector A and B')                # If no channels on Sector 'Beta', we penalize Sector 'Alpha'
                                    Call_Status = 'BLOCKED:No_Channels'                             # Call Status Updated to BLOCKED:No_Channels
                                    Alpha_Low_Capacity_counter += 1                                 # Counts when a user is not allocated a channel


                elif SectorB_RSL > SectorA_RSL:                                                     # Condition to check if Sector 'Beta' has highest RSL
                    #print('Chosen: Sector B')
                    Call_Attempts_Beta += 1                                                         # Increment the Call Attempt counter for Sector 'Beta'
                    Sector = 'Beta'                                                                 # Allot the sector which might be the serving sector
                    RSL = SectorB_RSL                                                               # Allot that particular sector's RSL
                    if SectorB_RSL < RSL_Thresh:                                                    # Checking if RSL from serving sector 'Beta' < Required Threshold to place a call
                        #print('Call Dropped....', SectorB_RSL, '<', RSL_Thresh)
                        Call_Status = 'DROPPED:Low_RSL'                                             # Status changed to Dropped, since we cannot serve a call with low signal strength
                        Beta_Call_Drop_counter += 1                                                 # If RSL < RSL_Thresh then mark that as a failure for that sector
                    else:                                                                           # Checking if RSL from serving sector 'Beta' >= Required Threshold to place a call
                        #print('Call Successful....', SectorB_RSL, '>', RSL_Thresh)
                        if 0 <= Traffic_Ch_Sector_Beta_counter < 15:                                # If channel on Sector 'Beta' is available then establish the call
                            #print('Channels Available...Allocated')
                            Call_Duration = int(np.random.exponential(180))                         # The duration of the call being made
                            Call_Status = 'ACTIVE:Established'                                      # Change token value to 'Call Established'
                            Traffic_Ch_Sector_Beta_counter += 1                                     # Increment the Channel under Usage counter on Sector 'Beta'
                            Beta_Success_Call_Conn_counter += 1                                     # Count the 'Beta' successful Call Connections
                        else:
                            #print('Channels Unavailable on Sector B...Checking with Sector A')      # In case Sector 'Beta' has no channels, we check with Sector 'Alpha'
                            if SectorA_RSL > RSL_Thresh:                                            # Only if Sector 'Alpha' RSL > Threshold
                                Call_Attempts_Alpha += 1                                            # Increment the Call Attempt counter for Sector 'Alpha'
                                #print('Sector A helping...')
                                if 0 <= Traffic_Ch_Sector_Beta_counter < 15:                        # Channel counter checks if channels are available for allocation
                                    Sector = 'Alpha'                                                # If available, the serving sector is 'Alpha'
                                    RSL = SectorA_RSL                                               # Sector 'Alpha' RSL is recorded
                                    #print('Channels Available...Allocated')
                                    Call_Duration = int(np.random.exponential(180))                 # The duration of the call being made
                                    Call_Status = 'ACTIVE:Established'                              # Call Status Updated to ACTIVE:Established
                                    Traffic_Ch_Sector_Alpha_counter += 1                            # Increment the Channel counter on Sector 'Alpha'
                                    Alpha_Success_Call_Conn_counter += 1                            # Count the 'Alpha' successful Call Connections
                                else:
                                    #print('No Channels available on Sector A and B')                # If no channels on Sector 'Alpha', we penalize Sector 'Beta'
                                    Call_Status = 'BLOCKED:No_Channels'                             # Call Status Updated to BLOCKED:No_Channels
                                    Beta_Low_Capacity_counter += 1                                  # Counts when a user is not allocated a channel


                if Call_Status != '':
                    User_Features.append(User_Location)                                             # Appending the User Location [in 'm']
                    User_Features.append(User_Direction)                                            # Appending the User Direction [NORTH or SOUTH]
                    User_Features.append(User_Distance)                                             # Appending the User Distance from the Base-Station [in 'm']
                    User_Features.append(Sector)                                                    # Appending the Serving Sector for that User
                    User_Features.append(RSL)                                                       # Appending the RSL for the Sector chosen
                    User_Features.append(Call_Status)                                               # Appending the User's Call Status ['ACTIVE:Established', 'BLOCKED:No_Channels' or FAILED:Low_RSL']
                    User_Features.append(Call_Duration)                                             # Appending the User's Call Duration (in seconds)
                    User_Features.append(Call_Record_Type)                                          # Appending the User's Call Completion Status ('Success' / 'Failure')
                    User_Features.append(HandOff_Status)                                            # Appending the User's Handoff Status ('HANDOFF-Success' / 'HANDOFF-Failure')
                    All_User_Records[User] = User_Features                                          # Insert the entire Feature List as a value to the User key

                    if Call_Status == 'ACTIVE:Established' and Sector == 'Alpha':                   # Keeping track of all Active Users and Alpha Sector keeps a track of it's own active users
                        Total_Active_Users[User] = User_Features
                        Active_Users_Alpha[User] = User_Features
                    elif Call_Status == 'ACTIVE:Established' and Sector == 'Beta':                  # Keeping track of all Active Users and Beta Sector keeps a track of it's own active users
                        Total_Active_Users[User] = User_Features
                        Active_Users_Beta[User] = User_Features
                    elif (Call_Status != 'ACTIVE:Established') and Sector == 'Alpha':
                        FOS_BS_Alpha[User] = User_Features                                          # Alpha Sector keeps a track of Failures (Failure of Service) due to insufficient Channels or Low RSL
                    elif (Call_Status != 'ACTIVE:Established') and Sector == 'Beta':
                        FOS_BS_Beta[User] = User_Features                                           # Beta Sector keeps a track of Failures (Failure of Service) due to insufficient Channels or Low RSL



# *************************** 9. SERVING EXISTING CALL ***************************
# For those Users who are currently engaged in a call, we provide service
def Serve_Active_Users(Users):
    # From the Data Structure Records pull out the required information for determination
    global Total_Active_Users
    global Call_Archive
    global Traffic_Ch_Sector_Alpha_counter
    global Traffic_Ch_Sector_Beta_counter
    global FOS_BS_Alpha
    global FOS_BS_Beta
    global Call_Attempts_Alpha
    global Call_Attempts_Beta
    global Alpha_Success_Call_Conn_counter
    global Beta_Success_Call_Conn_counter
    global Alpha_Call_Drop_counter
    global Beta_Call_Drop_counter
    global Alpha_HO_Attempts
    global Alpha_HO_Success
    global Alpha_HO_Failure
    global Beta_HO_Attempts
    global Beta_HO_Success
    global Beta_HO_Failure
    global Success_Calls_Alpha
    global Success_Calls_Beta
    global Alpha_Low_Capacity_counter
    global Beta_Low_Capacity_counter

    for i in range(1, Users+1):
        if i in Total_Active_Users:                                             # If the User is an Active User, we pull out that User's Active records
            #print('User ', i,': ', Total_Active_Users[i], ' ')
            User_Loc = Total_Active_Users[i][0]                                 # Pulling out User's current Location
            User_Dir = Total_Active_Users[i][1]                                 # Pulling out User's current Direction of Motion
            User_Dist = Total_Active_Users[i][2]                                # Pulling out User's Distance from the Base-Station
            User_Sector = Total_Active_Users[i][3]                              # Pulling out the current serving sector for the User
            User_RSL = Total_Active_Users[i][4]                                 # Pulling out the User's current RSL
            Call_Status = Total_Active_Users[i][5]                              # Pulling out User's Call Status
            Call_Duration = Total_Active_Users[i][6]                            # Pulling out User's Call Duration / Call-Time left
            Call_Record_Type = Total_Active_Users[i][7]                         # Pulling out User's Call Record Type (SUCCESS/FAILURE/NONE)
            HandOff_Status = Total_Active_Users[i][8]                           # Pulling out User's Hand-Off Status

            if Call_Status == 'ACTIVE:Established' and Call_Duration > 0:       # Check if the User has used up all his talking minutes
                Total_Active_Users[i][6] = Call_Duration - 1                    # Decrement the Call Duration by 1 for the User if the Call is Active and Established
            elif Call_Status == 'ACTIVE:Established' and Call_Duration == 0:    # Checking if the User has no Call-Time left
                #print('User Serviced and Done: Call Time Exhausted')
                Call_Record_Type = 'SUCCESS'                                    # Mark the Call as a SUCCESS
                Total_Active_Users[i][7] = Call_Record_Type                     # Make an entry about this Successful call in the Total Active User Records
                Call_Archive[i] = Total_Active_Users[i]                         # Add the Call and the User Details to a Call Archive for Records
                del Total_Active_Users[i]                                       # Remove the User from the among the Active Users
                if User_Sector == 'Alpha':                                      # Checking if the Sector is Alpha or Beta
                    Traffic_Ch_Sector_Alpha_counter -= 1                        # Free the Channel from the Alpha Sector
                    Success_Calls_Alpha += 1                                    # Count the successful calls on 'Alpha' Sector
                    del Active_Users_Alpha[i]                                   # Remove the Active user from the Active record for that sector
                else:
                    Traffic_Ch_Sector_Beta_counter -= 1                         # Free the Channel from the Beta Sector
                    Success_Calls_Beta += 1                                     # Count the successful calls on 'Beta' Sector
                    del Active_Users_Beta[i]                                    # Remove the Active user from the Active record for that sector
                continue                                                        # Move on to the next User


            if User_Loc < 0 or User_Loc > Road_Length:                          # Checking if the User has moved beyond the Region of the Road-Length
                #print('User Serviced and Done: User Moved Beyond')
                if User_Sector == 'Alpha':                                      # If the User was currently served by Sector 'Alpha'
                    Call_Record_Type = 'SUCCESS'                                # Mark the Call as a SUCCESS
                    Success_Calls_Alpha += 1                                    # Count the successful calls on 'Alpha' Sector
                    Total_Active_Users[i][7] = Call_Record_Type                 # Make an entry about this Successful call in the Total Active User Records
                    Call_Archive[i] = Total_Active_Users[i]                     # Add the Call and the User Details to a Call Archive for Records
                    del Total_Active_Users[i]                                   # Remove the User from the among the Active Users
                    del Active_Users_Alpha[i]                                   # Remove the Active user from the Active record for that sector
                    Traffic_Ch_Sector_Alpha_counter -= 1                        # Free the Channel from the 'Alpha' Sector
                else:                                                           # If the User was currently served by Sector 'Beta'
                    Call_Record_Type = 'SUCCESS'                                # Mark the Call as a SUCCESS
                    Success_Calls_Beta += 1                                     # Count the successful calls on 'Beta' Sector
                    Total_Active_Users[i][7] = Call_Record_Type                 # Make an entry about this Successful call in the Total Active User Records
                    Call_Archive[i] = Total_Active_Users[i]                     # Add the Call and the User Details to a Call Archive for Records
                    del Total_Active_Users[i]                                   # Remove the User from the among the Active Users
                    del Active_Users_Beta[i]                                    # Remove the Active user from the Active record for that sector
                    Traffic_Ch_Sector_Beta_counter -= 1                         # Free the Channel from the 'Beta' Sector
                continue                                                        # Move on to the next User


            if User_Dir == 'NORTH' and (0 <= User_Loc <= Road_Length):          # Check if the User is withing the Region of Interest/Control and update the User Location, Distance and RSL value
                User_Loc += User_Speed_V                                        # If the User is travelling North then we add the user speed to the user location
                User_Dist = np.sqrt(np.square(BS_Distance_from_Road)
                            + np.square(BS_Location - User_Loc))                # Computing the User Euclidean distance from the User to the Base-station
                Total_Active_Users[i][0] = User_Loc                             # Update the User Location in the Total Active Users records
                Total_Active_Users[i][2] = User_Dist                            # Update the User Distance in the Total Active Users records
                if User_Sector == 'Alpha':                                      # Checking for the sector
                    User_RSL = Compute_RSL(User_Loc, User_Dist, 'alpha')        # Compute the RSL for this sector 'Alpha'
                    Total_Active_Users[i][4] = User_RSL                         # Update the RSL for the User since his location changed
                    if User_RSL < RSL_Thresh:                                   # If the User's Mobile RSL < Threshold required, then Drop the Call
                        #print('Call Dropped....', User_RSL, '<', RSL_Thresh)
                        Call_Status = 'DROPPED:Low_RSL'                         # Call Status changed to 'DROPPED:Low_RSL'
                        Call_Record_Type = 'FAILED'                             # Call Record changed to Failed
                        Total_Active_Users[i][5] = Call_Status                  # Update the Call Status for that User
                        Total_Active_Users[i][7] = Call_Record_Type             # Update the Call Record Type for that User
                        FOS_BS_Alpha[i] = Total_Active_Users[i]                 # Recording this as a Failure of Service for that sector 'Alpha'
                        Call_Archive[i] = Total_Active_Users[i]                 # Adding this User to the Call Archive
                        del Total_Active_Users[i]                               # Delete the User from the Total Active Users records
                        del Active_Users_Alpha[i]                               # Remove the Active user from the Active record for that sector
                        Traffic_Ch_Sector_Alpha_counter -= 1                    # Free the channel for 'Alpha' sector
                        Alpha_Call_Drop_counter += 1                            # Count the 'Alpha' Sector Call Drops
                        continue                                                # Move on to the next User
                    else:
                        RSL_Beta = Compute_RSL(User_Loc, User_Dist, 'beta')     # Compute the RSL for other sector 'Beta'
                        if RSL_Beta > User_RSL + HOm:                           # If the User's Mobile RSL > Threshold required + Hand-Off margin
                            #print('HAND-OFF ATTEMPT...')
                            Alpha_HO_Attempts += 1                              # Counting Alpha Sector's Hand-Off Attempt
                            Call_Attempts_Beta += 1                             # Since this is an attempt to start a call on Beta sector
                            if 0 <= Traffic_Ch_Sector_Beta_counter < 15:        # Checking if the sector 'Beta' has free Channels
                                #print('Serving Sector Changed: BETA')
                                HandOff_Status = 'ALPHA->BETA-HANDOFF:SUCCESS'  # Hand-Off for 'Alpha' is marked as a SUCCESS
                                User_Sector = 'Beta'                            # We change the serving sector as 'Beta'
                                Total_Active_Users[i][8] = HandOff_Status       # Update the Hand-Off Status for the User in Active User records
                                Total_Active_Users[i][3] = User_Sector          # Update the User Sector for the User in Active User records
                                Active_Users_Beta[i] = Total_Active_Users[i]    # Add the User as an Active User on Sector 'Beta'
                                del Active_Users_Alpha[i]                       # Remove the Active user from the Active record for Sector 'Alpha'
                                Traffic_Ch_Sector_Alpha_counter -= 1            # Free the channel for 'Alpha' sector
                                Traffic_Ch_Sector_Beta_counter += 1             # Consume the channel for 'Beta' sector
                                Alpha_HO_Success += 1                           # Count the 'Alpha' Sector Successful Hand-Offs
                                Beta_Success_Call_Conn_counter += 1             # Since the call is connected on Beta Sector
                            else:
                                HandOff_Status = 'ALPHA->BETA-HANDOFF:FAILURE'  # Hand-Off for 'Alpha' is marked as a FAILURE
                                Total_Active_Users[i][8] = HandOff_Status       # Update the Hand-Off Status for the User
                                Alpha_HO_Failure += 1                           # Count the 'Alpha' Sector Failure Hand-Offs
                                Call_Attempts_Beta -= 1                         # Since we do not penalize Beta for trying
                                Beta_Low_Capacity_counter += 1                  # Increment the low capacity counter for Beta sector
                else:
                    User_RSL = Compute_RSL(User_Loc, User_Dist, 'beta')         # Else when the sector is 'Beta' compute the RSL for the sector
                    Total_Active_Users[i][4] = User_RSL                         # Update the RSL for the User
                    if User_RSL < RSL_Thresh:                                   # If the User's Mobile RSL < Threshold required, then Drop the Call
                        #print('Call Dropped....', User_RSL, '<', RSL_Thresh)
                        Call_Status = 'DROPPED:Low_RSL'                         # Call Status changed to 'DROPPED:Low_RSL'
                        Call_Record_Type = 'FAILED'                             # Call Record changed to Failed
                        Total_Active_Users[i][5] = Call_Status                  # Update the Call Status for that User
                        Total_Active_Users[i][7] = Call_Record_Type             # Update the Call Record Type for that User
                        FOS_BS_Beta[i] = Total_Active_Users[i]                  # Recording this as a Failure of Service for sector 'Beta'
                        Call_Archive[i] = Total_Active_Users[i]                 # Adding this User to the Call Archive
                        del Total_Active_Users[i]                               # Delete the User from the Total Active Users records
                        del Active_Users_Beta[i]                                # Remove the Active user from the Active record for that sector
                        Traffic_Ch_Sector_Beta_counter -= 1                     # Free the channel for 'Beta' sector
                        Beta_Call_Drop_counter += 1                             # Count the 'Beta' Sector Call Drops
                        continue                                                # Move on to the next User
                    else:
                        RSL_Alpha = Compute_RSL(User_Loc, User_Dist, 'alpha')   # Compute the RSL for other sector 'Alpha'
                        if RSL_Alpha > User_RSL + HOm:                          # If the User's Mobile RSL > Threshold required + Hand-Off margin
                            #print('HAND-OFF ATTEMPT...')
                            Beta_HO_Attempts += 1                               # Counting Beta Sector's Hand-Off Attempt
                            Call_Attempts_Alpha += 1                            # Since this is an attempt to start a call on Alpha sector
                            if 0 <= Traffic_Ch_Sector_Alpha_counter < 15:       # Checking if the sector 'Alpha' has free Channels
                                #print('Serving Sector Changed: ALPHA')
                                HandOff_Status = 'BETA->ALPHA-HANDOFF:SUCCESS'  # Hand-Off for 'Beta' is marked as a SUCCESS
                                User_Sector = 'Alpha'                           # We change the serving sector as 'Alpha'
                                Total_Active_Users[i][8] = HandOff_Status       # Update the Hand-Off Status for the User in Active User records
                                Total_Active_Users[i][3] = User_Sector          # Update the User Sector for the User in Active User records
                                Active_Users_Alpha[i] = Total_Active_Users[i]   # Add the User as an Active User on Sector 'Alpha'
                                del Active_Users_Beta[i]                        # Remove the Active user from the Active record for Sector 'Beta'
                                Traffic_Ch_Sector_Beta_counter -= 1             # Free the channel for 'Beta' sector
                                Traffic_Ch_Sector_Alpha_counter += 1            # Consume the channel for 'Alpha' sector
                                Beta_HO_Success += 1                            # Count the 'Beta' Sector Successful Hand-Offs
                                Alpha_Success_Call_Conn_counter += 1            # Since the call is connected on Beta Sector
                            else:
                                HandOff_Status = 'BETA->ALPHA-HANDOFF:FAILURE'  # Hand-Off for 'Beta' is marked as a FAILURE
                                Total_Active_Users[i][8] = HandOff_Status       # Update the Hand-Off Status for the User
                                Beta_HO_Failure += 1                            # Count the 'Beta' Sector Hand-Offs Failures
                                Call_Attempts_Alpha -= 1                        # Since we do not penalize Alpha for trying
                                Alpha_Low_Capacity_counter += 1                 # Increment the low capacity counter for Alpha sector

            elif User_Dir == 'SOUTH' and (0 <= User_Loc <= Road_Length):        # Check if the User is withing the Region of Interest/Control and update the User Location, Distance and RSL value
                User_Loc -= User_Speed_V                                        # If the User is travelling South then we subtract the user speed from the user location
                User_Dist = np.sqrt(np.square(BS_Distance_from_Road)
                                    + np.square(BS_Location - User_Loc))        # Computing the User Euclidean distance from the User to the Base-station
                Total_Active_Users[i][0] = User_Loc                             # Update the User Location in the Total Active Users records
                Total_Active_Users[i][2] = User_Dist                            # Update the User Distance in the Total Active Users records
                if User_Sector == 'Alpha':                                      # Checking for the sector
                    User_RSL = Compute_RSL(User_Loc, User_Dist, 'alpha')        # Compute the RSL for this sector 'Alpha'
                    Total_Active_Users[i][4] = User_RSL                         # Update the RSL for the User since his location changed
                    if User_RSL < RSL_Thresh:                                   # If the User's Mobile RSL < Threshold required, then Drop the Call
                        #print('Call Dropped....', User_RSL, '<', RSL_Thresh)
                        Call_Status = 'DROPPED:Low_RSL'                         # Call Status changed to 'DROPPED:Low_RSL'
                        Call_Record_Type = 'FAILED'                             # Call Record changed to Failed
                        Total_Active_Users[i][5] = Call_Status                  # Update the Call Status for that User
                        Total_Active_Users[i][7] = Call_Record_Type             # Update the Call Record Type for that User
                        FOS_BS_Alpha[i] = Total_Active_Users[i]                 # Recording this as a Failure of Service for that sector 'Alpha'
                        Call_Archive[i] = Total_Active_Users[i]                 # Adding this User to the Call Archive
                        del Total_Active_Users[i]                               # Delete the User from the Total Active Users records
                        del Active_Users_Alpha[i]                               # Remove the Active user from the Active record for that sector
                        Traffic_Ch_Sector_Alpha_counter -= 1                    # Free the channel for 'Alpha' sector
                        Alpha_Call_Drop_counter += 1                            # Count the 'Alpha' Sector Call Drops
                        continue                                                # Move on to the next User
                    else:
                        RSL_Beta = Compute_RSL(User_Loc, User_Dist, 'beta')     # Else compute the RSL for the other sector 'Beta'
                        if RSL_Beta > User_RSL + HOm:                           # If the User's Mobile RSL > Threshold required + Hand-Off margin
                            #print('HAND-OFF ATTEMPT...')
                            Alpha_HO_Attempts += 1                              # Counting Alpha Sector's Hand-Off Attempt
                            Call_Attempts_Beta += 1                             # Since this is an attempt to start a call on Beta sector
                            if 0 <= Traffic_Ch_Sector_Beta_counter < 15:        # Checking if the sector 'Beta' has free Channels
                                #print('Serving Sector Changed: BETA')
                                HandOff_Status = 'ALPHA->BETA-HANDOFF:SUCCESS'  # Hand-Off for 'Alpha' is marked as a SUCCESS
                                User_Sector = 'Beta'                            # We change the serving sector as 'Beta'
                                Total_Active_Users[i][8] = HandOff_Status       # Update the Hand-Off Status for the User in Active User records
                                Total_Active_Users[i][3] = User_Sector          # Update the User Sector for the User in Active User records
                                Active_Users_Beta[i] = Total_Active_Users[i]    # Add the User as an Active User on Sector 'Beta'
                                del Active_Users_Alpha[i]                       # Remove the Active user from the Active record for Sector 'Alpha'
                                Traffic_Ch_Sector_Alpha_counter -= 1            # Free the channel from 'Alpha' sector
                                Traffic_Ch_Sector_Beta_counter += 1             # Update the Hand-Off Status for the User
                                Alpha_HO_Success += 1                           # Count the 'Alpha' Sector Successful Hand-Offs
                                Beta_Success_Call_Conn_counter += 1             # Since the call is connected on Beta Sector
                            else:
                                HandOff_Status = 'ALPHA->BETA-HANDOFF:FAILURE'  # Hand-Off for 'Alpha' is marked as a FAILURE
                                Total_Active_Users[i][8] = HandOff_Status       # Update the Hand-Off Status for the User
                                Alpha_HO_Failure += 1                           # Count the 'Alpha' Sector Hand-Offs Failures
                                Call_Attempts_Beta -= 1                         # Since we do not penalize Beta for trying
                                Beta_Low_Capacity_counter += 1                  # Increment the low capacity counter for Beta sector
                else:
                    User_RSL = Compute_RSL(User_Loc, User_Dist, 'beta')         # Compute the RSL for the sector 'Beta'
                    Total_Active_Users[i][4] = User_RSL                         # Update the RSL for that User
                    if User_RSL < RSL_Thresh:                                   # If the User's Mobile RSL < Threshold required, then Drop the Call
                        #print('Call Dropped....', User_RSL, '<', RSL_Thresh)
                        Call_Status = 'DROPPED:Low_RSL'                         # Call Status changed to 'DROPPED:Low_RSL'
                        Call_Record_Type = 'FAILED'                             # Call Record changed to Failed
                        Total_Active_Users[i][5] = Call_Status                  # Update the Call Status for that User
                        Total_Active_Users[i][7] = Call_Record_Type             # Update the Call Record Type for that User
                        FOS_BS_Beta[i] = Total_Active_Users[i]                  # Recording this as a Failure of Service for that sector 'Beta'
                        Call_Archive[i] = Total_Active_Users[i]                 # Adding this User to the Call Archive
                        del Total_Active_Users[i]                               # Delete the User from the Total Active Users records
                        del Active_Users_Beta[i]                                # Remove the Active user from the Active record for that sector
                        Traffic_Ch_Sector_Beta_counter -= 1                     # Free the channel for 'Beta' sector
                        Beta_Call_Drop_counter += 1                             # Count the 'Beta' Sector Call Drops
                        continue                                                # Move on to the next User
                    else:
                        RSL_Alpha = Compute_RSL(User_Loc, User_Dist, 'alpha')   # Else compute the RSL for the other sector 'Alpha'
                        if RSL_Alpha > User_RSL + HOm:                          # If the User's Mobile RSL > Threshold required + Hand-Off margin
                            #print('HAND-OFF ATTEMPT...')
                            Beta_HO_Attempts += 1                               # Counting Beta Sector's Hand-Off Attempt
                            Call_Attempts_Alpha += 1                            # Since this is an attempt to start a call on Alpha sector
                            if 0 <= Traffic_Ch_Sector_Alpha_counter < 15:       # Checking if the sector 'Alpha' has free Channels
                                #print('Serving Sector Changed: ALPHA')
                                HandOff_Status = 'BETA->ALPHA-HANDOFF:SUCCESS'  # Hand-Off for 'Beta' is marked as a SUCCESS
                                User_Sector = 'Alpha'                           # We change the serving sector as 'Alpha'
                                Total_Active_Users[i][8] = HandOff_Status       # Update the Hand-Off Status for the User in Active User records
                                Total_Active_Users[i][3] = User_Sector          # Update the User Sector for the User in Active User records
                                Active_Users_Alpha[i] = Total_Active_Users[i]   # Add the User as an Active User on Sector 'Alpha'
                                del Active_Users_Beta[i]                        # Remove the Active user from the Active record for that sector
                                Traffic_Ch_Sector_Beta_counter -= 1             # Free the channel from 'Beta' sector
                                Traffic_Ch_Sector_Alpha_counter += 1            # Update the Hand-Off Status for the User
                                Beta_HO_Success += 1                            # Count the 'Beta' Sector Successful Hand-Off
                                Alpha_Success_Call_Conn_counter += 1            # Since the call is connected on Beta Sector
                            else:
                                HandOff_Status = 'BETA->ALPHA-HANDOFF:FAILURE'  # Hand-Off for 'Beta' is marked as a FAILURE
                                Total_Active_Users[i][8] = HandOff_Status       # Update the Hand-Off Status for the User
                                Beta_HO_Failure += 1                            # Count the 'Beta' Sector Hand-Off Failures
                                Call_Attempts_Alpha -= 1                        # Since we do not penalize Alpha for trying
                                Alpha_Low_Capacity_counter += 1                 # Increment the low capacity counter for Alpha sector



# *************************** 10. WRITING TO FILES ***************************
# This function produces data into a textfile by reading the Data Structures for every second/hour instance
def Write_to_TXT(dct, fl, flag, hr):
    if flag == 1:
        line = '\n************************************************ HOUR: '+str(hr)+' ************************************************'
        fl.write(line)
        fl.write('\n')
    else:
        fl.write('\n---------------------------------------- '+str(hr)+' SEC ----------------------------------------\n')
        for User in dct:
            if User:
                line = str(User) + ' : ' + str(dct[User])
                fl.write(line)
                fl.write('\n')
            else:
                break



# *************************** 11. CREATE A REPORT FOR SPECIFIED SECTOR ***************************
# This function produces a report textfile for each sector by reading all the data structures
def Create_Report(Channels, Call_Attempts, Success_Conn, Successful_Calls, HO_Attempts,
                  Successful_HO, HO_Failures, Call_Drops, Call_Blocks, fl, T):
    fl.write('\n******************************* HOUR: '+str(T)+' *******************************\n')
    fl.write('Channels currently in use : ' + str(Channels) + '\n')
    fl.write('Call Attempts : ' + str(Call_Attempts) + '\n')
    fl.write('Successful Call Connections : ' + str(Success_Conn) + '\n')
    fl.write('Failed Call Connections : ' + str(Call_Attempts - Success_Conn) + '\n')
    fl.write('Successfully Completed Calls : ' + str(Successful_Calls) + '\n')
    fl.write('Call Drops : ' + str(Call_Drops) + '\n')
    fl.write('Call Blocks : ' + str(Call_Blocks) + '\n')
    fl.write('Hand-Off Attempts : ' +str(HO_Attempts) + '\n')
    fl.write('Successful Hand-Offs : ' + str(Successful_HO) + '\n')
    fl.write('Hand-Off Failures : ' + str(HO_Failures) + '\n')



# *************************** 12. PLOTTING GRAPHS FOR ANALYSIS ***************************
def Graphical_Analysis(x, y1, y2, y3, y4, sector):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 12))        # Creating Subplots for Channels, Calls, Users, Hand-offs
    ax1.plot(x, y1, 'b')                                                        # Axis 01 for Channels
    ax1.set_title('CHANNEL vs TIME')                                            # Set Title
    ax1.set_xlabel('TIME')                                                      # Set X label
    ax1.set_ylabel('CHANNELS USED')                                             # Set Y label
    ax2.plot(x, y2, 'b')                                                        # Axis 02 for Calls
    ax2.set_title('CALLS vs TIME')                                              # Set title
    ax2.set_xlabel('TIME')                                                      # Set X label
    ax2.set_ylabel('CALLS MADE')                                                # Set Y label
    ax3.plot(x, y3, 'g')                                                        # Axis 03 for Active Users
    ax3.set_title('ACTIVE USERS vs TIME')                                       # Set title
    ax3.set_xlabel('TIME')                                                      # Set X label
    ax3.set_ylabel('USERS')                                                     # Set Y label
    ax4.plot(x, y4, 'g')                                                        # Axis 04 for Hand-offs
    ax4.set_title('HAND-OFFS vs TIME')                                          # Set Title
    ax4.set_xlabel('TIME')                                                      # Set X label
    ax4.set_ylabel('HOs')                                                       # Set Y label
    fig.tight_layout()                                                          # Create a tight layout to control range on x and y axis
    if sector == 'A':                                                           # If sector is Alpha
        fig.suptitle('SECTOR ALPHA', fontsize=25)                               # Write the Title as Alpha
        plt.savefig('ALPHA_GRAPH_DATA.png')                                     # Save the plot with the name Alpha
    else:                                                                       # If sector is Beta
        fig.suptitle('SECTOR BETA', fontsize=25)                                # Write the Title as Beta
        plt.savefig('BETA_GRAPH_DATA.png')                                      # Save the plot with the name Beta
    plt.close()



# *************************** 13. SUMMARY REPORT FOR BASE-STATION ***************************
# Writing a final summary file for the entire Base-station (including Alpha and Beta Sectors)
def Summary_Report(file):
    global Total_T
    global Traffic_Ch_Sector_Alpha_counter
    global Traffic_Ch_Sector_Beta_counter
    global Call_Attempts_Alpha
    global Call_Attempts_Beta
    global Alpha_Success_Call_Conn_counter
    global Beta_Success_Call_Conn_counter
    global Success_Calls_Alpha
    global Success_Calls_Beta
    global Alpha_Call_Drop_counter
    global Beta_Call_Drop_counter
    global Alpha_Low_Capacity_counter
    global Beta_Low_Capacity_counter
    global Alpha_HO_Attempts
    global Beta_HO_Attempts
    global Alpha_HO_Success
    global Beta_HO_Success
    file.write('SUMMARY REPORT FOR THE BASE-STATION\n\n')
    file.write('********** HOURS COMPLETED : '+str(Total_T)+' **********\n')
    file.write('\n> Total Channels In Use : '+str(Traffic_Ch_Sector_Alpha_counter + Traffic_Ch_Sector_Beta_counter))
    file.write('\n> Total Call Attempts : '+str(Call_Attempts_Alpha + Call_Attempts_Beta))
    file.write('\n> Total Successfully Connected Calls : '+str(Alpha_Success_Call_Conn_counter + Beta_Success_Call_Conn_counter))
    file.write('\n> Total Successful Calls (Completed) : '+str(Success_Calls_Alpha + Success_Calls_Beta))
    file.write('\n> Total Call Drops (Low RSL) : '+str(Alpha_Call_Drop_counter + Beta_Call_Drop_counter))
    file.write('\n> Total Call Drops (Low Capacity) : '+str(Alpha_Low_Capacity_counter + Beta_Low_Capacity_counter))
    file.write('\n> Total Hand-Off Attempts : '+str(Alpha_HO_Attempts + Beta_HO_Attempts))
    file.write('\n> Total Hand-Off Success : '+str(Alpha_HO_Success + Beta_HO_Success))
    file.write('\n> Total Hand-Off Failures : '+str(Alpha_HO_Attempts + Beta_HO_Attempts - Alpha_HO_Success - Beta_HO_Success))
    file.write('\n\n\t[ BASE-STATION EFFICIENCY : '+str(int(((Success_Calls_Alpha + Success_Calls_Beta + Alpha_HO_Success + Beta_HO_Success) / (Call_Attempts_Alpha + Call_Attempts_Beta)) * 100))+'% ]')
    file.write('\n\n*******************************************')
    file.close()



#################################################### MAIN FUNCTION #####################################################

#Main Begins Here:
All_User_Records_file = open('User_Records.txt', 'w+')                      # File stores All the records for all Users
Active_Alpha_Users_file = open('Alpha_Active_Users.txt', 'w+')              # File stores Users who are and were Active on the Alpha Sector
Active_Beta_Users_file = open('Beta_Active_Users.txt', 'w+')                # File stores Users who are and were Active on the Beta Sector
FOS_BS_Alpha_file = open('FOS_Alpha.txt', 'w+')                             # File stores the service failures for some users on Alpha Sector
FOS_BS_Beta_file = open('FOS_Beta.txt', 'w+')                               # File stores the service failures for some users on Beta Sector
Total_Active_Users_file = open('Total_Active_Users.txt', 'w+')              # File that stores all the Active Users at current (On all Sectors)
Final_Call_Archive_file = open('Archive.txt', 'w+')                         # File that is a Call Archive
Alpha_Report_File = open('Alpha_Report.txt', 'w+')                          # File which contains the report from Sector Alpha
Beta_Report_File = open('Beta_Report.txt', 'w+')                            # File which contains the report from Sector Beta
Summary_Report_File = open('Summary_Report_BS.txt', 'w+')                   # File which contains the final summary for the entire base-station


print('******************************* CELLULAR SIMULATION *******************************\n')
# Most Basic Simulation Parameters (User Input):
Road_Length = int(input('> Enter the Road Length (km) : ')) * 1000          # Getting the Road Length from the User in 'km' and converting to 'm'
Total_T = int(input('> Enter the Total Simulation Time (hrs) : '))          # Getting the Total Cellular Simulation time in 'hrs'
print('')

All_User_Records = {}                                                       # Stores User records for that iteration [User_Location, User_Direction, User_Distance, Sector, RSL, Call_Status]
Active_Users_Alpha = {}                                                     # Dictionary with Active Users and their information for Sector Alpha
Active_Users_Beta = {}                                                      # Dictionary with Active Users and their information for Sector Beta
FOS_BS_Alpha = {}                                                           # Failure of Service on Sector Alpha
FOS_BS_Beta = {}                                                            # Failure of Service on Sector Beta
Total_Active_Users = {}                                                     # Total Active Users and their corresponding details
Call_Archive = {}                                                           # Call Archive keeps a record of all serviced Users
Call_Attempts_Alpha = 0                                                     # Counts the Calls Attempts made on Alpha Sector
Call_Attempts_Beta = 0                                                      # Counts the Calls Attempts made on Beta Sector
Alpha_Success_Call_Conn_counter = 0                                         # Counts the Successful Call Connection for Users on Alpha Sector
Beta_Success_Call_Conn_counter = 0                                          # Counts the Successful Call Connection for Users on Beta Sector
Alpha_Call_Drop_counter = 0                                                 # Counts the Failures (Call Drops) on Alpha Sector
Beta_Call_Drop_counter = 0                                                  # Counts the Failures (Call Drops) on Beta Sector
Alpha_HO_Attempts = 0                                                       # Counts the Alpha Sector Hand-Off Attempts
Alpha_HO_Success = 0                                                        # Counts the Successful Hand-Offs on Alpha Sector
Alpha_HO_Failure = 0                                                        # Counts the Failed Hand-Offs on Alpha Sector
Beta_HO_Attempts = 0                                                        # Counts the Beta Sector Hand-Off Attempts
Beta_HO_Success = 0                                                         # Counts the Successful Hand-Offs on Beta Sector
Beta_HO_Failure = 0                                                         # Counts the Failed Hand-Offs on Beta Sector
Alpha_Low_Capacity_counter = 0                                              # Counts the Failure to connect calls on Alpha sector due to low capacity
Beta_Low_Capacity_counter = 0                                               # Counts the Failure to connect calls on Beta sector due to low capacity
Success_Calls_Alpha = 0                                                     # Counts the Successful Completed Calls on Alpha Sector
Success_Calls_Beta = 0                                                      # Counts the Successful Completed Calls on Beta Sector

# Data Structures used for Graphing Purposes
# These lists are used for plotting on the graphs
channels_Alpha = []     # Keeps Alpha Channels currently in use
channels_Beta = []      # Keeps Beta Channels currently in use
Calls_Alpha = []        # Keeps Alpha Calls Made
Calls_Beta = []         # Keeps Beta Calls Made
ActiveU_Alpha = []      # Keeps Active Alpha Users
ActiveU_Beta = []       # Keeps Active Beta Users
Succ_HO_Alpha = []      # Keeps Successful Alpha HOs
Succ_HO_Beta = []       # Keeps Successful Beta HOs
T = []                  # Keeps time

np.random.seed(0)                                                           # This will generate the same random numbers (for tracking)
Compute_Shadowing()                                                         # Computing the Shadowing Loss beforehand
path = 'antenna_pattern.txt'                                                # File given with degrees and their corresponding value
Antenna_Pattern = Read_AntennaPattern_File(path)                            # Reading the Antenna_Pattern.txt to a Dictionary


for time in range(0, (3600*Total_T)+1, Delta_T):                                        # Iterate for 1 hour = 3600s with step=Delta_T
    if time == 0:                                                                       # At the start we would like to write hour number on the reports
        Hour_No = time + 1                                                              # For Hour Number = 1,
        Write_to_TXT(All_User_Records, All_User_Records_file, 1, Hour_No)               # Write a Header on
        Write_to_TXT(Total_Active_Users, Total_Active_Users_file, 1, Hour_No)           # all output files
        Write_to_TXT(Active_Users_Alpha, Active_Alpha_Users_file, 1, Hour_No)           # signifying the hour
        Write_to_TXT(Active_Users_Beta, Active_Beta_Users_file, 1, Hour_No)             # for which we are
        Write_to_TXT(FOS_BS_Alpha, FOS_BS_Alpha_file, 1, Hour_No)                       # producing the output
        Write_to_TXT(FOS_BS_Beta, FOS_BS_Beta_file, 1, Hour_No)                         # which makes it look
        Write_to_TXT(Call_Archive, Final_Call_Archive_file, 1, Hour_No)                 # a bit systematic.

    elif time % 3600 == 0:
        Hour_No = time / 3600                                                           # For every other other completed, we generate Reports for each Sector
        print('# HOUR ', Hour_No, ' Simulation Complete..')
        Create_Report(Traffic_Ch_Sector_Alpha_counter, Call_Attempts_Alpha,
                      Alpha_Success_Call_Conn_counter, Success_Calls_Alpha,
                      Alpha_HO_Attempts, Alpha_HO_Success, Alpha_HO_Failure,
                      Alpha_Call_Drop_counter, Alpha_Low_Capacity_counter,
                      Alpha_Report_File, Hour_No)                                       # Produce Report for Alpha Sector
        Create_Report(Traffic_Ch_Sector_Beta_counter, Call_Attempts_Beta,
                      Beta_Success_Call_Conn_counter, Success_Calls_Beta,
                      Beta_HO_Attempts, Beta_HO_Success, Beta_HO_Failure,
                      Beta_Call_Drop_counter, Beta_Low_Capacity_counter,
                      Beta_Report_File, Hour_No)                                        # Produce Report for Beta Sector
        Write_to_TXT(All_User_Records, All_User_Records_file, 1, Hour_No+1)
        Write_to_TXT(Total_Active_Users, Total_Active_Users_file, 1, Hour_No+1)
        Write_to_TXT(Active_Users_Alpha, Active_Alpha_Users_file, 1, Hour_No+1)
        Write_to_TXT(Active_Users_Beta, Active_Beta_Users_file, 1, Hour_No+1)
        Write_to_TXT(FOS_BS_Alpha, FOS_BS_Alpha_file, 1, Hour_No+1)
        Write_to_TXT(FOS_BS_Beta, FOS_BS_Beta_file, 1, Hour_No+1)
        Write_to_TXT(Call_Archive, Final_Call_Archive_file, 1, Hour_No+1)

    else:                                                                               # For any other time instance (1s-3599s)
        if len(Total_Active_Users)!=0:                                                  # For the Users that are already Active and on a Call
            Serve_Active_Users(Total_Users)                                             # Call the Function which provides service to the exisiting/active Users
            Write_to_TXT(All_User_Records, All_User_Records_file, 0, time)              # Write the following
            Write_to_TXT(Total_Active_Users, Total_Active_Users_file, 0, time)          # data generated to the
            Write_to_TXT(Active_Users_Alpha, Active_Alpha_Users_file, 0, time)          # text-files for analysis.
            Write_to_TXT(Active_Users_Beta, Active_Beta_Users_file, 0, time)
            Write_to_TXT(FOS_BS_Alpha, FOS_BS_Alpha_file, 0, time)                      # Write the following
            Write_to_TXT(FOS_BS_Beta, FOS_BS_Beta_file, 0, time)                        # data generated to the
            Write_to_TXT(Call_Archive, Final_Call_Archive_file, 0, time)                # text-files for analysis.
            #print('\nAFTER PROCESS:')
            #print('\n**** TOTAL ACTIVE USERS ****\n', Total_Active_Users, '\n')
            #print('\n**** ALPHA FAILURE RECORDS ****\n', FOS_BS_Alpha, '\n')
            #print('\n**** BETA FAILURE RECORDS ****\n', FOS_BS_Beta, '\n')
            #print('\n**** CALL ARCHIVE ****\n', Call_Archive, '\n')
        else:                                                                           # For those Users who currently are not active/have calls
            New_Call_Placing(Total_Users)                                               # We call a function to see which new user will initiate a new call
            Write_to_TXT(All_User_Records, All_User_Records_file, 0, time)              # Write the following
            Write_to_TXT(Total_Active_Users, Total_Active_Users_file, 0, time)          # data generated to the
            Write_to_TXT(Active_Users_Alpha, Active_Alpha_Users_file, 0, time)          # textfiles for some
            Write_to_TXT(Active_Users_Beta, Active_Beta_Users_file, 0, time)            # further analysis.
            Write_to_TXT(FOS_BS_Alpha, FOS_BS_Alpha_file, 0, time)                      # Write the following
            Write_to_TXT(FOS_BS_Beta, FOS_BS_Beta_file, 0, time)                        # data generated to the
            Write_to_TXT(Call_Archive, Final_Call_Archive_file, 0, time)                # text-files for analysis.
            #print('\n**** USER RECORDS ****\n', All_User_Records, '\n')
            #print('\n**** TOTAL ACTIVE USERS ****\n', Total_Active_Users, '\n')
            #print('\n**** ALPHA ACTIVE USERS ****\n', Active_Users_Alpha, '\n')
            #print('\n**** BETA ACTIVE USERS ****\n', Active_Users_Beta, '\n')

    # Plotting the results obtained from above computations
    channels_Alpha.append(Traffic_Ch_Sector_Alpha_counter)                              # Contains channels being used on Alpha Sector
    Calls_Alpha.append(Success_Calls_Alpha)                                             # Contains successful calls for all users on sector Alpha
    ActiveU_Alpha.append(len(Active_Users_Alpha))                                       # Contains the count of Active Users on Alpha sector
    Succ_HO_Alpha.append(Alpha_HO_Success)                                              # Contains count of Alpha Sector Successful Hand-Offs

    channels_Beta.append(Traffic_Ch_Sector_Beta_counter)                                # Contains channels being used on Beta Sector
    Calls_Beta.append(Success_Calls_Beta)                                               # Contains successful calls for all users on sector Beta
    ActiveU_Beta.append(len(Active_Users_Beta))                                         # Contains the count of Active Users on Beta sector
    Succ_HO_Beta.append(Beta_HO_Success)                                                # Contains count of Beta Sector Successful Hand-Offs

    T.append(time)                                                                      # Appends time in seconds for the graph


Graphical_Analysis(T, channels_Alpha, Calls_Alpha, ActiveU_Alpha, Succ_HO_Alpha, 'A')   # Call made to the Graphing function for Alpha sector
Graphical_Analysis(T, channels_Beta, Calls_Beta, ActiveU_Beta, Succ_HO_Beta, 'B')       # Call made to the Graphing function for Beta sector
Summary_Report(Summary_Report_File)                                                     # Write Data to the Final Summary file: base-station

print('\n******************************* END OF SIMULATION *******************************')
