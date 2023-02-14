# EVCEval: An EV charging station access equilibrium model 

## Introduction

This script gives an EV-to-charging station assignment tool with M/D/C charging station queueing. The tool outputs the assignment matrix, the access cost matrix, as well as the expected wait time and utilization ratio of charging stations, and average access cost of the origins. It could be used to evaluate current or designed EV charging station configurations. 

The EV assignment is formulated as a capacitated user equilibrium (UE) problem, originally introduced in Beckman et al. (1956), to capture users’ charging station choice. At UE, charging stations used by the EVs from the same origin should have the smallest and the same access cost (travel cost + waiting cost). The capacity if charging station is reflected in the M/D/C queueing model, which assumes users arrive sufficiently randomly while charging time is deterministic (e.g. an empty tank will take half hour to fully recharge in a DCFC). Because there is no closed form expression for an M/D/C model, we make use of an approximation from Barceló et al. (1996). The model is solved using a Method of Successive Averages which guarantees convergence for convex problems like this one (Powell and Sheffi, 1982). 

References:

Beckmann, M., McGuire, C. B., & Winsten, C. B. (1956). Studies in the Economics of Transportation (No. 226 pp).

Barceló, F., Casares, V., & Paradells, J. (1996). M/D/C queue with priority: Application to trunked mobile radio systems. Electronics Letters, 32(18), 1644-1644.

Powell, W. B., & Sheffi, Y. (1982). The convergence of equilibrium algorithms with predetermined step sizes. Transportation Science, 16(1), 45-55.

-------------------------------------------------------

## Instructions

To use this Tool:

Please save the EV location list, charging station list and the travel cost matrix as csv files in the same folder as this script, then fill in the file names in the following cell correspondingly.

1. Prepare EV location csv file
    
    Columns:
    
        "ID" : 1,2,3,4,... (int64)
    
        "Number of EVs" : the number of EVs to be charged per unit time at that location (int64)
    
2. Prepare Charging Station csv file

    Note: Within each charging station, all the chargers are of the same type (Level 2 or DC Fast). If some charging stations have both Level 2 and DC Fast chargers, please seperate each of them as 2 charging stations at the same location.
    
    Columns:
    
        "ID" : 1,2,3,4,... (int64)
        
        "miu": service rate (number of EVs charged/unit time) of one charger at this charging
        station (int64)
        
        "Number of Chargers" : the number of chargers at this charging station (int64)
        
3. Prepare OD cost Matrix csv file

    This matrix should be the travel distance matrix. 
    
    Note:
    
        1) The sequence of the rows should correspond to the sequence of the EV Locations in the EV location csv file, and the sequence of the columns should correspond to the sequence of the charging station csv file (float64)
        2) No heading and index in the csv file

4. Set Average Travel Speed

    Parameter "v": average travel speed of the EV's.
    
    Note: Unit of the speed should be: unit of distance for OD cost/unit of time for service rate and number of EVs to be charged  
    
5. Set Weights for Access Time and Charging Time

    Parameter "access_time_weight":  weight of access time. Recommended value is 6.198.
    
    Parameter "charging_time_weight":  weight of charging time. Recommended value is 1.
    
    For the recommended values, please refer to Ge 2019.
    
    Reference: Ge, Y. (2019). Discrete Choice Modeling of Plug-in Electric Vehicle Use and Charging Behavior Using Stated Preference Data (Doctoral dissertation).

6. Set Convergence Criteria

    Parameter "e":  If the Euclidean norm of the difference between the assignment matrix of this iteration and the last iteration is less than e, the algorithm stops and outputs the results.  

7. Set Printing

    Parameter "pri": print error, mean cost and number of steady-state charging stations every  pri number of iterations. If no printing is needed before the final results, set pri as inf (pri = np.inf)

8. Run the algorithm
    After the algorithm converges with the convergence criterion that you set, please check if all the charging stations are at steady state. 
    The 6 output files are saved in a folder named "Results" where this script is saved. 
    
!!! Please make sure that there's no folder named "Results" at the directiry where the script and input files are saved. 
    The output files include:
    
     - Assignment Matrix.csv
     - Access Time Matrix.csv
     - Charging Time Matrix.csv
     - Access + Charging Time Matrix.csv
     - Charging Station Table.csv
         - Columns:“Utilization Ratio”, 
                   “Expected Queue Delay”, 
                   “Charging Time”, 
                   “Expected Queue Delay + Charging Time”
     - EV Parking Location Table.csv
         - Columns:“Average Access Time”, 
                   “Average Charging Time”, 
                   “Average Access +Charging Time”
    
    The following results are directly printed:
     - System Total Access Time
     - System Total Access Time + Charging Time
     - Average Access Time for one EV
     - Average Access Time + Charging Time for one EV
     - Number of steady-state (utilization ratio ≤ 1) charging stations at convergence 
     - Number of iterations until convergence
     - Run time of the algorithm (wall time)
     
!!! Please check if the number of steady-state charging stations at convergence equals to the total number of charging stations. If a large number of charging stations are not at steady state, the results are not accurate.
Run the algorithm below:


-------------------------------------------------------

## Example


Our paper on this:
https://www.researchgate.net/publication/349234204_An_electric_vehicle_charging_station_access_equilibrium_model_with_MDC_queueing


Thank you!

--
Bingqing (Chloe) Liu

Ph.D Candidate in Transportation Planning and Engineering   

C2SMART

New York University Tandon School of Engineering

bl2453@nyu.edu

