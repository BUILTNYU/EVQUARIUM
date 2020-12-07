# EV_Charging_Station_Access_Equlibrium_Model

This script gives an EV-to-charging station assignment tool with M/D/C charging station queueing.The tool outputs the assignment matrix, the access cost matrix, as well as the expected wait timeand utilization ratio of charging stations, and average access cost of the origins. It could be used to evaluate current or designed EV charging station configurations. 
The EV assignment is formulated as a capacitated user equilibrium (UE) problem, originallyintroduced in Beckman et al. (1956), to capture users’ charging station choice. At UE, charging stations used by the EVs from the same origin should have the smallest and the same access + charging time cost (travel time + queue delay + charging time). The capacity if charging station is reflected in the M/D/C queueing model, which assumes users arrive sufficiently randomly while charging time is deterministic (e.g. an empty tank will take half hour to fully recharge in a DCFC). Because there is no closed form expression for an M/D/C model, we make use of an approximation from Barceló et al. (1996). The model is solved using a Method of Successive Averages which guarantees convergence for convex problems like this one (Powell and Sheffi, 1982). 

To use this Tool:
1. Prepare EV location csv file
    Columns:
    "ID" : 1,2,3,4,... (int64)
    "Number of EVs" : the number of EVs to be charged per unit time at that location (int64)
    
2. Prepare Charging Station csv file
    Note: Within each charging station, all the chargers are of the same type 
          (Level 2 or DC Fast).If some charging stations in
    Columns:
        "CS_ID" : 1,2,3,4,... (int64)
        "miu": service rate (number of EVs charged/unit time) of one charger at this charging
        station (int64)
        "Number of Chargers" : the number of chargers at this charging station (int64)
        
3. Prepare OD cost Matrix csv file
    This matrix should be the travel distance matrix. 
    Note:
    1) The sequence of the rows should correspond to the sequence of the EV Locations in the
    EV location csv file, and the sequence of the columns should correspond to the sequence
    of the charging station csv file (float64)
    2) No heading and index in the csv file
    
Please save the EV location list, charging station list and the travel cost matrix as csv
files in the same folder as this script, then fill in the file names in the following cell
correspondingly.

4. Set Average Travel Speed
    Note: Unit of the speed should be: unit of distance for OD cost/unit of time for service 
rate and number of EVs to be charged

5. Set Convergence Criteria
    Parameter "e":  If the Euclidean norm of the difference between the assignment matrix of 
    this iteration and the last iteration is less than e, the algorithm stops and outputs the
results.   

6. Set Printing
    Parameter "pri": print error, mean cost and number of steady-state charging stations every  pri number of iterations. If no printing is needed before the final results, set pri as inf (pri = np.inf)

7. Run the algorithm
    After the algorithm converges with the convergence criterion that you set, please check
    if all the charging stations are at steady state. 
    The following output files are saved in the folder where this script is saved, including
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
     - Average Access Time of EV Locations at UE.csv
     
     The following values are directly printed, including
     - System Total Access Time
     - System Total Access Time + Charging Time
     - Average Access Time for one EV
     - Average Access Time + Charging Time
     - Number of steady-state (utilization ratio ≤ 1) charging stations at convergence 
       (should be equal to the total number of charging stations to get accurate results)
     - Number of iterations until convergence
     - Run time of the algorithm (wall time)


Thank you!

--
Bingqing (Chloe) Liu
Ph.D Candidate in Transportation Planning and Engineering
C2SMART
New York University Tandon School of Engineering
bl2453@nyu.edu

