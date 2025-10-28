# 🛣 NexGen Smart Route Planner

A predictive logistics platform built with *Streamlit* that determines the optimal vehicle type and route based on a user-defined balance of *Delay Reduction, Cost, and Environmental Impact ($\text{CO}_2$)*.

The application uses historical route and fleet data to project the performance of every potential vehicle/route scenario, then ranks them using a weighted multi-criteria decision model.

## ✨ Features

* *Multi-Criteria Optimization:* Users can adjust optimization weights for Time (Delay), Cost, and $\text{CO}_2$ emissions.
* *Predictive Metrics:* Calculates projected *Total Cost* and *Total $\text{CO}_2$* for all possible route-vehicle combinations, using vehicle-specific fuel efficiency and $\text{CO}_2$ factors.
* *Real-World Constraints:* Filters out impractical scenarios, such as using small vehicles (E-Bike, Scooter) for long-haul routes (e.g., over 300 km).
* *Interactive Visualization:* Displays the optimal recommendation with key metrics and a comparison chart of all vehicle/route scenarios.
* *Detailed Breakdown:* Provides an expandable view of the cost components for the suggested route.

---

## ⚙ Setup and Installation

### 1. Prerequisites

You need *Python 3.8+* installed on your system.

### 2. Required Files

This application requires *four CSV files* to be present in the same directory as the main Python script (route_planner.py):

1.  orders.csv (Order details: ID, Origin, Destination, Priority)
2.  routes_distance.csv (Route details: ID, Distance, Traffic Delay)
3.  vehicle_fleet.csv (Vehicle fleet details: Type, $\text{CO}_2$ per KM, Fuel Efficiency)
4.  cost_breakdown.csv (Historical non-fuel cost metrics per Order ID)

> *Note:* The script is designed to handle missing files gracefully with an error message in the Streamlit app.

### 3. Install Dependencies

Install the necessary Python packages using pip:


pip install streamlit pandas numpy altair

# 💻 Technical Code Explanation: NexGen Smart Route Planner

This document provides a detailed breakdown of the Python code's structure, logic, and core functions for the Streamlit application, focusing on data processing, metric calculation, and the optimization algorithm.

---

## 1. ⚙ Configuration and Data Setup

### Imports and Configuration
The code imports standard libraries (streamlit, pandas, numpy, altair) and uses st.set_page_config to set the page title, use a wider **layout="wide"**, and expand the sidebar by default.

### load_and_prepare_data() Function
This function is the backbone of the application's data layer, marked with the @st.cache_data decorator to ensure fast, efficient re-running by caching the results.

| Step | Description | Key Logic |
| :--- | :--- | :--- |
| *1. Load & Filter* | Loads four required CSV files and handles potential FileNotFoundError. It removes a hardcoded exclusion ('Express_Bike') and filters out orders where Origin equals Destination. | pd.read_csv, fleet_df[~...].isin(VEHICLES_TO_EXCLUDE) |
| *2. Cartesian Join* | Merges the base data (Orders + Routes + Costs) with the *entire vehicle fleet* using a temporary 'key' = 1. This creates a DataFrame containing *every possible route-vehicle scenario*. | base_df.merge(fleet_df, on='key', how='inner') |
| *3. Real-World Constraint* | Filters out impractical scenarios. It identifies "Long Haul" routes (Distance_KM > 300) and removes any pairing that uses a "small vehicle type" (E-Bike-C, Scooter) on these long routes. | merged_df['Is_Long_Haul'], conditional filtering using ~ (NOT) operator. |
| *4. Metric Projection* | Recalculates core logistics metrics based on the *proposed vehicle's* specifications: | * *Projected Fuel Consumed:* $\text{Distance} / \text{Fuel\_Efficiency\_Km\_L}$ |
| | | * *Projected Fuel Cost:* $\text{Proj\_Fuel\_Consumed\_L} \times \text{FUEL\_PRICE\_PER\_LITER}$ ($\mathbf{80.0}$ *INR*) |
| | | * **Total Cost (TotalCost):** $\text{Proj\_Fuel\_Cost} + \sum (\text{All Other Historical Costs})$ |
| | | * **Total $\text{CO}_2$ (TotalCO2):** $\text{Distance\_KM} \times \text{CO}_2 \text{ \_Per\_KM}$ |

---

## 2. 🧠 Optimization Logic

### normalize_metrics(df)
This function prepares the data for the multi-criteria decision model by scaling performance into comparable scores.

* *Goal:* Convert all metrics (which are "lower is better") into a *0-100 score* where *100 is optimal* (i.e., the lowest value observed for that metric in the dataset).
* *Formula:*
    $$\text{Score} = 100 \times \frac{\text{Max}{\text{Value}} - \text{Actual}{\text{Value}}}{\text{Max}{\text{Value}} - \text{Min}{\text{Value}}}$$
* *Output:* Creates three new columns: Score_Predicted, Score_Cost, and Score_CO2.

### calculate_weighted_score(row, w_delay, w_cost, w_co2)
This is the core decision-making algorithm.

* *Goal:* Calculate a single, final WeightedScore for a given route-vehicle scenario based on user-defined priorities.
* *Formula:* The sum of the normalized scores multiplied by their corresponding weights (passed as decimals, i.e., $w/100$):
    $$\text{WeightedScore} = (\text{Score}{\text{Delay}} \times w{\text{delay}}) + (\text{Score}{\text{Cost}} \times w{\text{cost}}) + (\text{Score}{\text{CO}2} \times w{\text{co}2})$$

---

## 3. 🖥 Streamlit Interface and Interactivity

### Sidebar and Weight Sliders
The sidebar handles user inputs, including filtering (Origin, Destination, Priority) and the three optimization weights.

* *Session State:* The application uses st.session_state to store and manage the weight values (w_delay, w_cost, w_co2).
* **Callback (update_weights):** This function is critical for maintaining the user experience. It is triggered by the on_change event of any weight slider. Its purpose is to *ensure the three weights always sum to $100\%$*. If one slider is moved, the function calculates the difference from 100 and distributes that difference (after rounding) across the other two weights, keeping them above 0.

### Main Content Logic
1.  *Filtering:* The main logic filters the pre-calculated df based on the user's selected *Origin, **Destination, and **Priority*.
2.  *Optimal Selection:* It applies the calculate_weighted_score to the filtered data and uses filtered_df['WeightedScore'].idxmax() to identify the single best-performing row (the optimal_route).
3.  *Visualization:*
    * *Metrics:* st.metric is used to display the optimal route's delay, cost, and $\text{CO}_2$, using the *difference versus the average of all filtered routes* as the $\mathbf{\text{delta}}$.
    * *Altair Chart:* A dynamic bar chart visualizes the WeightedScore for all competing vehicle types, highlighting the optimal recommendation in green (#10B981) for immediate comparison.
    * *Detail Breakdown:* An st.expander shows a detailed table and an Altair pie chart breaking down the historical non-fuel cost components for the selected optimal route.