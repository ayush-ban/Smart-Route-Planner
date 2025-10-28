import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

#  page configuration
st.set_page_config(
    page_title="NexGen Smart Route Planner",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  Data Loading and Preparation

@st.cache_data
def load_and_prepare_data():
    """
    Loads, merges, and calculates projected core metrics for every Route/Vehicle scenario,
    applying a crucial real-world distance constraint.
    """
    try:
        # 1. Load DataFrames from local CSV files
        orders_df = pd.read_csv('orders.csv')
        routes_df = pd.read_csv('routes_distance.csv')
        fleet_df = pd.read_csv('vehicle_fleet.csv')
        cost_df = pd.read_csv('cost_breakdown.csv')
        
    except FileNotFoundError as e:
        st.error(f"Error loading required file: **{e.filename}**. Please ensure all four CSV files are in the same directory as the script.")
        return pd.DataFrame() 

    # prepare Fleet DF: rename columns for internal use and consistency
    fleet_df = fleet_df.rename(columns={
        'Vehicle_Type': 'VehicleType_Proposed',
        'CO2_Emissions_Kg_per_KM': 'CO2_Per_KM',
        'Fuel_Efficiency_KM_per_L': 'Fuel_Efficiency_Km_L' # Used for calculation
    })
    
    # excluding express bikes from the fleet for practical routing
    VEHICLES_TO_EXCLUDE = ['Express_Bike']
    fleet_df = fleet_df[~fleet_df['VehicleType_Proposed'].isin(VEHICLES_TO_EXCLUDE)].reset_index(drop=True)
    # ----------------------------------------------------
    
    # filter out orders where Origin == Destination for meaningful routing
    orders_df = orders_df[orders_df['Origin'] != orders_df['Destination']].reset_index(drop=True)

    # Data Integration and Metric Calculation 
    
    cost_metric_cols = [
        'Distance_KM', 'Traffic_Delay_Minutes', 'Fuel_Consumption_L', 
        'Toll_Charges_INR', 'Fuel_Cost', 'Labor_Cost', 'Vehicle_Maintenance', 
        'Insurance', 'Packaging_Cost', 'Technology_Platform_Fee', 'Other_Overhead'
    ]
    
    # 1. Base Data: Orders + Routes + Historical Costs
    base_df = orders_df.merge(routes_df, on='Order_ID', how='inner')
    base_df = base_df.merge(cost_df, on='Order_ID', how='left')
    
    # Fill NaNs with 0 for calculation robustness
    base_df[cost_metric_cols] = base_df[cost_metric_cols].fillna(0)
    
    # 2. Cartesian Join: Combine every historical route with every possible vehicle type
    base_df['key'] = 1
    fleet_df['key'] = 1
    merged_df = base_df.merge(fleet_df, on='key', how='inner').drop('key', axis=1)

    # FIX: APPLYING REAL-WORLD CONSTRAINTS (Distance Filter)
    # This prevents impractical suggestions like using a bike for long-haul routes.
    MAX_BIKE_RANGE_KM = 300 
    
    # Filter out any scenario where the distance exceeds the vehicle's practical range
    merged_df['Is_Long_Haul'] = merged_df['Distance_KM'] > MAX_BIKE_RANGE_KM
    
    # Exclude all 'small vehicle types' if the route is long-haul
    small_vehicle_types = ['Express_Bike'] 
    
    # Keep the row only if: 
    # 1. It's a short haul route OR
    # 2. It's a long haul route AND the vehicle is NOT a small vehicle type
    
    filtered_df = merged_df[
        (~merged_df['Is_Long_Haul']) | 
        (~merged_df['VehicleType_Proposed'].str.contains('|'.join(small_vehicle_types), case=False, na=False))
    ].copy()
    
    # Fallback to the original merged_df if all data was filtered out (shouldn't happen)
    if filtered_df.empty:
        filtered_df = merged_df.copy()
    
    # =========================================================================

    # Recalculate Projected Metrics based on PROPOSED Vehicle ---
    
    # Using the current fuel (diesel as most cargo vehicles are diesel based) price for projected cost calculation
    FUEL_PRICE_PER_LITER = 88.0 
    
    # 3a. Projected Fuel Consumption (using proposed vehicle efficiency)
    filtered_df['Proj_Fuel_Consumed_L'] = filtered_df['Distance_KM'] / filtered_df['Fuel_Efficiency_Km_L'].replace(0, np.nan).fillna(99999) 

    # 3b. Projected Fuel Cost
    filtered_df['Proj_Fuel_Cost'] = filtered_df['Proj_Fuel_Consumed_L'] * FUEL_PRICE_PER_LITER

    # 3c. Final Projected Total Cost (PTc = Proj_Fuel + all other historical non-fuel costs)
    filtered_df['TotalCost'] = (
        filtered_df['Proj_Fuel_Cost'] +
        filtered_df['Labor_Cost'] +
        filtered_df['Vehicle_Maintenance'] +
        filtered_df['Insurance'] +
        filtered_df['Packaging_Cost'] +
        filtered_df['Technology_Platform_Fee'] +
        filtered_df['Other_Overhead'] +
        filtered_df['Toll_Charges_INR'] 
    )

    # 3d. Final Projected Total CO2 (Recalculated based on proposed vehicle CO2 factor)
    filtered_df['TotalCO2'] = filtered_df['Distance_KM'] * filtered_df['CO2_Per_KM']
    
    # 3e. Time Metric: This is the Traffic Delay
    filtered_df['Predicted_Delay_Hours'] = filtered_df['Traffic_Delay_Minutes'] / 60.0
    
    # Final cleanup: Remove rows where core metrics are still zero or null
    final_df = filtered_df[
        (filtered_df['TotalCost'] > 0) & 
        (filtered_df['TotalCO2'] >= 0) & 
        (filtered_df['Predicted_Delay_Hours'] > 0) 
    ]
    
    # Final Route Metrics DataFrame: Select and rename final columns for display
    final_df = final_df[[
        'Order_ID', 'Origin', 'Destination', 'Priority', 'VehicleType_Proposed',
        'Predicted_Delay_Hours', 'TotalCost', 'TotalCO2', 'Distance_KM', 
        'Fuel_Consumption_L', 'Toll_Charges_INR', 'Fuel_Cost', 'Labor_Cost', 
        'Vehicle_Maintenance', 'Insurance', 'Packaging_Cost', 'Technology_Platform_Fee', 'Other_Overhead'
    ]].copy()
    
    # Rename columns for display consistency
    final_df = final_df.rename(columns={
        'Predicted_Delay_Hours': 'Predicted Delay (h)', 
        'TotalCost': 'Cost (₹)', 
        'TotalCO2': 'CO2 (kg)',
        'VehicleType_Proposed': 'Vehicle Type',
        'Toll_Charges_INR': 'Toll_Charges_INR',
        'Fuel_Cost': 'Historical_Fuel_Cost_INR' 
    })

    return final_df

# Load the prepared data
df = load_and_prepare_data()

# If the DataFrame is empty (e.g., due to file not found), stop the application logic
if df.empty:
    st.stop()


# Optimization Logic

def normalize_metrics(df):
    """Normalize the three key metrics (Predicted Delay, Cost, CO2) to a 0-100 scale."""
    
    for col in ['Predicted Delay (h)', 'Cost (₹)', 'CO2 (kg)']: 
        min_val = df[col].min()
        max_val = df[col].max()
        
        if max_val <= min_val:
             df[f'Score_{col.split(" ")[0]}'] = 100
        else:
             df[f'Score_{col.split(" ")[0]}'] = 100 * (max_val - df[col]) / (max_val - min_val)
             
    return df

df = normalize_metrics(df)

def calculate_weighted_score(row, w_delay, w_cost, w_co2):
    """Calculates the weighted score for a single route."""
    return (
        (row.get('Score_Predicted', 0) * w_delay) +
        (row.get('Score_Cost', 0) * w_cost) + 
        (row['Score_CO2'] * w_co2)
    )

# Streamlit Application Layout


# # Title and Description
st.title("🛣️ NexGen Smart Route Planner")
st.markdown("""
Welcome to the predictive logistics platform. Use the controls on the left to prioritize 
**Delay Reduction, Cost, and Environmental Impact ($\text{CO}_2$)** for your next shipment. 
The system suggests the optimal route/vehicle based on historical performance data.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Route Parameters")

# Origin and Destination Selection
origins = df['Origin'].unique()
destinations = df['Destination'].unique()
selected_origin = st.sidebar.selectbox("Select Origin Warehouse:", sorted(origins), index=0)

# Filter destinations to exclude origin
available_destinations = sorted([d for d in destinations if d != selected_origin])
# Fallback for index if list is empty
destination_index = 0
if 'selected_destination' in locals() and selected_destination in available_destinations:
    destination_index = available_destinations.index(selected_destination)
elif available_destinations:
    destination_index = 0
else:
    st.warning("No unique destinations available.")
    selected_destination = None


selected_destination = st.sidebar.selectbox("Select Final Destination:", available_destinations, index=destination_index if available_destinations else 0)

# Priority Filter (Optional constraint based on orders data)
selected_priority = st.sidebar.selectbox(
    "Order Priority Level:",
    ['All'] + list(df['Priority'].unique())
)

# Optimization Weights
st.sidebar.header("Optimization Weights (Total must be 100%)")

# Initialize weights in session state if not present
if 'w_delay' not in st.session_state: 
    st.session_state.w_delay = 33
if 'w_cost' not in st.session_state:
    st.session_state.w_cost = 34
if 'w_co2' not in st.session_state:
    st.session_state.w_co2 = 33

# Custom callback to ensure weights sum to 100
def update_weights(changed_weight):
    """Adjusts the other two weights when one slider is moved."""
    current_total = st.session_state.w_delay + st.session_state.w_cost + st.session_state.w_co2
    if current_total != 100:
        difference = 100 - current_total
        
        slots = []
        if changed_weight != 'w_delay': slots.append('w_delay')
        if changed_weight != 'w_cost': slots.append('w_cost')
        if changed_weight != 'w_co2': slots.append('w_co2')
        
        if slots:
            adjustment_per_slot = difference / len(slots)
            for slot in slots:
                st.session_state[slot] = max(0, st.session_state[slot] + adjustment_per_slot)
                
            st.session_state.w_delay = int(np.round(st.session_state.w_delay))
            st.session_state.w_cost = int(np.round(st.session_state.w_cost))
            st.session_state.w_co2 = 100 - st.session_state.w_delay - st.session_state.w_cost 


# FIX APPLIED: Using keyword arguments for 'value' to resolve the Session State warning.
w_delay = st.sidebar.slider("⏱️ Delay Reduction Weight (%)", 
                            min_value=0, 
                            max_value=100, 
                            value=st.session_state.w_delay, 
                            key='w_delay', 
                            on_change=lambda: update_weights('w_delay'))

w_cost = st.sidebar.slider("💰 Cost Reduction Weight (%)", 
                           min_value=0, 
                           max_value=100, 
                           value=st.session_state.w_cost, 
                           key='w_cost', 
                           on_change=lambda: update_weights('w_cost'))

w_co2 = st.sidebar.slider("🌿 CO2 Reduction Weight (%)", 
                          min_value=0, 
                          max_value=100, 
                          value=st.session_state.w_co2, 
                          key='w_co2', 
                          on_change=lambda: update_weights('w_co2'))


# Ensure the sum is always 100% and display it
current_sum = w_delay + w_cost + w_co2
st.sidebar.markdown(f"**Total Weight:** **{current_sum}%**")
if current_sum != 100:
    st.sidebar.warning("Weights were auto-adjusted to total 100%.")


# --- 4. Main Content Logic and Results Display ---

if selected_destination is None:
    st.warning("Please select a valid origin and destination pair.")
elif df.empty:
    st.error("The application could not load or process the data. Please check your CSV files.")
else:
    # 4.1: Filter Data
    filtered_df = df[
        (df['Origin'] == selected_origin) &
        (df['Destination'] == selected_destination)
    ].copy()

    if selected_priority != 'All':
        filtered_df = filtered_df[filtered_df['Priority'] == selected_priority]

    if filtered_df.empty:
        st.info(f"No practical historical routes found for {selected_origin} to {selected_destination} with priority '{selected_priority}'. Please try different inputs.")
    else:
        # 4.2: Calculate Score and Find Optimal Route
        filtered_df['WeightedScore'] = filtered_df.apply(
            lambda row: calculate_weighted_score(row, w_delay/100, w_cost/100, w_co2/100),
            axis=1
        )

        # The best route has the highest score
        optimal_route = filtered_df.loc[filtered_df['WeightedScore'].idxmax()]

        st.subheader("✅ Optimal Route Suggestion")
        
        # 4.3: Display Key Metrics of Optimal Route
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Recommended Vehicle", optimal_route['Vehicle Type'])
        
        # Use deltas to show comparison against the AVERAGE of the filtered routes
        avg_delay = filtered_df['Predicted Delay (h)'].mean()
        avg_cost = filtered_df['Cost (₹)'].mean() 
        avg_co2 = filtered_df['CO2 (kg)'].mean()
        
        delay_delta = optimal_route['Predicted Delay (h)'] - avg_delay
        cost_delta = optimal_route['Cost (₹)'] - avg_cost 
        co2_delta = optimal_route['CO2 (kg)'] - avg_co2


        col2.metric("Predicted Delay (h)", 
                    f"{optimal_route['Predicted Delay (h)']:.1f}", 
                    delta=f"{delay_delta:.1f} vs Avg", 
                    delta_color="inverse")
        col3.metric("Predicted Cost (₹)", 
                    f"₹{optimal_route['Cost (₹)']:.2f}", 
                    delta=f"₹{cost_delta:.2f} vs Avg", 
                    delta_color="inverse")
        col4.metric("Predicted CO2 (kg)", 
                    f"{optimal_route['CO2 (kg)']:.1f}", 
                    delta=f"{co2_delta:.1f} kg vs Avg", 
                    delta_color="inverse")
        
        st.markdown("---")

        st.subheader("🔍 Performance Comparison")
        
        comparison_df = filtered_df.sort_values('WeightedScore', ascending=False)
        comparison_df['Rank'] = np.arange(1, len(comparison_df) + 1)
        
        
        st.markdown("##### Weighted Optimization Score by Vehicle Type")
        
        score_chart = alt.Chart(comparison_df).mark_bar().encode(
            x=alt.X('Vehicle Type:N', sort='-y'),
            y=alt.Y('WeightedScore', title='Optimization Score (0-100)'),
            color=alt.condition(
                alt.datum['Order_ID'] == optimal_route['Order_ID'],
                alt.value('#10B981'),
                alt.value('#6B7280')
            ),
            tooltip=['Vehicle Type', 'Predicted Delay (h)', 'Cost (₹)', 'CO2 (kg)', 'WeightedScore'] 
        ).properties(
            height=300
        ).interactive()
        
        st.altair_chart(score_chart, use_container_width=True)


        # 4.4: Detail Breakdown 
        with st.expander(f"Detailed Breakdown for Recommended Route ({optimal_route['Vehicle Type']})"):
            
            detail_cols = [
                'Historical_Fuel_Cost_INR', 'Labor_Cost', 'Vehicle_Maintenance', 'Insurance', 
                'Packaging_Cost', 'Other_Overhead', 'Toll_Charges_INR',
                'Distance_KM', 'Fuel_Consumption_L', 'Predicted Delay (h)', 'CO2 (kg)' 
            ]
            
            details_series = optimal_route[detail_cols]
            details_df = details_series.to_frame(name='Value').reset_index().rename(columns={'index': 'Component'})
            
            cost_components_chart = details_df[details_df['Component'].isin([
                'Historical_Fuel_Cost_INR', 'Labor_Cost', 'Vehicle_Maintenance', 'Insurance', 
                'Packaging_Cost', 'Other_Overhead', 'Toll_Charges_INR'
            ])]
            
            cost_chart = alt.Chart(cost_components_chart).mark_arc(outerRadius=120).encode(
                theta=alt.Theta("Value", stack=True),
                color=alt.Color("Component"),
                tooltip=["Component", alt.Tooltip("Value", format=".2f")],
                order=alt.Order("Value", sort="descending")
            ).properties(
                title="Cost Component Distribution (INR)",
            )
            
            c1, c2 = st.columns(2)
            with c1:
                st.altair_chart(cost_chart, use_container_width=True)
                
            with c2:
                # Update currency symbol for the detail table
                details_df['Value'] = details_df.apply(
                    lambda row: f"₹ {row['Value']:.2f}" if row['Component'] in cost_components_chart['Component'].tolist() else f"{row['Value']:.2f}",
                    axis=1
                )
                st.table(details_df)
        
        st.markdown("---")
        
        # 4.5 Mock Map 
        st.subheader("Route Map Mock-up")
        
        # Mock coordinates mapping for visualization purposes
        mock_coords = {
            'City 1': (34.0522, -118.2437),  
            'City 2': (40.7128, -74.0060),  
            'City 3': (41.8781, -87.6298),  
            'City 4': (32.7767, -96.7970),  
            'City 5': (29.7604, -95.3698),  
            'City 6': (33.4484, -112.0740), 
            'City 7': (39.7392, -104.9903), 
            'City 8': (37.7749, -122.4194), 
            'City 9': (47.6062, -122.3321)  
        }
        
        city_to_coord = {}
        for city in df['Origin'].unique():
            if city not in mock_coords:
                if len(city_to_coord) < len(mock_coords):
                    mock_key = list(mock_coords.keys())[len(city_to_coord)]
                    city_to_coord[city] = mock_coords[mock_key]
                else:
                    city_to_coord[city] = (39.8283, -98.5795) 
            else:
                city_to_coord[city] = mock_coords[city]

        if selected_origin in city_to_coord and selected_destination in city_to_coord:
            map_data = pd.DataFrame({
                'lat': [city_to_coord[selected_origin][0], city_to_coord[selected_destination][0]],
                'lon': [city_to_coord[selected_origin][1], city_to_coord[selected_destination][1]],
                'Name': [selected_origin, selected_destination]
            })

            st.map(map_data, latitude='lat', longitude='lon', size=200, zoom=3)
            st.caption(f"Showing coordinates for: **{selected_origin}** (Start) and **{selected_destination}** (End).")
        else:
            st.info("Map not available: Missing geographical coordinates for selected cities.")