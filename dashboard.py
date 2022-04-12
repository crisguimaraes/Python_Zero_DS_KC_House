# =======================================
#                IMPORTS
# =======================================
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import folium
import geopandas

from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from datetime import datetime

# =======================================
#          PAGE CONFIGURATION
# =======================================

st.set_page_config(layout='wide')

st.title('House Rocket - House Sales in King County - USA')

# =======================================
#          DATA EXTRACTION
# =======================================

# Extract data
@st.cache(allow_output_mutation=True)
def get_data(nrows):
    data = pd.read_csv( path, nrows=nrows )

    return data

# Extract geofile
@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)

    return geofile

# =======================================
#         DATA TRANSFORMATION
# =======================================

# -----------------------------
#      Helper functions
# -----------------------------

def set_feature(data):
    """ Converts sqft_lot in m2_lot
    :param data: dataset with column 'sqft_lot'
    :return:: dataset with column 'price_m2' """""
    data['price_m2'] = data['price'] / (data['sqft_lot'] * 0.0929)

    return data

def perc_diff(bigger,smaller):
    """Calculates the percentual difference between two int or float numbers
    :param bigger: greater value
    :param smaller: smaller value
    :return: dif_perc """
    dif_perc = round(((bigger - smaller) / smaller * 100), 2)

    return dif_perc

# -----------------------------
#      Data Overview
# -----------------------------
# 1. Filter properties by one or several zipcodes.

    # Purpose: View properties by zipcode
    # Note: several lat/long in this dataset have the same zipcode, so the zipcode was used as a region grouper.
    # User Action: Choose one or more desired zipcodes.
    # View: A table with all attributes, filtered by zipcodes.

# 2. Choose one or more attributes to view.

    # Purpose: View properties features.
    # User Action: Choose desired features.
    # View: A table with all selected attributes.

def overview_data(data):

# Filters: Overview -----------------------------

    # widgets for attributes and zipcodes selection
    st.sidebar.title('Data Overview')
    f_attributes = st.sidebar.multiselect('Enter columns', data.columns)
    f_zipcode = st.sidebar.multiselect('Enter zipcode', data['zipcode'].unique())

    # making filters work

    # attributes + zipcode -> need rows and cols
    if (f_zipcode != []) & (f_attributes != []):
        # data_overview is used just for the first table
        data_overview = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]
        # data is used for the other components that not first table
        data = data.loc[data['zipcode'].isin(f_zipcode), :]

    # just zipcode -> just filter rows, all colums
    elif (f_zipcode != []) & (f_attributes == []):
        data_overview = data.loc[data['zipcode'].isin(f_zipcode), :]
        data = data.loc[data['zipcode'].isin(f_zipcode), :]

    # just attributes -> just filter cols, all rows
    elif (f_zipcode == []) & (f_attributes != []):
        data_overview = data.loc[:, f_attributes]

    # no attributes -> returns original ds
    else:
        data_overview = data.copy()

# Table: Data Overview -----------------------------

    st.title('Data Overview')

    # Show all columns
    st.write(data_overview.head(), height=400)

# Table: Averages by Zip Code -----------------------------

# 3. Look at the total number of properties, average price, average living room size and average price per
# square meter in each of the zipcodes.

    # Purpose: View averages of some metrics by region.
    # User Action: Enter the desired metrics.
    # View: A table with all attributes selected.

    # place the tables next to each other
    c1, c2 = st.columns((1, 1))

    # average metrics
    # Total number of properties in each zipcode
    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()

    # Average prices on each zipcode
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()

    # Average living room area in each zipcode
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()

    # Average prices per m2 in each zipcode
    df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()


    # merge dataframes per zipcodes
    m1 = pd.merge(df1, df2, on='zipcode',how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df4, on='zipcode', how='inner')

    # Rename columns
    df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQFT LIVING', 'PRICE/M2']

    # Show dataframe in c1 (left)
    c1.header('Averages by Zipcode')
    c1.dataframe(df, height=300)


# Table: Descriptive Attributes -----------------------------

# 4. Analyze each column descriptively.

    # Purpose: View descriptive metrics (mean, median, standard deviation) for each of the chosen attributes.
    # User Action: Enter the desired metrics.
    # View: Table with descriptive metrics per attribute.

    # Descriptive metrics
    num_attributes = data.select_dtypes(include=['float64', 'int64'])
    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))

    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    # concat columns on the same dataframe
    df1 = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()

    # Rename columns
    df1.columns = ['ATTRIBUTES', 'MAX', 'MIN', 'MEAN', 'MEDIAN', 'STD']


    # Show dataframe in c2 (right)
    c2.header('Descriptive Attributes')
    c2.dataframe(df1, height=300)

    return None

# -----------------------------
#      Region Overview
# -----------------------------

def portfolio_density(data, geofile):

# 5. A map with portfolio density by region and also price density.

    # Density: concentration of something.
    # Purpose: View the portfolio density on the map, that is, the number of properties by region and by price.
    # User Action: No action.
    # View: A map with properties density by region.

    # Map Title
    st.title('Region Overview')

    # place the maps next to each other
    c1, c2 = st.columns((1, 1))
    c1.header('Portfolio Density')

    # Reducing dataframe size
    df = data.sample(500)

    # Base Map - Folium (empty map)
    density_map = folium.Map(location=[data['lat'].mean(),
                                       data['long'].mean()], default_zoom_start=15)

    # Add points on map
    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup='Sold R${0} on: {1}, Features: {2} sqft, {3} bedrooms,'
                            '{4} bathrooms, {5} year built'.format(row['price'],
                                                                   row['date'],
                                                                   row['sqft_living'],
                                                                   row['bedrooms'],
                                                                   row['bathrooms'],
                                                                   row['yr_built'])).add_to(marker_cluster)

    # Plot map (left)
    with c1:
        folium_static(density_map)

# Map: Price Density -----------------------------
    c2.header('Price Density')

    # Average price by zipcode
    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    # Rename columns
    df.columns = ['ZIP', 'PRICE']

    # Filter only dataset regions on geofile file
    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(),
                                            data['long'].mean()], default_zoom_start=15)

    # Creates base map
    region_price_map.choropleth(data=df,
                                geo_data=geofile,
                                columns=['ZIP', 'PRICE'],
                                key_on='feature.properties.ZIP',
                                fill_color='OrRd',
                                fill_opacity=0.7,
                                line_opacity=0.3, #0.2
                                legend_name='AVG PRICE ($)')

    # Plot map (right)
    with c2:
        folium_static(region_price_map)

    return None

# -----------------------------
#      Commercial Attributes
# -----------------------------

def  commercial_distribution(data):

    st.sidebar.title('Commercial Options')
    st.title('Commercial Attributes')

# Line Graph: Average Price per Year Built ------------------------------

# 6. Check the annual price change.

    # Purpose: Observe annual price changes.
    # User Action: Filter data by year.
    # View: A line graph with years in x and average prices in y.


    # Filters

    st.sidebar.title('-----------------------------------')
    st.sidebar.title('Commercial Attributes')

    # Extract date
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')

    # Filter - Average Price per Year Built
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    st.sidebar.subheader('Average Price per Year Built')
    f_year_built = st.sidebar.slider('Max Year Built', min_year_built, max_year_built, min_year_built)

    # Use filter data
    df = data.loc[data['yr_built'] <= f_year_built]

    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    # Plot
    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

# Line Graph: Average Price per Day ------------------------------

# 7. Check the daily price change.

    # Purpose: Observe daily price changes.
    # User Action: Filters data by day.
    # View: A line chart with days in x and average prices in y.

    # Filter

    st.sidebar.subheader('Select Price per Day')

    # Filters
    # data['date'] = pd.to_datetime(data['date']).dt.strftime( '%Y-%m-%d')
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')

    f_date = st.sidebar.slider('Date', min_date, max_date, min_date)

    # data filtering
    data['date'] = pd.to_datetime(data['date'])
    df = data.loc[data['date'] <= f_date]

    st.header('Average Price per Day')

    df = df[['date', 'price']].groupby('date').mean().reset_index()

    # plot
    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    return None

# -----------------------------
#      House Attributes
# -----------------------------

def attributes_distribution(data):
# 8. Check properties distribution (histogram) by:

    # - Price;
    # - Bedrooms number;
    # - Bathrooms number;
    # - Floors number;
    # - Water view.
    # Purpose: To observe properties concentration by price, bedrooms, bathrooms, floors and water view.
    # User Action: Filter price, bedrooms, bathrooms, floors and water view.
    # View: A histogram with each attribute defined.

    # Histograms

    st.sidebar.title('Attibutions Options')
    st.title('Houses Attributes')

# Bar Graph: Price Distribuition -----------------------------------

    # filters
    st.sidebar.title('-----------------------------------')
    st.sidebar.title('House Attributes')
    st.sidebar.subheader('Price Distribution')

    # Range Values
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    f_price = st.sidebar.slider('Max Price', min_price, max_price, avg_price)

    # data filtering
    df = data.loc[data['price'] <= f_price]

    # data plot
    st.header('Price Distribution')
    fig = px.histogram(df, x='price', nbins=50)  #nbins = n° de barras
    st.plotly_chart(fig, use_container_width=True)

# Bar Graph: Houses Per Bedrooms -----------------------------------
# Filter
    st.sidebar.subheader('Houses per Bedroom')

    # Get nd array unique bedrooms list
    unique_bedrooms = data['bedrooms'].unique()

    #Converts nd array to dict, and then to list to pass to index of selectbox:
    unique_bedrooms_list = list(dict(enumerate(unique_bedrooms.flatten(), 0)))

    # index sorted by the last key of dictionary (grater number)
    f_bedrooms = st.sidebar.selectbox('Max Number of Bedrooms', sorted(set(data['bedrooms'].unique())),
                                      index=list(unique_bedrooms_list).index(unique_bedrooms_list[-1]))


    #Graph
    c1, c2 = st.columns(2)

    # House per bedrooms
    c1.header('Houses per Bedroom')

    df = data[data['bedrooms'] <= f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19, color_discrete_sequence=['darkorange'])
    c1.plotly_chart(fig, use_container_width=True)

# Bar Graph: Houses per Bathroom ------------------------------------
#Filter
    st.sidebar.subheader('Houses per Bathroom')

    # Get nd array unique bathrooms list
    unique_bathrooms = data['bathrooms'].unique()

    # Converts nd array to dict, and then to list to pass to index of selectbox:
    unique_bathrooms_list = list(dict(enumerate(unique_bathrooms.flatten(), 0)))

    # index sorted by the last key of dictionary (grater number)
    f_bathrooms = st.sidebar.selectbox('Max Number of Bathrooms', sorted(set(data['bathrooms'].unique())),
                                    index=list(unique_bathrooms_list).index(unique_bathrooms_list[-1]) )


    # Graph
    c2.header('Houses per Bathroom')

    df = data[data['bathrooms'] <= f_bathrooms]
    fig = px.histogram(df, x='bathrooms', nbins=19,color_discrete_sequence=['darkblue'] )
    c2.plotly_chart(fig, use_container_width=True)

# Bar Graph: Houses per Floor ---------------------------------------
# Filter
    st.sidebar.subheader('Houses per Floor')

    # Get nd array unique bathrooms list
    unique_floors = data['floors'].unique()

    # Converts nd array to dict, and then to list to pass to index of selectbox:
    unique_floors_list = list(dict(enumerate(unique_floors.flatten(), 0)))

    # index sorted by the last key of dictionary (grater number)
    f_floors = st.sidebar.selectbox('Max Number of Floors', sorted(set(data['floors'].unique())),
                                        index=list(unique_floors_list).index(unique_floors_list[-1]) )


# Graph
    c1, c2 = st.columns(2)

    c1.header('Houses per Floor')
    df = data[data['floors'] <= f_floors]

    fig = px.histogram(df, x='floors', nbins=19, color_discrete_sequence=['darkgreen'])
    c1.plotly_chart(fig, use_container_width=True)

# Bar Graph: Waterview ----------------------------------------------
# Filter
    st.sidebar.subheader('Houses per Water view')
    f_waterview = st.sidebar.checkbox('Only Houses with Waterview')

    if f_waterview:
        df = data[data['waterfront'] == 1]
    else:
        df = data.copy()

# Graph
    c2.header('Waterview')
    fig = px.histogram(df, x='waterfront', nbins=10, color_discrete_sequence=['darkred'])
    c2.plotly_chart(fig, use_container_width=True)

    return None


# -----------------------------
#    Create a buy report
# -----------------------------

def buy_report(data):
    st.title('Business Recommendation Report')
    st.header('Purchasing Recommendation')

    # Business problem: Which properties should House Rocket buy and at what price?

    # Confirm that condition 5 is the best
    data[['condition', 'price']].groupby('condition').mean().reset_index()

    # Group properties by zipcode
    df_zip = data[['price', 'zipcode']].groupby(['zipcode']).median().reset_index()

    # Suggest properties in good condition that are below the median price in the region
    # Join the df_zip dataframe with the data by zipcode
    df = pd.merge(data, df_zip, on='zipcode', how='inner')

    # Rename columns in new dataset
    df.rename(columns={'price_x': 'buy_price', 'price_y': 'median_price'}, inplace=True)  # to keep in the same df

    # Cycle through values and assign to the 'Recommendation' variable the options of 'buy' or 'do not buy'
    df['recommendation'] = 'NA'
    for i in range(len(df)):
        if (df.loc[i, 'buy_price'] < df.loc[i, 'median_price']) & (df.loc[i, 'condition'] >= 4):
            df.loc[i, 'recommendation'] = 'buy'
        else:
            df.loc[i, 'recommendation'] = 'not buy'

    # create a dataset with only properties recommended for purchase
    buy_recom = df.loc[df['recommendation'] == 'buy'].copy()

    # create a column with the meaning of each condition - 5 - excellent and 4 - very good
    buy_recom['condition_status'] = buy_recom['condition'].apply(
              lambda x: 'excelent' if x == 5 else 'good' if x == 4 else None)

    # creates a report with only the important information
    buy_report = buy_recom[['id', 'zipcode', 'buy_price', 'condition_status', 'recommendation', 'lat', 'long']]

    # reset indices for report aesthetics
    buy_report = buy_report.reset_index(drop=True)

    # display the report
    st.dataframe(buy_report)
    st.header('Location of Recommendation Properties')

    # MAP

    # creates and displays a map with recommended properties for purchase

    houses = buy_report[['id', 'lat', 'long', 'buy_price']]

    fig = px.scatter_mapbox(houses,
                            lat='lat',
                            lon='long',
                            size='buy_price',
                            color_continuous_scale=px.colors.cyclical.IceFire,
                            size_max=15,
                            zoom=10)

    fig.update_layout(mapbox_style='open-street-map')
    fig.update_layout(height=600, margin={'r': 0, 't': 0, 'l': 0, 'b': 0})

    st.plotly_chart(fig)

    return buy_report

# -----------------------------
#    Create a sell report
# -----------------------------

def sell_repport(data, buy_recom_ds):

    buy_recom = buy_recom_ds

    st.header('Sales Recommendation Report')

    # Business Problem: Once the house is purchased, when is the best time to sell it and at what price

# Report

    #Create season column, defining summer and winter:
    data['date_month'] = pd.to_datetime(data['date']).dt.month     #month (int)


    data['season'] = data['date_month'].apply(lambda x: 'winter' if (x == 12 or x<=2) else 'summer' if
                     ( 6 <= x <= 8) else 'NA')

    #keep only 'winter' and 'summer' dates
    data = data.loc[data['season'] != 'NA']

    # Confirm if there is a price difference due to seasonality

    group_zips = (data[['price','season']].groupby('season').median().reset_index() )

    # Calculates the price change (median) between summer and winter
    res_price_diff = perc_diff(group_zips['price'][1], group_zips['price'][0])
    # print(res_price_diff) #5.81 mais caro no verão com relação ao inverno

    summer_median_price = 455000.0

    # Sale conditions:

    # 1. If the property purchase price is higher than the median of the region + more expensive seasonality:
    # The sale price will be equal to the purchase price + 10%
    # 2. If the purchase price is less than the region median + more expensive seasonality:
    # The sale price will be equal to the purchase price plus 30%


    # Create a new dataset by joining the buy and sell recommendation datasets, in order to calculate the profit and
    # have the final report

    # Dataset 'buy_recom' for report: id, zipcode, median_price, buy_price
    # Get:
    # 1. season
    # 2. region median price
    # 3. sale price
    # 4. profit

    # 1. Season
    # Create a new sales recommendation dataset
    recom_sell = buy_recom.copy()
    # Recommend selling in summer, then set the season as summer
    recom_sell['season'] = 'summer'

    # 2. The median price for the region
    # The selling price, in the recommended location, on summer. Set in report as 'summer_median_price
    recom_sell['summer_median_price'] = summer_median_price

    # Reset indices to 0,1,2...to loop through the sequence
    recom_sell = recom_sell.reset_index(drop=True)
    #print(recom_sell)

    # 3. Sale price
    # Analyze if we can ask for 30% above the price to stay close to the median, as initially thought:
    # recom_sell[['buy_price', 'summer_median_price']]

    # Checking the difference between the amount paid by the houses and the median for the summer in the zipcode
    # for i in range(len(recom_sell)):
        # print(recom_sell.loc[i,'summer_median_price'], recom_sell.loc[i,'buy_price'])
        # conf_diff = perc_diff(recom_sell.loc[i,'summer_median_price'],recom_sell.loc[i,'buy_price'] )
        # print(conf_diff)

    # based on these percentages: set 30% of the amount paid as the sale value
    recom_sell['sale_price'] = 'NA'
    for i in range(len(recom_sell)):
        recom_sell.loc[i, 'sale_price'] = ((recom_sell.loc[i, 'buy_price'] * 0.3) + (recom_sell.loc[i, 'buy_price']))

    # check the report, now with sale price
    # print(recom_sell)

    # 4. Profit
    recom_sell['profit'] = 'NA'
    for i in range(len(recom_sell)):
        recom_sell.loc[i,'profit'] = (recom_sell.loc[i,'sale_price'] - recom_sell.loc[i,'buy_price'])

    # Create final sales report
    sell_report = recom_sell[['id', 'zipcode', 'season', 'summer_median_price', 'buy_price', 'sale_price', 'profit']]

    # Show the report
    st.dataframe(sell_report)

    # Page footer
    st.write(" \n\n"
             "Made by **Cristiane de Carvalho Guimarães**"
             " \n\n"
             "Details on my: "
                    "[Github](https://github.com/crisguimaraes) "
             "\n\n")

    return None


if __name__ == '__main__':
# ETL
    # data extration
    path = 'kc_house_data.csv'
    data = get_data(500)

    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    geofile = get_geofile(url)

    # data transformation
    data = set_feature(data)

    overview_data(data)

    portfolio_density(data, geofile)

    commercial_distribution(data)

    attributes_distribution(data)

    recom_buy_ds = buy_report(data)

    # Sell report
    sell_repport(data, recom_buy_ds)




