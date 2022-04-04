# library

import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import folium
import geopandas

from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from datetime import datetime

# web page configuring
st.set_page_config(layout='wide')

st.title('House Rocket Analysis')

# Funtions

@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv( path )
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')

    return data

@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)

    return geofile

def set_feature(data):
    data['price_m2'] = data['price'] / (data['sqft_lot'] * 0.305)

    return data

def overview_data(data):

    f_attributes = st.sidebar.multiselect('Enter columns', data.columns)
    f_zipcode = st.sidebar.multiselect('Enter zipcode', data['zipcode'].unique())

    st.header('Data Overview')

    if (f_zipcode != []) & (f_attributes != []):
        data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]
    elif (f_zipcode != []) & (f_attributes == []):
        data = data.loc[data['zipcode'].isin(f_zipcode), :]
    elif (f_zipcode == []) & (f_attributes != []):
        data = data.loc[:, f_attributes]
    else:
        data = data.copy()

    st.dataframe(data)

    # colocar as tabelas uma ao lado da outra
    c1, c2 = st.columns((1, 1))

    # average metrics
    # N° total de imóveis em cada zipcode
    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()

    # Média de preços em cada zipcode
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()

    # Média da área da sala de estar em cada zipcode
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()

    # Média de preços por m2 em cada zipcode
    df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()


    # merge
    m1 = pd.merge(df1, df2, on='zipcode',how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df4, on='zipcode', how='inner')

    # Rename columns
    df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQFT LIVING', 'PRICE/M2']

    c1.header('Average Values')
    c1.dataframe(df, height=600)

    # Statistics Descriptive
    num_attributes = data.select_dtypes(include=['float64', 'int64'])
    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))

    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    # concat
    df1 = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()

    df1.columns = ['ATTRIBUTES', 'MAX', 'MIN', 'MEAN', 'MEDIA', 'STD']

    c2.header('Statistics Values')
    c2.dataframe(df1, height=600)


    return None

def portfolio_density(data, geofile):

    # Base Map
    st.title('Region Overview')

    c1, c2 = st.columns((1, 1))
    c1.header('Portfolio Density')

    df = data.sample(100)

    # Density Map - Folium
    density_map = folium.Map(location=[data['lat'].mean(),
                                       data['long'].mean()], default_zoom_start=15)

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
    with c1:
        folium_static(density_map)

    # Region Price Map
    c2.header('Price Density')

    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', 'PRICE']

    # df = df.sample(10)

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(),
                                            data['long'].mean()], default_zoom_start=15)

    region_price_map.choropleth(data=df,
                                geo_data=geofile,
                                columns=['ZIP', 'PRICE'],
                                key_on='feature.properties.ZIP',
                                fill_color='OrRd',
                                fill_opacity=0.7,
                                line_opacity=0.2,
                                legend_name='AVG PRICE')

    with c2:
        folium_static(region_price_map)

    return None

def  commercial_distribution(data):

    st.sidebar.title('Commercial Options')
    st.title('Commercial Attributes')

    # ------------- Average Price Per Year

    # Filters
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    st.sidebar.subheader('Select Max year Built')
    f_year_built = st.sidebar.slider('Year Built', min_year_built, max_year_built, min_year_built)

    # data select
    st.header('Average Price per Year Built')

    df = data.loc[data['yr_built'] < f_year_built]

    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # ------------- Average Price Per Day

    st.header('Average Price per Day')
    st.sidebar.subheader('Select Price per Day')

    # Filters
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')

    f_date = st.sidebar.slider('Date', min_date, max_date, min_date)

    # data filtering
    data['date'] = pd.to_datetime(data['date'])
    df = data.loc[data['date'] < f_date]
    df = df[['date', 'price']].groupby('date').mean().reset_index()

    # plot
    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # ------------- Histogram
    st.header('Price Distribuition')
    st.sidebar.subheader('Select Max Price')

    # filters
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    f_price = st.sidebar.slider('Price', min_price, max_price, avg_price)

    # data filtering
    df = data.loc[data['price'] < f_price]

    # data plot
    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    return None

def attributes_distribution(data):

    # Histograms

    st.sidebar.title('Attibutions Options')
    st.title('Houses Attributes')

    # filters

    f_bedrooms = st.sidebar.selectbox('Max number of bedrooms', sorted(set(data['bedrooms'].unique())))
    f_bathrooms = st.sidebar.selectbox('Max number of bathrooms', sorted(set(data['bathrooms'].unique())))

    c1, c2 = st.columns((1, 1))

    # House per bedrooms
    c1.header('Houses per bedrooms')
    df = data[data['bedrooms'] <= f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19, color_discrete_sequence=['darkorange'])
    c1.plotly_chart(fig, use_container_width=True)

    # House per bathrooms
    c2.header('Houses per bathrooms')
    df = data[data['bathrooms'] <= f_bathrooms]
    fig = px.histogram(df, x='bathrooms', nbins=19, color_discrete_sequence=['darkblue'])
    c2.plotly_chart(fig, use_container_width=True)

    # filters

    f_floors = st.sidebar.selectbox('Max number of floors', sorted(set(data['floors'].unique())))
    f_waterview = st.sidebar.checkbox('Only Houses with water view')

    c1, c2 = st.columns((1, 1))

    # House per floors
    c1.header('Houses per Floors')
    df = data[data['floors'] <= f_floors]
    fig = px.histogram(df, x='floors', nbins=19, color_discrete_sequence=['darkgreen'])
    c1.plotly_chart(fig, use_container_width=True)

    # House per water view
    c2.header('Houses per Water View')
    if f_waterview:
        df = data[data['waterfront'] == 1]
    else:
        df = data.copy()

    fig = px.histogram(df, x='waterfront', nbins=10, color_discrete_sequence=['darkred'])
    c2.plotly_chart(fig, use_container_width=True)

    return None


if __name__ == '__main__':
# ETL
    # data extration
    path = 'kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

    data = get_data(path)
    geofile = get_geofile(url)

    # data transformation
    data = set_feature(data)

    overview_data(data)

    portfolio_density(data, geofile)

    commercial_distribution(data)

    attributes_distribution(data)
