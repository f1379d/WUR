#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output
from sklearn.cluster import KMeans


# In[5]:


# Read data from csv file
df = pd.read_csv(
    'https://raw.githubusercontent.com/f1379d/Dash/main/Times.csv')

# Create a list containing years in dataset
years = list(pd.Series(df['Year']).unique())

# Create a list of dictionaries in form of {'label':'2021', 'value': 2021}
years = [{'label': str(year), 'value': year} for year in years]

# Create a list of dictionaries of features
features = [
    {'label': 'Teaching', 'value': 'Teaching'},
    {'label': 'Citations', 'value': 'Citations'},
    {'label': 'Research', 'value': 'Research'},
    {'label': 'Industry Income', 'value': 'Industry Income'},
    {'label': 'International Outlook', 'value': 'International Outlook'},
]


# In[6]:


# Build App
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# --------------- HTML components Section -------------
app.layout = html.Div([
    # Headline of page
    html.H1('Times Rank Clustering'),

    # Dropdown for selecting year
    html.Label('Year:'),
    dcc.Dropdown(
        id='year',
        options=years,
        placeholder='Select a year',
        multi=True,
        style={'width': 500}
    ),

    # Dropdown for selecting features
    html.Label('Features:'),
    dcc.Dropdown(
        id='feature',
        options=features,
        placeholder='Select a feature',
        multi=True,
        style={'width': 500}
    ),

    # Dropdown for selecting number of clusters
    html.Label('Number of clusters:'),
    dcc.Dropdown(
        id='n',
        options=[{'label': str(n), 'value': n} for n in range(1, 33)],
        value=1,
        multi=False,
        style={'width': 500}
    ),

    # Toggle switch for trendline
    html.Label('Trendline:'),
    daq.ToggleSwitch(
        id='trendline_switch',
        style={'width': 100},
        value=False
    ),

    # Graph component for showing result in form of a scatter plot
    dcc.Graph(id='graph')

],)

# --------------- Functions for Callback Section -------------


@app.callback(
    Output('graph', 'figure'),
    Input('year', 'value'),
    Input('feature', 'value'),
    Input('n', 'value'),
    Input('trendline_switch', 'value')
)
def update_graph(year, feature, n, trendline_switch):
    # If only one year was selected, it would be converted to form of a list
    if(not isinstance(year, list)):
        tmp = year
        year = []
        year.append(tmp)

    # If only one feature was selected, it would be converted to form of a list
    if not isinstance(feature, list):
        tmp = feature
        feature = []
        feature.append(tmp)

    # Rank will be added to the feature as a compulsory feature for clustering
    feature.append('Rank')

    # Create a new data frame for further use
    dff = pd.DataFrame(
        columns=['Label', 'Year', 'Rank', 'Percentage', 'Cluster'])

    # Find graph points for each of the selected year(s)
    for selected_year in year:
        # Define input for KMeans algorithm
        X = df.loc[df['Year'] == selected_year, feature]
        # Apply clustering
        model = KMeans(n_clusters=n, random_state=0)
        model.fit(X)

        # Finding centroid of each cluster
        centroids = model.cluster_centers_

        # Turn centroids into data frame
        df_centroids = pd.DataFrame(centroids)
        df_centroids.columns = feature

        # Rank clusters' centroids based on rank and convert it inor np.array object
        centroids = df_centroids.sort_values(
            by='Rank', ascending=False).reset_index(drop=True).to_numpy()

        # Change format of data for ease of plotting
        for i in range(len(centroids)):
            for j in range(len(centroids[i])-1):
                dff = dff.append({'Label': feature[j], 'Percentage': centroids[i][j],
                                  'Rank': centroids[i][-1],
                                  'Cluster': int(i)+1, 'Year': selected_year}, ignore_index=True)

    dff = dff.reset_index(drop=True)

    # Creating new plot for returning to output graph
    fig = px.scatter(
        dff,
        x='Cluster',
        y='Percentage',
        color='Label',
        symbol='Label',
        range_y=[0, 100],
        hover_data=['Rank'],
        facet_col='Year',
        facet_col_spacing=0.08,
        template='plotly_dark',
        trendline='ols' if trendline_switch else None,
    )

    return fig


# Run app and display result external
if __name__ == '__main__':
    app.run_server(debug=False)


# In[ ]:
