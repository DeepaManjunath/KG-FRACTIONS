#!/usr/bin/env python
# coding: utf-8

# In[2]:


import dash
from dash import html, dcc, Input, Output, State

import plotly.graph_objs as go
import networkx as nx
import pandas as pd
from plotly.offline import plot
import plotly.offline as py_offline
from dash.dependencies import Input, Output, State


modified_excel_file_path = '/Users/deepamanjunath/Downloads/FRACTIONS.xlsx'
modified_data_df = pd.read_excel(modified_excel_file_path)



def create_modified_graph(data_df):
    # Create a directed graph
    G = nx.DiGraph()

    # Add head nodes for each unique 'Topic'
    for topic in data_df['Topic'].dropna().unique():
        G.add_node(topic, level='head', color='red', size=40)

    # Fill missing values in 'Topic', 'Subtopic', and 'ObjectiveName'
    data_df['Topic'] = data_df['Topic'].ffill()
    data_df['Subtopic'] = data_df['Subtopic'].ffill()
    data_df['ObjectiveName'] = data_df['ObjectiveName'].ffill()

    # Add nodes and edges for 'Subtopics' and 'ObjectiveName'
    for _, row in data_df.iterrows():
        if pd.notna(row['Subtopic']):
            G.add_node(row['Subtopic'], level='subtopic', color='green', size=30)
            G.add_edge(row['Topic'], row['Subtopic'])
        if pd.notna(row['ObjectiveName']):
            G.add_node(row['ObjectiveName'], level='objective', color='purple', size=20)
            G.add_edge(row['Subtopic'], row['ObjectiveName'])

    # Add nodes for 'LearningObjectives' and create a mapping
    learning_objectives_map = {}
    for _, row in data_df.iterrows():
        if pd.notna(row['LearningObjectives']):
            learning_objective = row['LearningObjectives']
            objective_id = f"{learning_objective[:80]}..."
            G.add_node(objective_id, level='learning_objective', color='yellow', size=10)
            G.add_edge(row['ObjectiveName'], objective_id)
            learning_objectives_map[learning_objective] = objective_id
    
    #print('LOM',learning_objectives_map)
    # Process 'Pre-requisites' and add additional edges
    for index, row in data_df.iterrows():
        if pd.notna(row['LearningObjectives']):
            current_learning_objective = row['LearningObjectives']
            current_objective_node = learning_objectives_map.get(current_learning_objective)
     
            # Iterate through the DataFrame until a different 'LearningObjective' is encountered
            for inner_index, inner_row in data_df.loc[index:].iterrows():
                
                if pd.notna(inner_row['LearningObjectives']) and inner_row['LearningObjectives'] != current_learning_objective:
                    break  # Stop if a different 'LearningObjective' is encountered
                if pd.notna(inner_row['Pre-requisites']):
                    #print(inner_row['Pre-requisites'])
                    #print('LOM',learning_objectives_map)
                    pre_reqs = [x.strip() for x in inner_row['Pre-requisites'].split(',')]
                    for pre_req in pre_reqs:
                        pre_req_node = learning_objectives_map.get(pre_req)
                        #print("pre_req_node", pre_req_node)
                        if pre_req_node and pre_req_node != current_objective_node:
                            G.add_edge(current_objective_node, pre_req_node, color='purple', style='dotted')

    return G


    


# Create the graph
G = create_modified_graph(modified_data_df)

# Prepare the Plotly figure
pos = nx.spring_layout(G)  # Node positioning
edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

node_x, node_y, node_size, node_color = [], [], [], []
for node in G.nodes(data=True):
    x, y = pos[node[0]]
    node_x.append(x)
    node_y.append(y)
    node_size.append(node[1]['size'])
    node_color.append(node[1]['color'])

node_trace = go.Scatter(
    x=node_x, y=node_y, text=list(G.nodes()), mode='markers', hoverinfo='text',
    marker=dict(showscale=False, size=node_size, color=node_color, line_width=2))

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='<br>Knowledge Graph of Fractions',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600
                ))


# Dash app setup
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(
        id='knowledge-graph',
        figure=fig
    )
])

import dash
from dash import html, dcc, Input, Output, State

import plotly.graph_objs as go
import networkx as nx
import pandas as pd
from dash.dependencies import Input, Output, State

# ... (previous code remains the same)

# Global variables to store the last clicked node
last_clicked_node = None

@app.callback(
    Output('knowledge-graph', 'figure'),
    [Input('knowledge-graph', 'clickData'),
     Input('knowledge-graph', 'hoverData'),
     Input('knowledge-graph', 'relayoutData')],
    [State('knowledge-graph', 'figure')]
)
def update_graph(clickData, hoverData, relayoutData, current_figure):
    global last_clicked_node

    # Determine which input triggered the callback
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Reset the graph when the mouse hovers away from nodes after a click
    if trigger_id == 'knowledge-graph' and not hoverData and last_clicked_node:
        last_clicked_node = None
        return go.Figure(data=[edge_trace, node_trace], layout=current_figure['layout'])

    # Handle node click event
    if trigger_id == 'knowledge-graph' and clickData:
        clicked_node = clickData['points'][0]['text']

        # Reset the graph if the same node is clicked again
        if clicked_node == last_clicked_node:
            last_clicked_node = None
            return go.Figure(data=[edge_trace, node_trace], layout=current_figure['layout'])

        # Update the graph based on the clicked node
        fig = go.Figure(data=[edge_trace, node_trace], layout=current_figure['layout'])

        # Logic for handling click on a node
        connected_nodes = list(G.neighbors(clicked_node))
        connected_edges = [(u, v) for u, v in G.edges() if u == clicked_node or v == clicked_node]

        # Highlight connected nodes
        for node in connected_nodes:
            fig.add_trace(go.Scatter(
                x=[pos[node][0]],
                y=[pos[node][1]],
                text=[node],
                mode='markers+text',
                textposition='bottom center',
                hoverinfo='text',
                marker=dict(
                    size=10,
                    color='lightblue',
                    line=dict(width=2)
                ),
                showlegend=False
            ))

        # Highlight connected edges
        for edge in connected_edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=2, color='blue'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))

        # Update the last clicked node
        last_clicked_node = clicked_node

        return fig

    return current_figure


   
         










if __name__ == '__main__':
    #app.run_server(debug=True,)
    app.run_server(debug=True, port=8097)


# In[ ]:




