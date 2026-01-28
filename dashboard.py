import pandas as pd
import plotly.express as px
import panel as pn
import plotly.graph_objects as go
import numpy as np
pn.extension("plotly")


detailed_results = pd.read_csv("detailed_coverage_results.csv")  

coverages_columns = [c for c in detailed_results.columns if "coverages" in c.lower()]

column_selector = pn.widgets.Select(name="Select Column", options=coverages_columns)
def create_coverage_plot(column_name):
    col_data = detailed_results[column_name].dropna()
    
    # Histogram bins
    # Histogram bins
    bin_edges = np.histogram_bin_edges(col_data, bins='auto')
    
    # Compute counts per bin
    bin_counts, _ = np.histogram(col_data, bins=bin_edges)
    
    # Compute bin midpoints
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Color bins: blue if **any part of bin > 0.95**, else black
    colors = ["blue" if i > 0.95 else "black" for i in bin_midpoints]
    
    # Create histogram bars manually for custom colors
    fig = go.Figure()
    for i in range(len(bin_edges)-1):
        fig.add_trace(go.Bar(
            x=[(bin_edges[i]+bin_edges[i+1])/2],
            y=[bin_counts[i]],
            width=[bin_edges[i+1]-bin_edges[i]],
            marker_color=colors[i],
            showlegend=False
        ))
    
    # Add red line at 0.95
    fig.add_vline(x=0.95, line_color="red", line_dash="dash", line_width=2)

    fig.update_layout(
        title=f"Coverage Analysis: {column_name}",
        xaxis_title=column_name,
        yaxis_title="Count",
        template="plotly_white",
        bargap=0.05
    )
    return fig

# Bind plot to dropdown
plot_pane = pn.bind(create_coverage_plot, column_name=column_selector)

# Panel subsection
subsection = pn.Column(
    "## Coverage Column Analysis",
    column_selector,
    plot_pane
)

subsection.servable()