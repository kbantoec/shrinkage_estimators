from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure
from bokeh.layouts import row
import pandas as pd

main_dir: str = 'C:/Users/YBant/Desktop/backtests_1605/'
all_120e: pd.DataFrame = pd.read_feather(f'{main_dir}data/navs/mean_variance_equally_weighted_market_cap_'
                                         f'weighted_120_excess_simple_sample_constant_correlation_'
                                         f'single_index_identity.feather').set_index('date').mul(100)
all_30e: pd.DataFrame = pd.read_feather(f'{main_dir}data/navs/mean_variance_equally_weighted_market_cap_'
                                        f'weighted_30_excess_simple_sample_constant_correlation_'
                                        f'single_index_identity.feather').set_index('date').mul(100)

all_e: dict = dict(all_30e=all_30e, all_120e=all_120e)

source = ColumnDataSource(data={
        'x': all_120e.index,
        'y': all_e['all_120e'].mean_variance_120_portfolio_excess_simple_returns_sample})

# Save the minimum and maximum values of the index: xmin, xmax
xmin, xmax = all_120e.index.min(), all_120e.index.max()

# Save the minimum and maximum values of the column: ymin, ymax
ymin = all_120e.mean_variance_120_portfolio_excess_simple_returns_sample.min()
ymax = all_120e.mean_variance_120_portfolio_excess_simple_returns_sample.max()

# Create the figure: plot
plot = figure(title='Portfolio Cumulative Excess Simple Returns', plot_height=400, plot_width=700,
              x_range=(xmin, xmax), y_range=(ymin, ymax), x_axis_label='Date', y_axis_label='%')

# Add the color mapper to the circle glyph
plot.line(x='x', y='y', source=source)

# Set the legend.location attribute of the plot to 'top_right'
plot.legend.location = 'top_left'

# Make a slider object: slider
slider = Slider(start=30, end=120, step=90, value=120, title='Rolling Window')


def update_plot(attr, old, new):

    # Read the current value of the slider: scale
    rollwin = slider.value

    # Compute the updated y using np.sin(scale/x): new_y
    new_y = all_e[f'all_{rollwin}e'][f"mean_variance_{rollwin}_portfolio_excess_simple_returns_sample"]

    # Update source with the new data values
    source.data = {'x': all_e[f'all_{rollwin}e'].index, 'y': new_y}

    # Set the range of all axes
    plot.x_range.start = all_e[f'all_{rollwin}e'].index.min()
    plot.x_range.end = all_e[f'all_{rollwin}e'].index.max()
    plot.y_range.start = new_y.min()
    plot.y_range.end = new_y.max()


slider.on_change('value', update_plot)

# Make a row layout of Column(slider) and plot ancd add it to the current document
layout = row(slider, plot)

# Add the plot to the current document and add the title
curdoc().add_root(layout)
curdoc().title = 'Shrinkage Project'
