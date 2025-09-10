#Chat GPt is used to create docstrings of functions and somewhat to clean the function

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Arial'

#water balance plot

def wb_plot(results, save_plot: bool = False, plot_name: str = 'wb_plot.jpeg', print_wb: bool = True):
    """
    Plots and visualizes different water balance components and optionally saves the plot.

    Parameters:
    -----------
    results : pandas.DataFrame
        A DataFrame containing the water balance data. The DataFrame is expected to have the following columns:
        - Ks: Soil water depletion coefficient
        - ETc: Crop evapotranspiration
        - ETa: Actual crop evapotranspiration
        - Rain: Rainfall
        - Irrig: Irrigation
        - Runoff: Runoff
        - TAW: Total available water
        - RAW: Readily available water
        - Dr: Depletion of soil water
        - DP: Deep percolation
        - DOY (Day of Year): Day of year, typically used as the index or a column for x-axis.
    
    save_plot : bool, optional, default: False
        If True, the plot will be saved to a file with the specified `plot_name`.

    plot_name : str, optional, default: 'wb_plot.jpeg'
        The filename for the saved plot. The file will be saved in the current working directory.
        Ensure that the `plot_name` includes the appropriate file extension (e.g., '.jpeg', '.png').

    print_wb : bool, optional, default: True
        If True, a summary of the water balance components will be displayed as the plot's title.

    Returns:
    --------
    None
        The function displays the plot in the current output cell (if using a Jupyter notebook) or in a separate window (if using a script).
        If `save_plot` is True, the plot is saved to the specified file.

    Notes:
    ------
    - The `results` DataFrame must contain all the necessary columns; otherwise, the function will raise an error.
    - The plot is composed of three subplots:
        1. The first subplot shows Ks, ETc, and adjusted ETc.
        2. The second subplot shows Rainfall, Irrigation, and Runoff.
        3. The third subplot shows TAW, RAW, soil water depletion, and deep percolation.
    - The function uses a monospaced font for the plot title to ensure proper alignment of the water balance summary.
    """
    sns.set_style('ticks')

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), dpi=250, sharex=True)

    # ===============================================
    # First subplot (Ks, ETc, and adjusted ETc)
    # ===============================================
    axes[0].plot(results.iloc[:, 1], results['Ks'], color='green', ls='--', label='Ks')
    axes[0].set_ylabel(r'$K_{s}$')
    ax01 = axes[0].twinx()
    ax01.plot(results.iloc[:, 1], results['ETc'], color='coral', label='ETc')
    ax01.plot(results.iloc[:, 1], results['ETa'], color='olive', label='ETa')
    ax01.set_ylabel(r'$ET_{c}$ & $ET_{a}$ (mm)')
    ax01.set_xticks([])

    # ===============================================
    # Second subplot (Rainfall, Irrigation, and Runoff)
    # ===============================================
    axes[1].bar(results.iloc[:, 1], results['Rain'], color='dodgerblue', alpha=0.6, label='Rainfall')
    axes[1].bar(results.iloc[:, 1], results['Irrig'], color='green', alpha=0.6, label='Irrigation')
    axes[1].bar(results.iloc[:, 1], results['Runoff'], color='yellow', label='Runoff')
    axes[1].set_ylabel('Rainfall & Runoff (mm)')
    axes[1].set_xticks([])

    # ===============================================
    # Third subplot (TAW, RAW, soil water depletion, and deep percolation)
    # ===============================================
    axes[2].set_ylim(results['TAW'].iloc[-1] + 10, 0)
    axes[2].plot(results.iloc[:, 1], results['TAW'], color='blue', label='TAW')
    axes[2].plot(results.iloc[:, 1], results['RAW'], color='darkslategrey', lw=2, label='RAW')
    axes[2].plot(results.iloc[:, 1], results['Dr'], color='red', alpha=0.7, label='Dr')
    axes[2].bar(results.iloc[:, 1], results['DP'], color='goldenrod', label='Percolation')
    axes[2].set_xlabel('DOY')
    axes[2].set_ylabel('TAW, RAW, Dr & DP (mm)')

    # ===============================================
    # Set x-ticks at regular intervals (every 10 days, for example)
    # ===============================================
    try:
        x_tic = [day.split('-')[-1] for day in results.iloc[:, 1]]
        tick_positions = range(0, len(x_tic), 10)

        for ax in axes:
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([x_tic[i] for i in tick_positions], rotation=45)
    except Exception as e:
        print(f"An error occurred while setting x-ticks: {e}")
        for ax in axes:
            ax.set_xticks([])

    # ===============================================
    # Collecting all legend handles and labels
    # ===============================================
    handles, labels = [], []

    # First subplot legends (primary and secondary axis)
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = ax01.get_legend_handles_labels()
    handles.extend(h1 + h2)
    labels.extend(l1 + l2)

    # Second subplot legend
    h3, l3 = axes[1].get_legend_handles_labels()
    handles.extend(h3)
    labels.extend(l3)

    # Third subplot legend
    h4, l4 = axes[2].get_legend_handles_labels()
    handles.extend(h4)
    labels.extend(l4)

    # ===============================================
    # Creating a common legend above the subplots
    # ===============================================
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.16, 1.0), fontsize=10)

    # ===============================================
    # Print water balance summary in plot title
    # ===============================================
    if print_wb:
        plt.suptitle(f'''ETc = {round(results.ETc.sum(), 2)}, ETa = {round(results.ETa.sum(), 2)}
Rain = {round(results.Rain.sum(), 2)}, Irrig. = {round(results.Irrig.sum(), 2)}, Irrig. count = {(results['Irrig'] != 0).sum()}
Runoff = {round(results.Runoff.sum(), 2)}, Percolation = {round(results.DP.sum(), 2)}''', fontfamily='monospace')

    # ===============================================
    # Save the plot if required
    # ===============================================
    if save_plot:
        plt.savefig(plot_name, bbox_inches='tight')

    # ===============================================
    # Display the plot
    # ===============================================
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make space for the title and legend
    return plt.show()


#weather plot

def weather_plot(data: pd.DataFrame, start: int, end: int, save_plot: bool = False, plot_name: str = 'climate_plot.jpeg'):
    """
    Plots and visualizes climate data components (temperature, humidity, rainfall, wind speed, and solar radiation)
    for a specified year and day range, and optionally saves the plot.

    Parameters:
    -----------
    data : pandas.DataFrame
        A DataFrame containing the climate data with the following columns:
        - tmmx: Maximum temperature
        - tmmn: Minimum temperature
        - rmax: Maximum relative humidity
        - rmin: Minimum relative humidity
        - pr: Precipitation (rainfall)
        - vs: Wind speed
        - srad: Solar radiation
        The DataFrame index is expected to be a date-time index corresponding to the day of the year (DOY).
    
    start : int
        The start day of the period of interest (DOY).
    
    end : int
        The end day of the period of interest (DOY).

    save_plot : bool, optional, default: False
        If True, the plot will be saved to a file with the specified `plot_name`.
    
    plot_name : str, optional, default: 'climate_plot.jpeg'
        The filename for the saved plot. The file will be saved in the current working directory.
        Ensure that the `plot_name` includes the appropriate file extension (e.g., '.jpeg', '.png').

    Returns:
    --------
    None
        The function displays the plot in the current output cell (if using a Jupyter notebook) or in a separate window (if using a script).
        If `save_plot` is True, the plot is saved to the specified file.

    Notes:
    ------
    - The `data` DataFrame must contain all the necessary columns and be indexed by a date-time index.
    - The function automatically adjusts the data range for leap years.
    - The plot consists of five subplots:
        1. Maximum and Minimum Temperature (°C)
        2. Maximum and Minimum Relative Humidity (%)
        3. Rainfall (mm) and Cumulative Rainfall (mm) with a twin axis
        4. Wind Speed (m/s)
        5. Solar Radiation (W/m²)
    """
    #Retrieving year of interest
    year = int(start[:4])
    
    # Adjust the data for leap year if necessary
    if year % 4 == 0:
        data = data.loc[f'{year}-01':f'{year}-366']
    else:
        data = data.loc[f'{year}-01':f'{year}-365']
    
    sns.set_style('ticks')
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), dpi=300, sharex=True)

    # Plot temperature data with °C symbol and appropriate colors
    sns.lineplot(data=data, x=data.index, y='tmmx', ax=axes[0], label='Max. Temp. (°C)', color='red')
    sns.lineplot(data=data, x=data.index, y='tmmn', ax=axes[0], label='Min. Temp. (°C)', color='blue')
    add_vertical_lines(axes[0], [start, end], ['Planting Date', 'Maturity Date'], y_position=data['tmmn'].min())

    # Plot relative humidity data with appropriate colors
    sns.lineplot(data=data, x=data.index, y='rmax', ax=axes[1], label='Max. RH (%)', color='green')
    sns.lineplot(data=data, x=data.index, y='rmin', ax=axes[1], label='Min. RH (%)', color='orange')
    add_vertical_lines(axes[1], [start, end], ['Planting Date', 'Maturity Date'], y_position=data['rmin'].min())

    # Plot cumulative rain with barplot and lineplot on twin axis, with appropriate colors
    ax03 = axes[2].twinx()
    sns.barplot(data=data, x=data.index, y='pr', ax=axes[2], width=2, label='Rainfall (mm)', color='skyblue')
    sns.lineplot(data=data, x=data.index, y=data['pr'].cumsum(), ax=ax03, label='Cumulative Rainfall (mm)', color='darkblue')

    # Plot cumulative rain for the specific range (start:end)
    subset = data.loc[start:end]  
    sns.lineplot(data=subset, x=subset.index, y=subset['pr'].cumsum(), ax=ax03, ls='--', label='Cum. Rainfall (crop period)', color='darkgreen')
    ax03.text(subset.index[-1], subset['pr'].sum(), f'{subset["pr"].sum()} mm', verticalalignment='bottom', horizontalalignment='center', fontsize=8)
    add_vertical_lines(axes[2], [start, end], ['Planting Date', 'Maturity Date'], y_position=data['pr'].min())

    # Plot wind speed data with appropriate color
    sns.lineplot(data=data, x=data.index, y='vs', ax=axes[3], label='Wind Speed (m/s)', color='purple')
    add_vertical_lines(axes[3], [start, end], ['Planting Date', 'Maturity Date'], y_position=data['vs'].min())

    # Plot solar radiation data with appropriate color
    sns.lineplot(data=data, x=data.index, y='srad', ax=axes[4], label='Solar Rad. (W/m²)', color='gold')
    add_vertical_lines(axes[4], [start, end], ['Planting Date', 'Maturity Date'], y_position=data['srad'].min())

    # ===============================================
    # Consolidate Legends
    # ===============================================
    # Gather all legend handles and labels from each axis
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)
        ax.get_legend().remove()  # Remove individual legends
    handles2, labels2 = ax03.get_legend_handles_labels()
    ax03.get_legend().remove()

    # Set x and y-labels for each subplot
    for ax, label in zip(axes, ['Temperature (°C)', 'RH (%)', 'Rainfall (mm)', 'Wind speed (m/s)', 'Solar radiation (W/m²)']):
        ax.set_ylabel(label)
        ax.set_xlabel('')
    ax03.set_ylabel('Cum. rainfall (mm)')
    axes[-1].set_xlabel('DOY')

    # Create a single legend on the center-right
    fig.legend(handles + handles2, labels + labels2, loc=(0.8, 0.7), fontsize=8)

    # ===============================================
    # Adjust Tick Intervals
    # ===============================================
    # Set x-ticks at intervals of 50
    try:
        for ax in axes:
            x_tic = [day.split('-')[-1] for day in data.index]
            # Determine tick positions
            tick_positions = range(0, len(x_tic), 15)

            # Set x-ticks at these positions and set corresponding labels
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([x_tic[i] for i in tick_positions], rotation=45)
    except Exception as e:
        print(f"An error occurred while setting x-ticks: {e}")
        for ax in axes:
            ax.set_xticks([])  # Set x-ticks to empty if an error occurs


    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for the legend
    
    # Save plot if requested
    if save_plot:
        plt.savefig(plot_name, bbox_inches='tight')

    # Display the plot
    plt.show()

def add_vertical_lines(ax, x_positions, labels, y_position, **kwargs):
    """
    Adds vertical lines and text annotations to a given axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to add the lines and text.
        x_positions (list): List of x-positions where the vertical lines should be drawn.
        labels (list): List of labels corresponding to each vertical line.
        y_position (float): The y-coordinate for the text annotations.
        kwargs (dict): Additional arguments to pass to the axvline method.
    """
    for x, label in zip(x_positions, labels):
        ax.axvline(x, ls='--', lw=0.7, color='black')
        ax.text(x, y_position, label, verticalalignment='bottom', 
                horizontalalignment='center', fontsize=8, **kwargs)
        

#Root and plant growth plot


def growth_plot(data: pd.DataFrame, depths: list, par: object, save_plot: bool = False, plot_name: str = 'growth_plot.jpeg'):
    """
    Plots and visualizes plant growth parameters including root depth and plant height over time,
    along with soil layer depths and growth stages. Optionally, saves the plot.

    Parameters:
    -----------
    data : pandas.DataFrame
        A DataFrame containing the plant growth data with the following columns:
        - The second column (data.iloc[:, 1]) is assumed to be time or day of the year (DOY).
        - Zr: Root depth in meters.
        - h: Plant height in meters.
    
    depths : list
        A list of soil layer depths in centimeters to be marked as horizontal lines.
    
    par : object
        An object containing growth stage parameters:
        - par.Lini: Initial stage length in days.
        - par.Ldev: Development stage length in days.
        - par.Lmid: Mid-stage length in days.
        - par.Lend: End stage length in days.
    
    save_plot : bool, optional, default: False
        If True, the plot will be saved to a file with the specified `plot_name`.
    
    plot_name : str, optional, default: 'growth_plot.jpeg'
        The filename for the saved plot. The file will be saved in the current working directory.
        Ensure that the `plot_name` includes the appropriate file extension (e.g., '.jpeg', '.png').

    Returns:
    --------
    None
        The function displays the plot in the current output cell (if using a Jupyter notebook) or in a separate window (if using a script).
        If `save_plot` is True, the plot is saved to the specified file.

    Notes:
    ------
    - The DataFrame `data` must contain the necessary columns (`Zr`, `h`).
    - The `par` object should contain the stage lengths for initial, development, mid, and end stages.
    - Soil layers and growth stages are indicated by horizontal and vertical lines, respectively.
    """

    sns.set_style('ticks')

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot root depth and plant height
    sns.lineplot(data=data, x=data.iloc[:, 1], y=100 * data.Zr, label='Root depth (cm)', color='darkgoldenrod', ax=ax)
    sns.lineplot(data=data, x=data.iloc[:, 1], y=100 * data.h, alpha=0.7, label='Plant height (cm)', color = 'limegreen', ax=ax)
    ax.set_ylabel('Root depth & plant height (cm)')

    # Plot soil layer depths as horizontal lines
    for i, depth in enumerate(depths):
        ax.axhline(depth, ls='--', lw=0.7, color='brown')
        ax.text(par.Lini - 20, depth, f'Soil layer {i + 1}: {depth} cm', verticalalignment='bottom',
                horizontalalignment='center', fontsize=8)

    # Plot growth stages as vertical lines
    stage_lengths = [par.Lini, par.Lini + par.Ldev, par.Lini + par.Ldev + par.Lmid, par.Lini + par.Ldev + par.Lmid + par.Lend]
    stage_labels = ['Ini. stage', 'Dev. stage', 'Mid stage', 'End stage']
    stage_durations = [par.Lini, par.Ldev, par.Lmid, par.Lend]

    for st_len, label, pst_len in zip(stage_lengths, stage_labels, stage_durations):
        ax.axvline(st_len, ls='--', lw=0.7, color='green')
        ax.text(st_len, 0.4 * depths[0], f'{label} ({pst_len} days)', rotation=90, verticalalignment='bottom',
                horizontalalignment='right', fontsize=8)

    # Set x-ticks with error handling
    try:
        x_tic = [day.split('-')[-1] for day in data.iloc[:,1]]
        # Determine tick positions
        tick_positions = range(0, len(x_tic), 10)

        # Set x-ticks at these positions and set corresponding labels
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([x_tic[i] for i in tick_positions], rotation=45)
    except Exception as e:
        print(f"An error occurred while setting x-ticks: {e}")
        ax.set_xticks([])  # Set x-ticks to empty if an error occurs

    # Consolidate the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    fig.legend(handles, labels, ncols=2, loc=(0.4, 0.95), fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save plot if requested
    if save_plot:
        plt.savefig(plot_name, bbox_inches='tight')

    # Display the plot
    plt.show()


#Kc plot


def kc_plot(results, secondary_axis: bool = True, save_plot: bool = False, plot_name: str = 'kc_plot.jpeg'):
    """
    Plots and visualizes Kcb, Kc, and Ke components from the results DataFrame, 
    with an optional secondary axis for Rainfall and Irrigation data.

    Parameters:
    -----------
    results : pandas.DataFrame
        A DataFrame containing the water balance data. The DataFrame is expected to have the following columns:
        - Kcb: Basal crop coefficient
        - Kc: Crop coefficient
        - Ke: Soil evaporation coefficient
        - Rain: Rainfall (optional, for the secondary axis)
        - Irrig: Irrigation (optional, for the secondary axis)
        - The second column is typically used for the x-axis (e.g., Day of Year, DOY).

    secondary_axis : bool, optional, default: True
        If True, the function will plot the secondary axis with Rainfall and Irrigation bars.
        If False, only the line plots for Kcb, Kc, and Ke will be shown.

    save_plot : bool, optional, default: False
        If True, the plot will be saved to a file with the specified `plot_name`.

    plot_name : str, optional, default: 'kc_plot.jpeg'
        The filename for the saved plot. Ensure that the `plot_name` includes the appropriate file extension (e.g., '.jpeg', '.png').

    Returns:
    --------
    None
        The function displays the plot in the current output cell (if using a Jupyter notebook) or in a separate window (if using a script).
        If `save_plot` is True, the plot is saved to the specified file.

    Notes:
    ------
    - The `results` DataFrame must contain the necessary columns (Kcb, Kc, Ke).
    - If `secondary_axis` is True, the DataFrame must also contain the Rain and Irrigation columns.
    """
    
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Plotting the secondary axis (Rainfall and Irrigation) if secondary_axis is True
    if secondary_axis:
        ax1 = ax.twinx()
        sns.barplot(data=results, x=results.iloc[:, 1], y='Irrig', alpha=0.4, color='dodgerblue', ax=ax1, label='Irrigation')
        sns.barplot(data=results, x=results.iloc[:, 1], y='Rain', alpha=0.4, color='green', ax=ax1, label='Rainfall')
        ax1.set_ylabel('Rainfall & Irrigation (mm)')
        ax1.set_xlabel('')

    # Plotting the line plots for Kcb, Kc, and Ke
    sns.lineplot(data=results, x=results.iloc[:, 1], y='Kcb', label='Kcb', color='lawngreen', ax=ax)
    sns.lineplot(data=results, x=results.iloc[:, 1], y='Kc', label='Kc', color='skyblue', ax=ax)
    sns.lineplot(data=results, x=results.iloc[:, 1], y='Ke', label='Ke', ls='--', color='peru', ax=ax)
    ax.set_ylabel('Ke, Kcb, & Kc')

    try:
        # Set x-ticks at regular intervals
        x_tic = [day.split('-')[-1] for day in results.iloc[:, 1]]
        tick_positions = range(0, len(x_tic), 10)  # Setting ticks every 10 intervals
    
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([x_tic[i] for i in tick_positions], rotation=45)
    except Exception as e:
        print(f"An error occurred while setting x-ticks: {e}")
        ax.set_xticks([])

    # Consolidate the legend
    handles, labels = [], []
    h1, l1 = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    handles.extend(h1)
    labels.extend(l1)
    if secondary_axis is True:
        ax1.get_legend().remove()
        h2, l2 = ax1.get_legend_handles_labels()
        handles.extend(h2)
        labels.extend(l2)
    fig.legend(handles, labels, ncols=3, loc=(0.4, 0.94), fontsize=8)

    # Save the plot if needed
    if save_plot:
        plt.savefig(plot_name, bbox_inches='tight')

    # Display the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return plt.show()
