import os

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import torch

from core.data import open_data
from core.data import remove_quartiles, removing_rows_w_missing, data_split, select_data, Scaler

def makeplot(well:pd.DataFrame, top_depth:float, bottom_depth:float, model_name:str, selected_well:str, output_dir:str) -> None:
    """    
    Function that creates and saves plot
        Arguments:
        ---------
            - well (pd.DataFrame): Well log data from a single well
            - top_depth (float): Maximum depth of the well
            - bottom_depth (float): Minimum depth of the well
            - model_name (str): Name of the model used
            - selected_well (str): Name of the selected well
            - output_dir (str): Path to save plot.
        Return:
        ---------
            None
    """
    
    lithology_numbers = {30000: {'lith':'Sandstone', 'lith_num':1, 'hatch': '..', 'color':'#ffff00'},
                 65030: {'lith':'Sandstone/Shale', 'lith_num':2, 'hatch':'-.', 'color':'#ffe119'},
                 65000: {'lith':'Shale', 'lith_num':3, 'hatch':'--', 'color':'#bebebe'},
                 80000: {'lith':'Marl', 'lith_num':4, 'hatch':'', 'color':'#7cfc00'},
                 74000: {'lith':'Dolomite', 'lith_num':5, 'hatch':'-/', 'color':'#8080ff'},
                 70000: {'lith':'Limestone', 'lith_num':6, 'hatch':'+', 'color':'#80ffff'},
                 70032: {'lith':'Chalk', 'lith_num':7, 'hatch':'..', 'color':'#80ffff'},
                 88000: {'lith':'Halite', 'lith_num':8, 'hatch':'x', 'color':'#7ddfbe'},
                 86000: {'lith':'Anhydrite', 'lith_num':9, 'hatch':'', 'color':'#ff80ff'},
                 99000: {'lith':'Tuff', 'lith_num':10, 'hatch':'||', 'color':'#ff8c00'},
                 90000: {'lith':'Coal', 'lith_num':11, 'hatch':'', 'color':'black'},
                 93000: {'lith':'Basement', 'lith_num':12, 'hatch':'-|', 'color':'#ef138a'}}
    
    fig, ax = plt.subplots(figsize=(15,10))

    #Set up the plot axes
    ax1 = plt.subplot2grid((1,6), (0,0), rowspan=1, colspan = 1)
    ax2 = plt.subplot2grid((1,6), (0,1), rowspan=1, colspan = 1, sharey = ax1)
    ax3 = plt.subplot2grid((1,6), (0,2), rowspan=1, colspan = 1, sharey = ax1) #Twins the y-axis for the density track with the neutron track
    ax4 = plt.subplot2grid((1,6), (0,3), rowspan=1, colspan = 1, sharey = ax1)
    ax5 = plt.subplot2grid((1,6), (0,4), rowspan=1, colspan = 1, sharey = ax1)
    ax6 = plt.subplot2grid((1,6), (0,5), rowspan=1, colspan = 1, sharey = ax1)
    
    # As our curve scales will be detached from the top of the track,
    # this code adds the top border back in without dealing with splines
    ax10 = ax1.twiny()
    ax10.xaxis.set_visible(False)
    ax11 = ax2.twiny()
    ax11.xaxis.set_visible(False)
    ax12 = ax3.twiny()
    ax12.xaxis.set_visible(False)
    ax13 = ax4.twiny()
    ax13.xaxis.set_visible(False)
    ax14 = ax5.twiny()
    ax14.xaxis.set_visible(False)
    ax15 = ax6.twiny()
    ax15.xaxis.set_visible(False)
    
    # Gamma Ray track
    ax1.plot(well["GR"], well['DEPTH_MD'], color = "green", linewidth = 0.5)
    ax1.set_xlabel("Gamma")
    ax1.xaxis.label.set_color("green")
    #ax1.set_xlim(0, 200)
    ax1.set_ylabel("Depth (m)")
    ax1.tick_params(axis='x', colors="green")
    ax1.spines["top"].set_edgecolor("green")
    ax1.title.set_color('green')
    #ax1.set_xticks([0, 50, 100, 150, 200])

    # Density track
    ax2.plot(well["RHOB"], well['DEPTH_MD'], color = "red", linewidth = 0.5)
    ax2.set_xlabel("Density")
    #ax2.set_xlim(1.95, 2.95)
    ax2.xaxis.label.set_color("red")
    ax2.tick_params(axis='x', colors="red")
    ax2.spines["top"].set_edgecolor("red")
    #ax2.set_xticks([1.95, 2.45, 2.95])

    # Neutron track placed ontop of density track
    ax3.plot(well["NPHI"], well['DEPTH_MD'], color = "blue", linewidth = 0.5)
    ax3.set_xlabel('Neutron')
    ax3.xaxis.label.set_color("blue")
    #ax3.set_xlim(0.45, -0.15)
    ax3.tick_params(axis='x', colors="blue")
    #ax3.spines["top"].set_position(("axes", 1.08))
    #ax3.spines["top"].set_visible(True)
    ax3.spines["top"].set_edgecolor("blue")
    #ax3.set_xticks([0.45,  0.15, -0.15])
    
    # Neutron track placed ontop of density track
    ax4.plot(well["DTC"], well['DEPTH_MD'], color = "purple", linewidth = 0.5)
    ax4.set_xlabel('Compressional Wave')
    ax4.xaxis.label.set_color("purple")
    #ax3.set_xlim(0.45, -0.15)
    ax4.tick_params(axis='x', colors="purple")
    #ax3.spines["top"].set_position(("axes", 1.08))
    #ax3.spines["top"].set_visible(True)
    ax4.spines["top"].set_edgecolor("purple")
    #ax3.set_xticks([0.45,  0.15, -0.15])

    # Lithology track
    ax5.plot(well["FORCE_2020_LITHOFACIES_LITHOLOGY"], well['DEPTH_MD'], color = "black", linewidth = 0.5)
    ax5.set_xlabel("Lithology")
    ax5.set_xlim(0, 1)
    ax5.xaxis.label.set_color("black")
    ax5.tick_params(axis='x', colors="black")
    ax5.spines["top"].set_edgecolor("black")
    
    ax6.plot(well["LITHOLOGY_PREDICTED"], well['DEPTH_MD'], color = "black", linewidth = 0.5)
    ax6.set_xlabel(f"Lithology {model_name}")
    ax6.set_xlim(0, 1)
    ax6.xaxis.label.set_color("black")
    ax6.tick_params(axis='x', colors="black")
    ax6.spines["top"].set_edgecolor("black")

    for key in lithology_numbers.keys():
        color = lithology_numbers[key]['color']
        hatch = lithology_numbers[key]['hatch']
        ax5.fill_betweenx(well['DEPTH_MD'], 0, well['FORCE_2020_LITHOFACIES_LITHOLOGY'], where=(well['FORCE_2020_LITHOFACIES_LITHOLOGY']==key),
                         facecolor=color, hatch=hatch)
        ax6.fill_betweenx(well['DEPTH_MD'], 0, well['LITHOLOGY_PREDICTED'], where=(well['LITHOLOGY_PREDICTED']==key),
                         facecolor=color, hatch=hatch)
        

    ax5.set_xticks([0, 1])
    ax6.set_xticks([0, 1])

    # Common functions for setting up the plot can be extracted into
    # a for loop. This saves repeating code.
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_ylim(bottom_depth, top_depth)
        ax.grid(which='major', color='lightgrey', linestyle='-')
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.spines["top"].set_position(("axes", 1.02))
        
        
    for ax in [ax2, ax3, ax4, ax5, ax6]:
        plt.setp(ax.get_yticklabels(), visible = False)
        
    plt.tight_layout()
    fig.subplots_adjust(wspace = 0.15)
    
    selected_well = selected_well.replace("/", "_")
    selected_well = selected_well.replace("\\", "_")
    filename_plot = f'{selected_well}_{model_name}.png'
    fig.savefig(os.path.join(output_dir, filename_plot))


def plot_results(cfg:dict, model, scaler, model_name:str, test_wells:list[str], output_dir:str, device:str, le=None):
    """
    Open and preprocess data to make the plot.
        Arguments:
        ---------
            - cfg (dict): Dictionary containig config details
            - model: Model object
            - scaler (Scaler): Scaler object
            - model_name (str): Name of the model used
            - test_wells (list[str]): List of test well names
            - output_dir (str): Path to save plot
            - le (sklearn.preprocessing.LabelEncoder): Label encoder object. Used only for XGBoost, since it requires encoding when labels are not consecutive in training.
        Return:
        ---------
            None
    """
    
    data, le = open_data(dataset_name, data_dir, logs, verbose=verbose)
    
    data = remove_quartiles(data, cfg['logs'], verbose=False)
    data = removing_rows_w_missing(data, cfg['logs'], cfg['class_col'])
    
    selected_well = random.choice(test_wells)
    well = data[data['WELL'] == selected_well]
    
    if len(well[cfg['class_col']]) % cfg['seq_size'] != 0:
            well = well[:-(len(well[cfg['class_col']]) % cfg['seq_size'])]
    
    y = well[cfg['class_col']]
    
    x = well.copy()
    x = x[cfg['logs']]
    
    x[cfg['logs']] = scaler.transform(x[cfg['logs']])
    
    if cfg['input_format'] == 'dl':

        x = x.to_numpy()
        x = torch.from_numpy(x).float()
        # Correct calculation for the number of rows to skip
        # First, we exclude the last 50 rows from our calculation
        total_rows = x.size(0)
        rows_excluding_last_n = total_rows - cfg['seq_size']

        # Then we find out how many complete 50-row segments we can have from the remaining rows
        complete_segments = rows_excluding_last_n // cfg['seq_size']

        # Now, calculate the starting point for slicing
        # It's the total rows minus the rows we want to include in our reshaped tensor
        starting_row = total_rows - (complete_segments * cfg['seq_size'] + cfg['seq_size'])

        # Slice the tensor from the calculated starting row
        sliced_tensor = x[starting_row:]

        # Reshape the tensor
        reshaped_tensor = sliced_tensor.view(-1, cfg['seq_size'], 4)

        # Check the new shape
        model.eval()
        reshaped_tensor = reshaped_tensor.long().to(device)

        output, probs = model.forward(reshaped_tensor)

        probs = probs.view(-1, cfg['num_classes']).detach().cpu().numpy()  # Reshape to 2D tensor (batch_size * seq_len, num_classes)

        y_predict = np.argmax(probs, axis=1).tolist()
        y_predict = le.inverse_transform(y_predict)
        
    else:
        
        y_predict = model.predict(x)
        y_predict = le.inverse_transform(y_predict)
    
    y_predict = le_dataset.inverse_transform(y_predict)
    well['LITHOLOGY_PREDICTED'] = y_predict
    
    makeplot(well, np.min(well['DEPTH_MD']), max(well['DEPTH_MD']), model_name, selected_well, output_dir)
