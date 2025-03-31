import os

import numpy as np
import matplotlib.pyplot as plt


def makeplot(y_true, y_pred, model_name:str, filename:str, output_dir:str) -> None:
    """    
    Function that creates and saves plot
        Arguments:
        ---------
            - y_true (array-like): Ground truth values.
            - y_pred (array-like): Predicted values from the model.
            - model_name (str): Name of the model used for predictions.
            - filename (str): Name of the file to save the plot as.
            - output_dir (str): Directory path where the plot will be saved.
        Return:
        ---------
            None
    """
    
    lithology_numbers = {30000: {'lith':'Sandstone', 'hatch': '..', 'color':'#ffff00'},
                 65030: {'lith':'Sandstone/Shale', 'hatch':'-.', 'color':'#ffe119'},
                 65000: {'lith':'Shale', 'hatch':'--', 'color':'#bebebe'},
                 80000: {'lith':'Marl', 'hatch':'', 'color':'#7cfc00'},
                 74000: {'lith':'Dolomite', 'hatch':'-/', 'color':'#8080ff'},
                 70000: {'lith':'Limestone', 'hatch':'+', 'color':'#80ffff'},
                 70032: {'lith':'Chalk', 'hatch':'..', 'color':'#80ffff'},
                 88000: {'lith':'Halite', 'hatch':'x', 'color':'#7ddfbe'},
                 86000: {'lith':'Anhydrite', 'hatch':'', 'color':'#ff80ff'},
                 99000: {'lith':'Tuff', 'hatch':'||', 'color':'#ff8c00'},
                 90000: {'lith':'Coal', 'hatch':'', 'color':'black'},
                 93000: {'lith':'Basement', 'hatch':'-|', 'color':'#ef138a'}}
    
    fig, ax = plt.subplots(figsize=(6,10))

    #Set up the plot axes
    
    ax5 = plt.subplot2grid((1,2), (0,0), rowspan=1, colspan = 1)
    ax6 = plt.subplot2grid((1,2), (0,1), rowspan=1, colspan = 1, sharey = ax5)
    
    # As our curve scales will be detached from the top of the track,
    # this code adds the top border back in without dealing with splines
    ax14 = ax5.twiny()
    ax14.xaxis.set_visible(False)
    ax15 = ax6.twiny()
    ax15.xaxis.set_visible(False)

    # Lithology track
    ax5.plot(y_true, list(reversed(range(len(y_true)))), color = "black", linewidth = 0.5)
    ax5.set_xlabel("Lithology")
    ax5.set_xlim(0, 1)
    ax5.xaxis.label.set_color("black")
    ax5.tick_params(axis='x', colors="black")
    ax5.spines["top"].set_edgecolor("black")
    
    ax6.plot(y_pred, list(reversed(range(len(y_pred)))), color = "black", linewidth = 0.5)
    ax6.set_xlabel(f"Lithology {model_name}")
    ax6.set_xlim(0, 1)
    ax6.xaxis.label.set_color("black")
    ax6.tick_params(axis='x', colors="black")
    ax6.spines["top"].set_edgecolor("black")

    for key in lithology_numbers.keys():
        color = lithology_numbers[key]['color']
        hatch = lithology_numbers[key]['hatch']
        ax5.fill_betweenx(list(reversed(range(len(y_true)))), 0, y_true, where=(y_true==key),
                         facecolor=color, hatch=hatch)
        ax6.fill_betweenx(list(reversed(range(len(y_pred)))), 0, y_pred, where=(y_pred==key),
                         facecolor=color, hatch=hatch)
        

    ax5.set_xticks([0, 1])
    ax6.set_xticks([0, 1])

    # Common functions for setting up the plot can be extracted into
    # a for loop. This saves repeating code.
    for ax in [ax5, ax6]:
        #ax.set_ylim(bottom_depth, top_depth)
        ax.grid(which='major', color='lightgrey', linestyle='-')
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.spines["top"].set_position(("axes", 1.02))
        
        
    for ax in [ax5, ax6]:
        plt.setp(ax.get_yticklabels(), visible = False)
        
    plt.tight_layout()
    fig.subplots_adjust(wspace = 0.15)
    
    #selected_well = selected_well.replace("/", "_")
    #selected_well = selected_well.replace("\\", "_")
    filename_plot = f'{filename}.png'
    fig.savefig(os.path.join(output_dir, filename_plot))
