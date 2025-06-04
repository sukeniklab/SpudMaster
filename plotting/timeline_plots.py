import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_timeseries(df, excluded_timepoint: list[int] = [0], time_interval = 6, group_values = ['construct'], dpi_resolution = 500,
                   )
    
    
    df = df.copy()
    for inter in excluded_timepoint:
        df = df.loc[df['timepoint'] != inter]
        
    df.loc[:, 'timepoint'] = (df['timepoint'] * time_interval)
    generator = df.groupby('construct', observed=True)
    mpl.rcParams['figure.dpi'] = dpi_resolution
    fig_len = df.construct.unique()
    fig, ax =plt.subplots(fig_len,  1, figsize=(np.ceil((fig_len+1)/2), fig_len+1)) 
    for index, container in enumerate(generator):
        construct = container[0]
        
        if index < fig_len:
            container_df = container[1]
            
            
            indexer = 0
            colors = ['k', 'dodgerblue', 'firebrick']
            for parameter in container_df.experimentparameter.unique():
                
                current_parameter = container_df.loc[container_df['experimentparameter']== parameter]
                
                
                ax[index].scatter(x=current_parameter['timepoint'], y=current_parameter['delta_Efret_median'], alpha=0.6, c=colors[indexer], zorder=3)
                ax[index].errorbar(current_parameter['timepoint'], current_parameter['delta_Efret_median'], yerr = current_parameter['std_median_per_timepoint'], c=colors[indexer],
                                                        capsize=2, capthick=2, alpha=0.6)
                indexer += 1
            ax[index].set_ylim(-0.06, 0.045)
            ax[index].set_yticks([ -0.045, -0.03, -0.015, 0, 0.015, 0.03])
            ax[index].set_xticks([])
            ax[index].axvline(x=24, c='k', linestyle='--')
            ax[index].axhline(y=0.0, c='k', linestyle='--')
            ax2 = ax[index].twinx()
            ax2.set_yticks([])
            ax2.set_ylabel(ylabel = 'Tile ' + str(construct))
    
    tick_array = np.array(current_parameter['timepoint'])
    ax[-1].set_xticks(tick_array)
    fig.suptitle('Heatshock Experiment Timeline', fontsize=10, x=0.5, y=0.89)
    plt.subplots_adjust(hspace=0)
    plt.savefig(r'D:\OneDrive\Virus\Paper1\Paper1_figures\Violin_figures\infection_timeline_figure_percentile-0.15.png', format='png', bbox_inches='tight')