import pandas as pd
import matplotlib.pyplot as plt

#What is the distribution of the mean temperatures? Show your results as histograms.

#What is the average mean temperature of the buildings?

#What is the average temperature standard deviation?

#How many buildings had at least 50 percent of their area above 18ºC?

#How many buildings had at least 50 percent of their area below 15ºC?




def analyze_data(df):
    # average mean temperature
    average_mean_temp = df['mean_temp'].mean()
    print(f"Mean_temps: {average_mean_temp}" )

    # average mean temperature standard deviation
    standard_deviation = df['mean_temp'].std()
    print(f"Standard deviation of mean temperatures {standard_deviation}")
    
    # amount of buildings with at least 50% of their area above 18ºC
    buildings_18 = df[df['pct_above_18'] > 50.0]
    #buildings_with_temp_above_18 = len(buildings_with_temp_above_18)
    bd_count = buildings_18.shape[0]
    print(f"Buildings with mean temperature above 18ºC: {bd_count}")

    buildings_15 = df[df['pct_below_15'] > 50.0]
    bd_count = buildings_15.shape[0]
    print(f"Buildings with mean temperature below 15ºC: {bd_count}")


    temperatures = df['mean_temp']
    plot = temperatures.plot.hist(bins=20,
                            x='mean temperature',
                            y='frequency',
                            title='Histogram of Mean Temperature',
                            color='orange',
                            edgecolor='black',
                            zorder = 2)
    
    plot.set_xlabel('Mean Temperature')
    plot.set_ylabel('Frequency')
    plt.grid(visible=True, zorder=0)
    plt.show()
    return
    

    hist = df.hist(data['mean_temp'], bins=20, alpha=0.5, color='blue', edgecolor='black')
    hist.set_title('Histogram of Mean Temperature')
    hist.set_xlabel('Mean Temperature')
    hist.set_ylabel('Frequency')
    hist.show()

    return summary_stats


if __name__ == '__main__':
    data = pd.read_csv('CUDA/CUDAresultsALL.csv')
    analyze_data(data)
