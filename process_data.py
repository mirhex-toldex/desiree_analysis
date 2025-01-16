import pandas as pd
from tdmsdata import TdmsData
import matplotlib.pyplot as plt
import numpy as np
import os
import importlib
import doppler_shift_2024
import statsmodels.api as sm
from scipy import stats 
from scipy.ndimage import gaussian_filter1d


isotope_mapping = {
    'Sn-120': 120,
    'Sn-122': 122,
    'Sn-124': 124,
    'Sn-116': 116,
    'Sn-118': 118,
    'Sn-112': 112,
    'Sn-114': 114,
    'Sn-115': 115,
    'Sn-117': 117,
    'Sn-119': 119
}

def read_tdms(folder_path, file, channel):
    TDMS = TdmsData(''.join([folder_path, file]))
    raw_data = TDMS.get_raw_data(''.join(['Channel ', str(channel)]))  # 2D np array, channel 1 = RAES, channel 3 = ID gated signal, channel 4 = ID gated bkg, channel 5 = nongated
    return raw_data

def create_df(raw_data):
    return pd.DataFrame({'Cycle No.': raw_data[:, 0], 'Time (sec)': raw_data[:, 1], 
                         'Laser Frequency (THz)': raw_data[:, 2], 'Approx Time': raw_data[:, 3], # Approx time is a timestamp for calib and interferometer 
                         'SDUMP': raw_data[:, 4], 'LE Probe': raw_data[:, 5]}) # SDUMP is dump current of PREVIOUS cycle (nA) and LE probe is beam energy (V) 

def doppler_shift_calc(dataset: pd.DataFrame, isotope: int) -> pd.DataFrame:
    importlib.reload(doppler_shift_2024)
    doppler_df = dataset.copy()
    freq = doppler_df['Laser Frequency (THz)']
    measured_voltage = doppler_df['LE Probe']
    shifted_freq = doppler_shift_2024.getshift(freq, isotope, measured_voltage)
    doppler_df['Laser Frequency (THz)'] = shifted_freq
    return doppler_df

def get_start_time(df: pd.DataFrame) -> float:
    time_bins = int(df['Time (sec)'].max()) 
    time_df = (
        df
        .assign(Time_bin=pd.cut(df['Time (sec)'], time_bins))    # Bin 'Time (sec)' and create 'Time_bin' column
        .groupby('Time_bin', observed=True)                               # Aggregate data within each bin
        .size()                                                           # Gives the counts for each bin 
        .reset_index(name='Count raw')                                        # Convert to DataFrame with bin counts
        .assign(Bin_center=lambda df: df['Time_bin'].apply(lambda x: x.mid))  # Calculate bin centers
    )
    time_df.columns = ['Time bin', 'Count raw', 'Bin center']
    time_df['Bin center'] = time_df['Bin center'].astype(float)

    # Filter for bin centers between 0 and 50 s to find inj time 
    time_df_filtered = time_df[(time_df['Bin center'] >= 0) & (time_df['Bin center'] <= 50)]
    max_count = time_df_filtered['Count raw'].max()
    inj_time = time_df_filtered.loc[time_df_filtered['Count raw'].idxmax(), 'Bin center']
    start_time = inj_time + 21 # diode starts scanning 20 seconds after inj 
    return start_time

def dynamic_filters(start_time, doppler_df): # finds end time and filters df
    time = 'Time (sec)'
    freq = 'Laser Frequency (THz)'

    # dynamic filter for beginning of scan
    filtered_df = doppler_df[doppler_df[time] >= start_time] 

    # dynamic filter for end of scan
    df = filtered_df.sort_values(by=[time]).reset_index(drop=True) # first sort by time 

    steps = []
    for i in range(len(df) - 1):
        current_freq = df.loc[i, freq]
        next_freq = df.loc[i + 1, freq]
        step = next_freq - current_freq
        if step != 0:
            steps.append(np.abs(step))

    average_step = np.mean(steps)

    for i in range(len(df) - 1):
        current_freq = df.loc[i, freq]
        next_freq = df.loc[i + 1, freq]
        step = next_freq - current_freq
        if step*-1 < 0 and np.abs(step) >= 2*average_step:  # Sign change condition and threshold to avoid cutting out jumping 
            cutoff_index = i  # Mark the index where the cutoff occurs
            # end_time = df.loc[cutoff_index, time]  # Get the time value at the cutoff index
            filtered_df = df.loc[:cutoff_index]   # Filter the data before the cutoff
            return filtered_df
    return df

def triangle_filters(file, start_time, doppler_df): # for scans that have both freq directions 
    time = 'Time (sec)'
    freq = 'Laser Frequency (THz)'
    cycle_len = 345 # s this is true for set1_ref3,4,5
    center = start_time + 150 # takes 150 s to scan one way (set1)

    filtered_df = doppler_df[(doppler_df[time] >= start_time) & (doppler_df[time] <= cycle_len-start_time)]

    left = filtered_df[filtered_df[time] <= center].copy()
    right = filtered_df[filtered_df[time] > center].copy()
    left['nested cycle'] = 1
    right['nested cycle'] = 2
    filtered_df = pd.concat([left,right])
        
    return filtered_df

def scale_ops(filtered: pd.DataFrame, cycle: int, nested_cycle: int, file, poly_degree):
    cycle_data = filtered.groupby('Laser Frequency (THz)')['Time (sec)']
    sum_bycycle = cycle_data.sum()
    count_bycycle = cycle_data.count()
    avg_bycycle = sum_bycycle / count_bycycle
    freq_bycycle = np.array(avg_bycycle.index)

    # Weighted Polynomial Fit
    weights = count_bycycle  # Weighted by the number of times the frequency was measured
    X_poly = np.array(np.vander(avg_bycycle, N=poly_degree + 1, increasing=False))  # Polynomial features
    model = sm.WLS(freq_bycycle, X_poly, weights=weights).fit()
    fit_line = model.predict(X_poly)

    # Unweighted Polynomial Fit
    unweighted_model = sm.OLS(freq_bycycle, X_poly).fit()
    unweighted_fit_line = unweighted_model.predict(X_poly)

    # Get coefficients
    coeffs = model.params  # Polynomial coefficients
    r_value = model.rsquared  # Coefficient of determination (R^2)
    
    # if nested_cycle != 0: 
    #     plt.scatter(filtered['Time (sec)'], filtered['Laser Frequency (THz)'])
    #     plt.scatter(avg_bycycle, freq_bycycle, color='red', label='Average')
    #     plt.plot(avg_bycycle, fit_line, color='red', label='WLS Fit (Weighted)')
    #     plt.plot(avg_bycycle, unweighted_fit_line, color='blue', label='OLS Fit (Unweighted)')
    #     plt.title(f'{file} - Cycle {int(cycle)} - Degree {poly_degree}')
    #     plt.show()
    # # Plot the results
    # if cycle <= 1:
    # if 'Sn-120_set8_ref3' in file:
    #     plt.figure()
    #     plt.scatter(filtered['Time (sec)'], filtered['Laser Frequency (THz)'], label='Data')
    #     plt.scatter(avg_bycycle, freq_bycycle, color='red', label='Average')
    #     plt.plot(avg_bycycle, fit_line, color='red', label='WLS Fit (Weighted)')
    #     plt.plot(avg_bycycle, unweighted_fit_line, color='blue', label='OLS Fit (Unweighted)')
    #     plt.title(f'{file} - Cycle {int(cycle)} - Degree {poly_degree}')
    #     if nested_cycle != 0: 
    #         plt.title(f'{file} - Cycle {int(cycle)}_{int(nested_cycle)} - Degree {poly_degree}')
    #     plt.legend()
    #     plt.show()

    return coeffs

def filter_last_cycle_by_time(df, time_threshold=50): # takes care of last cycles that are not a part of the data but somehow they started to be recorded 
    last_cycle = df['Cycle No.'].max()
    last_cycle_df = df[df['Cycle No.'] == last_cycle]
    if last_cycle_df['Time (sec)'].max() < time_threshold:
        return df[df['Cycle No.'] != last_cycle]  
    elif last_cycle_df['Time (sec)'].max() - last_cycle_df['Time (sec)'].min() < 30: # for the case of Sn-120_set8_ref2 where the last cycle only has times 70-80 s ish 
        return df[df['Cycle No.'] != last_cycle]  
    else:
        return df

def get_scale(file, start_time, doppler_df: pd.DataFrame) -> list:
    doppler_df = filter_last_cycle_by_time(doppler_df, time_threshold=50) # removes last cycle if necessary 
    doppler_df = doppler_df.dropna(subset=['Laser Frequency (THz)'])
    triangle_scans = ['Sn-120_set1_ref3', 'Sn-120_set1_ref4', 'Sn-120_set1_ref5']

    cycle_scales = []
    filtered_list = []
    grouped_cycle = doppler_df.groupby('Cycle No.')
    poly_degree = 5

    for cycle, group in grouped_cycle: # cycle is number, group is df of the cycle 
        if any(scan in file for scan in triangle_scans): # first data needs to be filtered by cycle 
            filtered_with_nested_cycle = triangle_filters(file, start_time, group)
            nested_cycle = filtered_with_nested_cycle.groupby('nested cycle') # obj
            for nested_cycle_no, nested_group in nested_cycle:
                filtered = nested_group # these are the individual dfs 
                coeffs = scale_ops(filtered, cycle, nested_cycle_no, file, poly_degree=4)
                cycle_scales.append({'cycle': cycle, 'nested cycle': nested_cycle_no, 'coefficients': coeffs.tolist()}) 
                filtered_list.append(filtered) # adding all dfs to a list 
                # plt.scatter(filtered['Time (sec)'], filtered['Laser Frequency (THz)'])
                # print(cycle, nested_cycle_no)
        elif 'backwards' in file: # Sn-120 ref3_set3
            filtered = group[(group['Time (sec)'] >= start_time) & (group['Time (sec)'] <= 90)] # the filter isn't working so it's easier to just cut it at 90 s before the laser jumps back  
            coeffs = scale_ops(filtered, cycle, 0, file, poly_degree)  
            cycle_scales.append({'cycle': cycle, 'coefficients': coeffs.tolist()}) 
            filtered_list.append(filtered)
        else: # all other scans not triangle             
            filtered = dynamic_filters(start_time, group) # returns a df that is cycle specific and now filtered 
            if filtered['Time (sec)'].max() < 40: # this takes care of Sn-120_set6_ref3 cycle 1
                continue  # Skip this cycle if max time < 50 s

            coeffs = scale_ops(filtered, cycle, 0, file, poly_degree)
            cycle_scales.append({'cycle': cycle, 'coefficients': coeffs.tolist()}) 
            filtered_list.append(filtered) # adding all dfs to a list 
       
    filtered_df = pd.concat(filtered_list, ignore_index=True)

    return cycle_scales, filtered_df

def get_bkg(df): 
    bkg_time = 5 # s but also bin number
    bkg_df = df[df['Time (sec)'] <= bkg_time]  
    bkg_binned = (
        bkg_df
        .assign(Time_bin=pd.cut(bkg_df['Time (sec)'], bkg_time))  
        .groupby('Time_bin', observed=True)  
        .size() 
        .reset_index(name='Count')  
    )
    bkg = bkg_binned['Count'].mean()
    return bkg

def process_scaled_df(doppler_df: pd.DataFrame, scale_info: list, bkg: float) ->  pd.DataFrame:
    bin_width = 30 # MHz 

    # get a scaled df for each cycle from unbinned and prefiltered data (doppler df is really filtered df) 
    cycle_dfs = []
    for info_by_cycle in scale_info: # scale_info is a list of dictionaries 
        nested_cycle = info_by_cycle.get('nested cycle') # if nested cycle does not exist it will return None
        cycle_number = info_by_cycle['cycle']
        coeffs = info_by_cycle['coefficients']
        if nested_cycle == None:
            cycle_df = doppler_df[doppler_df['Cycle No.'] == cycle_number].copy() # filter for cycle
            scaled_cycle_df = cycle_df.assign(
                scaled_freq=lambda df: np.polyval(coeffs, df['Time (sec)'])
            )
            cycle_dfs.append(scaled_cycle_df)
        else: # nested cycles 
            grouped_cycles = doppler_df.groupby(['Cycle No.', 'nested cycle'])
            for (cycle_no, nested_cycle_no), group in grouped_cycles:
                if cycle_no == cycle_number and nested_cycle_no == nested_cycle:
                    scaled_group = group.assign(
                        scaled_freq=lambda df: np.polyval(coeffs, df['Time (sec)'])
                    )
                    cycle_dfs.append(scaled_group)

    scaled_df = pd.concat(cycle_dfs, ignore_index=True)
    
    freq_range = (scaled_df['scaled_freq'].max() - scaled_df['scaled_freq'].min()) * 1e6 # MHz
    scaled_bins = int(np.ceil(freq_range / bin_width)) # number of bins will change to ensure bin worth is consistent 

    scaled_df = (
        scaled_df
        .assign(Freq_bin_scaled=lambda df: pd.cut(df['scaled_freq'], scaled_bins))
        .groupby('Freq_bin_scaled', observed=True) # separate counts for each bin and cycle
        .size()
        .reset_index(name='Count raw')
        .assign(Bin_center=lambda df: df['Freq_bin_scaled'].apply(lambda x: x.mid))
        .assign(count_bkg=lambda df: df['Count raw'] - bkg)
    )
    scaled_df.columns = ['Freq bin', 'Count raw', 'Bin center', 'Count-bkg']#, 'Norm count']

    scaled_df['Freq bin'] = scaled_df['Freq bin'].astype('category')  # Match global dtype
    scaled_df['Bin center'] = scaled_df['Bin center'].astype(float)
    # print(scaled_df)
        # print(scaled_cycle_df)
    #     cycle_dfs.append(scaled_cycle_df)
  
    # # combine all cycles into a new df 
    # # scaled_df = pd.concat(cycle_dfs, ignore_index=True)
    # combined_df = pd.concat(cycle_dfs, ignore_index=True)
    # print(combined_df)
    # # Weighted average across cycles
    # summary_df = (
    #     combined_df
    #     .groupby('Bin center', observed=True)
    #     .agg(
    #         Total_count=('Count raw', 'sum'),
    #         Cycle_count=('Cycle No.', 'nunique')  # Count unique cycles per bin
    #     )
    #     .reset_index()
    # )

    # summary_df['Weighted_count'] = summary_df['Total_count'] * summary_df['Cycle_count']

    # # Normalize and smooth (optional)
    # summary_df['Normalized_count'] = summary_df['Weighted_count'] / summary_df['Weighted_count'].max()
    # summary_df['Smoothed_count'] = gaussian_filter1d(summary_df['Normalized_count'], sigma=2)
    # print(summary_df)

    # return summary_df
    return scaled_df 


def time_and_freq_dfs(doppler_df: pd.DataFrame, bkg):
    # Build time_df which consists of the time bins, their counts, and bin centers
    time_bins = 50
    time_df = (
        doppler_df
        .assign(Time_bin=pd.cut(doppler_df['Time (sec)'], time_bins))    # Bin 'Time (sec)' and create 'Time_bin' column
        .groupby('Time_bin', observed=True)                               # Aggregate data within each bin
        .size()                                                           # Gives the counts for each bin 
        .reset_index(name='Count raw')                                        # Convert to DataFrame with bin counts
        .assign(Bin_center=lambda df: df['Time_bin'].apply(lambda x: x.mid))  # Calculate bin centers
        .assign(count_bkg=lambda df: df['Count raw'] - bkg)
        .assign(norm_count=lambda df: df['count_bkg'] / df['count_bkg'].max())
    )
    time_df.columns = ['Time bin', 'Count raw', 'Bin center', 'Count-bkg', 'Norm count']
    time_df['Bin center'] = time_df['Bin center'].astype(float)   

    freq_bins = doppler_df['Laser Frequency (THz)'].nunique()
    freq_df = (
    doppler_df
    .assign(Freq_bin=pd.cut(doppler_df['Laser Frequency (THz)'], freq_bins))   
    .groupby('Freq_bin', observed=True)                               
    .size()                                                          
    .reset_index(name='Count raw')                                        
    .assign(Bin_center=lambda df: df['Freq_bin'].apply(lambda x: x.mid)) 
    .assign(count_bkg=lambda df: df['Count raw'] - bkg)
    .assign(norm_count=lambda df: df['count_bkg'] / df['count_bkg'].max())
    )
    freq_df.columns = ['Freq bin', 'Count raw', 'Bin center', 'Count-bkg', 'Norm count']
    freq_df['Bin center'] = freq_df['Bin center'].astype(float)

    return time_df, freq_df

def get_dfs(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.tdms'):
            isotope = next((value for key, value in isotope_mapping.items() if key in filename), None)
            raw_data = read_tdms(folder_path, filename, channel=1) 
            raw_df = create_df(raw_data)
            doppler_df = doppler_shift_calc(raw_df, isotope)
            start_time = get_start_time(doppler_df)
            cycle_scales, filtered_df = get_scale(filename, start_time, doppler_df)
            bkg = get_bkg(doppler_df)
            scaled_df = process_scaled_df(filtered_df, cycle_scales, bkg)
            time_df, freq_df = time_and_freq_dfs(filtered_df, bkg)
            yield filename, scaled_df, time_df, freq_df, isotope
        
# get_dfs('/Users/xnimir/Desktop/Sn exp 2024/data/set3/')


