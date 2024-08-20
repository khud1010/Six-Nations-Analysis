#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#%%
# Read in the csv from the data scrape
og_df = pd.read_csv('./six_nations_team_stats.csv')
print('='*50)
print(f"DATAFRAME INFO:\n{og_df.info()}")
print(f"\nDATAFRAME DESCRIPTION:\n{og_df.describe()}")

# Calculate the average for each team across all years
team_averages = og_df.groupby('COUNTRY').mean()
team_lineout_avg = team_averages[['LINEOUT TAKES', 'LINEOUT STEALS', 'LINEOUTS WON']]

# Ensure the 'COUNTRY' column is included in the DataFrame (if it's not the index)
if og_df.index.name == 'COUNTRY':
    og_df.reset_index(inplace=True)

# Merge the averages back into the original DataFrame
df_w_lineouts = og_df.merge(team_lineout_avg, on='COUNTRY', suffixes=('', '_avg'))

# Fill NaN values in the specified columns with the averages
df_w_lineouts['LINEOUT TAKES'] = df_w_lineouts['LINEOUT TAKES'].fillna(df_w_lineouts['LINEOUT TAKES_avg'])
df_w_lineouts['LINEOUT STEALS'] = df_w_lineouts['LINEOUT STEALS'].fillna(df_w_lineouts['LINEOUT STEALS_avg'])
df_w_lineouts['LINEOUTS WON'] = df_w_lineouts['LINEOUTS WON'].fillna(df_w_lineouts['LINEOUTS WON_avg'])

# Drop the temporary average columns
df_w_lineouts.drop(columns=['LINEOUT TAKES_avg', 'LINEOUT STEALS_avg', 'LINEOUTS WON_avg'], inplace=True)

# Fill SUCCESSFUL CONVERSIONS column for missing value
missing_column = df_w_lineouts[df_w_lineouts['SUCCESSFUL CONVERSIONS'].isnull()]
missing_country = missing_column['COUNTRY']

all_missing_country_stats = df_w_lineouts[df_w_lineouts['COUNTRY'] == missing_country.values[0]]

df_w_lineouts['SUCCESSFUL CONVERSIONS'].fillna(value=df_w_lineouts['SUCCESSFUL CONVERSIONS'].mean(), inplace=True) 

# Read in list of six nations winners
winners_df = pd.read_csv('./six_nations_champions.csv')

# Add 'winner' column initialized to 0
df_w_lineouts['WINNER'] = 0

# Iterate through winners_df
for idx, row in winners_df.iterrows():
    # Define the condition for finding matching rows in df
    condition = (df_w_lineouts['YEAR'] == row['YEAR']) & (df_w_lineouts['COUNTRY'] == row['COUNTRY'].upper())
    
    # Update the 'WINNER' column for matching rows
    df_w_lineouts.loc[condition, 'WINNER'] = 1

# Drop the year column and any remaining column with a nan
no_nans_df = df_w_lineouts.drop(columns='YEAR')
no_nans_df = no_nans_df.dropna(axis=1) 

final_df = no_nans_df.select_dtypes(include=['number'])

# Calculate and plot the correlation matrix
corr_matrix = final_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

print(f"\nDATAFRAME DESCRIPTION (POST DATA PROCESSING):\n{final_df.describe()}")

# Calculate average statistics by country and year
average_stats = df_w_lineouts.groupby(['COUNTRY', 'YEAR']).mean().reset_index()

# Plot Total Points
plt.figure(figsize=(12, 8))
sns.lineplot(data=average_stats, x='YEAR', y='TOTAL POINTS', hue='COUNTRY', marker='o')
plt.title('Total Points by Country Over Time')
plt.xlabel('Year')
plt.ylabel('Average Total Points')
plt.tight_layout()
plt.show()

# Plot ATTACKING RUCK ARRIVALS
plt.figure(figsize=(12, 8))
sns.lineplot(data=average_stats, x='YEAR', y='ATTACKING RUCK ARRIVALS', hue='COUNTRY', marker='o')
plt.title('ATTACKING RUCK ARRIVALS by Country Over Time')
plt.xlabel('Year')
plt.ylabel('Average ATTACKING RUCK ARRIVALS')
plt.tight_layout()
plt.show()

# Plot LINEBREAKS
plt.figure(figsize=(12, 8))
sns.lineplot(data=average_stats, x='YEAR', y='LINEBREAKS', hue='COUNTRY', marker='o')
plt.title('LINEBREAKS by Country Over Time')
plt.xlabel('Year')
plt.ylabel('Average LINEBREAKS')
plt.tight_layout()
plt.show()

# Get a complete list of countries
countries = average_stats['COUNTRY'].unique()

# Count the number of wins (i.e., WINNER == 1) for each country
win_counts = average_stats[average_stats['WINNER'] == 1]['COUNTRY'].value_counts().reset_index()
win_counts.columns = ['COUNTRY', 'WIN_COUNT']

# Ensure all countries are in the data, even if they have no wins
win_counts = win_counts.set_index('COUNTRY').reindex(countries, fill_value=0).reset_index()

# Plotting the bar plot
plt.figure(figsize=(12, 8))
sns.barplot(data=win_counts, x='COUNTRY', y='WIN_COUNT', hue='COUNTRY')
plt.title('Number of Wins by Country')
plt.xlabel('Country')
plt.ylabel('Number of Wins')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

final_df.to_csv('./final_six_nations_stats.csv')
#%%

