# %%
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd

def scrape_stats(years):
    # Set up the Selenium WebDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    all_data = []

    for year in years:
        # URL of the webpage
        url = f"https://www.sixnationsrugby.com/en/m6n/stats/{year}?tab=teams"

        driver.get(url)

        driver.implicitly_wait(10)

        # Find all div elements with the statistics
        numerical_data_elements = driver.find_elements(By.CLASS_NAME, 'performersTable_performersTable__acohX') 

        year_data = []
        for element in numerical_data_elements:
            data = element.text
            year_data.append(data)
        
        all_data.append(process_data(year_data, year))

    driver.quit()

    return pd.concat(all_data)

def process_data(data_list, year):
    all_dfs = []
    for data in data_list:
        # Split the data into lines
        lines = data.split('\n')
        # Extract the category from the first line
        category = lines[0].strip()
        # Extract the column names from the second line
        columns = lines[1:3]
        # Flatten the remaining data and group every three elements
        flat_data = [item for sublist in [line.split() for line in lines[3:]] for item in sublist]
        grouped_data = [flat_data[i:i+3] for i in range(0, len(flat_data), 3)]
        [l.append(year) for l in grouped_data]

        columns.insert(1, 'COUNTRY')
        columns[2] = category
        columns.append('YEAR')

        # Create a DataFrame 
        df = pd.DataFrame(grouped_data, columns=columns)
        df = df.drop(columns=['POS'])
        
        all_dfs.append(df)

     # Merge all DataFrames on 'COUNTRY' and 'YEAR'
    merged_df = pd.concat([df.set_index(['COUNTRY', 'YEAR']) for df in all_dfs], axis=1).reset_index()
    return merged_df

# List of years from 2000 to 2024
years = list(range(2000, 2025))

# Call the function and store the results
six_nations_numerical_data = scrape_stats(years)

six_nations_numerical_data.to_csv('./six_nations_team_stats.csv', index=False)  

# %%
