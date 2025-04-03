import time
import csv
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

# Setup Selenium with Chrome
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Runs Chrome in the background
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

service = Service(executeable_path = "chromedriver.exe")
driver = webdriver.Chrome(service=service, options=options)

# Define output CSV file
csv_filename = "crop_prices.csv"

# List of district IDs (excluding 20 & 31)
district_ids = [i for i in range(1, 36) if i not in [20, 31]]

# Open CSV file for writing
with open(csv_filename, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    # Writing header row
    writer.writerow(["District", "Market", "Commodity", "Arrivals", "Unit", "Variety",
                     "Grade", "Min Price", "Max Price", "Modal Price", "Price Unit"])

    for district_id in district_ids:
        print(f"Scraping data for District ID: {district_id}...")

        # Construct the URL
        url = f"https://agmarknet.gov.in/priceAndArrivals/MarketwiseSpecificCommodity4.aspx?state1=MH&&district1={district_id}"
        driver.get(url)
        time.sleep(2)  # Allow page to load

        try:
            # Find district name
            district_name = driver.find_element(By.ID, "lblTitle").text.split(",")[0].strip()

            # Extract crop price table
            rows = driver.find_elements(By.CSS_SELECTOR, "#gridRecords tr")

            for row in rows[1:]:  # Skip the header row
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) < 10:
                    continue  # Skip empty rows

                # Extract data
                market = cols[0].text.strip() or district_name
                data = [market] + [col.text.strip() for col in cols[1:]]
                writer.writerow([district_name] + data)

            print(f"✅ Data saved for District ID: {district_id}")

        except Exception as e:
            print(f"❌ Failed to retrieve data for district {district_id}: {e}")

# Close the browser
driver.quit()

print(f"\n✅ CSV file '{csv_filename}' created successfully!")

df = pd.read_csv(csv_filename)

# Remove unwanted text from 'District' column
df['District'] = df['District'].str.replace("Latest Market Prices Available ", "", regex=False)

# Clean 'Market' column (if it has similar unwanted text)
df['Market'] = df['Market'].str.replace("Latest Market Prices Available ", "", regex=False)

# Remove text inside brackets from 'Commodity' column
df['Commodity'] = df['Commodity'].apply(lambda x: re.sub(r"\(.*?\)", "", str(x)).strip())

# Convert 'Minimum Price' and 'Maximum Price' from Rs per Quintal to Rs per kg
df['Min Price'] = pd.to_numeric(df['Min Price'], errors='coerce') / 100
df['Max Price'] = pd.to_numeric(df['Max Price'], errors='coerce') / 100

# Select only required columns
df = df[['District', 'Market', 'Commodity', 'Min Price', 'Max Price']]

# Save cleaned data
cleaned_file_path = "cleaned_crop_prices.csv"
df.to_csv(cleaned_file_path, index=False)

print(f"Cleaned data saved to {cleaned_file_path}")
