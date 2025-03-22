import pandas as pd

class CropPriceFinder:
    def __init__(self, file_path):
        """
        Initialize the CropPriceFinder with the cleaned dataset.
        :param file_path: Path to the cleaned CSV file.
        """
        self.df = pd.read_csv(file_path)
    
    def get_crop_price(self, crop_name, district=None, market=None):
        """
        Retrieve min and max price for a given crop.
        If multiple records exist, return the average min/max price unless a district or market is specified.
        :param crop_name: Name of the crop.
        :param district: Optional, specify district to narrow search.
        :param market: Optional, specify market to narrow search.
        :return: Dictionary containing min and max price.
        """
        # Filter by crop name
        filtered_df = self.df[self.df['commodity'].str.contains(crop_name, case=False, na=False)]
        
        # Further filter by district if provided
        if district:
            filtered_df = filtered_df[filtered_df['district'].str.contains(district, case=False, na=False)]
        
        # Further filter by market if provided
        if market:
            filtered_df = filtered_df[filtered_df['market'].str.contains(market, case=False, na=False)]
        
        # If no data is found, return None
        if filtered_df.empty:
            return {"message": "No data found for the given query."}
        
        # Calculate the average price if multiple entries exist
        min_price = filtered_df['min_price'].mean()
        max_price = filtered_df['max_price'].mean()
        
        return {
            "crop": crop_name,
            "district": district if district else "Multiple",
            "market": market if market else "Multiple",
            "min_price": round(min_price, 2),
            "max_price": round(max_price, 2)
        }

# Example Usage
if __name__ == "__main__":
    crop_finder = CropPriceFinder("cleaned_crop_prices.csv")
    result = crop_finder.get_crop_price("Brinjal", district="Ahmednagar")
    print(result)
