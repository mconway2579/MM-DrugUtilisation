#!/bin/bash

# Define the URL and output file name for CDC data
URL="https://data.cdc.gov/api/views/7nwe-3aj9/rows.csv?accessType=DOWNLOAD"
DATA_DIR="Data"
OUTPUT_FILE="${DATA_DIR}/cdc_data.csv"

# Create the "Data" directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Download the CDC data file
wget -O "$OUTPUT_FILE" "$URL"

# Print success message
if [[ $? -eq 0 ]]; then
  echo "File downloaded successfully as $OUTPUT_FILE"
else
  echo "Failed to download $URL."
fi

BASE_URL="https://download.medicaid.gov/data/StateDrugUtilizationData"

# Loop through the years 1991 to 2018
for YEAR in {1991..2018}; do
  # Construct the file URL
  FILE_URL="${BASE_URL}${YEAR}.csv"
  
  # Construct the output file name
  OUTPUT_FILE="${DATA_DIR}/StateDrugUtilizationData${YEAR}.csv"
  
  # Download the file
  echo "Downloading ${FILE_URL}..."
  wget -O "$OUTPUT_FILE" "$FILE_URL"
  
  # Check if the download was successful
  if [[ $? -eq 0 ]]; then
    echo "Downloaded ${OUTPUT_FILE} successfully."
  else
    echo "Failed to download ${FILE_URL}."
  fi
done
