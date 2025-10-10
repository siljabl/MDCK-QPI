#!/bin/bash

# TOMOCUBE
# Loop through numbers from 0 to 40
for i in {0..40}; do
  # Find files that match the pattern
  for file in *_"${i}"_heights*.tiff; do
    # Check if the file exists to avoid errors
    if [[ -f "$file" ]]; then
      # Construct the new filename
      new_file="MDCK-li_height_${i}.tiff"
      # Rename the file
      mv "$file" "$new_file"
      echo "Renamed '$file' to '$new_file'"
    fi
  done

  for file in *_"${i}"_mean*.tiff; do
    # Check if the file exists to avoid errors                                                                                            
    if [[ -f "$file" ]]; then
      # Construct the new filename                                                                                                        
      new_file="MDCK-li_refractive_index_${i}.tiff"
      # Rename the file                                                                                                                   
      mv "$file" "$new_file"
      echo "Renamed '$file' to '$new_file'"
    fi
  done
done


# HOLOMONITOR
# Use a wildcard to find all matching files
for i in {1..400}; do
  # Find files that match the pattern
  for file in Well\ *\ _reg_Zc0fluct_"${i}".tiff; do
    # Check if the file exists to avoid errors
    if [[ -f "$file" ]]; then
      # Construct the new filename
      new_file="MDCK-li_reg_zero_corr_fluct_${i}.tiff"
      # Rename the file
      mv "$file" "$new_file"
      echo "Renamed '$file' to '$new_file'"
    else
      echo "No match for file '$file'. Skipping."
    fi
  done
done

# Use a wildcard to find all matching files
for i in {1..400}; do
  # Find files that match the pattern
  for file in Well\ *\ _reg_Zc_"${i}".tiff; do
    # Check if the file exists to avoid errors
    if [[ -f "$file" ]]; then
      # Construct the new filename
      new_file="MDCK-li_reg_zero_corr_${i}.tiff"
      # Rename the file
      mv "$file" "$new_file"
      echo "Renamed '$file' to '$new_file'"
    else
      echo "No match for file '$file'. Skipping."
    fi
  done
done
