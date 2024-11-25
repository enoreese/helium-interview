# Data Preprocessing Documentation
## 1. Overview
This document outlines the preprocessing steps applied to the raw patient and hospital visit datasets for the purpose of developing predictive models in the healthcare domain.

## 2. Raw Data
The raw data consists of two primary datasets:

- Patient Data (`Patient_Demo.csv`):
    - Columns: `institution_id`, `patient_id`, `dob` (date of birth), `sex`, `state`.
- Hospital Visit Data (`Patient_Hospital_Visit.csv`):
    - Columns: `visit_id`, `patient_id`, `institution_id`, `admitted_at`, `discharged_at`, `inserted_at`, `visit_type`, `facility_type`.

## 3. Patient Data Processing
#### Handling State Information
- Extracted Nigerian states and cities from a JSON file.
- Cleaned and standardized the state column.
- Mapped states using fuzzy matching with a threshold of 50.
- Imputed missing states with the label "not_provided" for records with empty state information.

#### Handling Date of Birth
- Calculated the age from the dob column.
- Imputed missing age values using the median.

#### Handling Gender Information
- Encoded gender using Label Encoding.
- Imputed missing gender values with the most frequent gender.

#### Creating New Columns
Selected relevant columns for the main patient dataset:
- Columns: `institution_id`, `patient_id`, `new_state` -> `state`, `imp_age` -> `age`, `imp_sex` -> `sex`.

### 4. Hospital Visit Data Processing
- Drop duplicate entries in the hospital visit dataset based on 'visit_id'
- Selected only those records where patient information is available.
- Dropped unnecessary columns (admitted_at, discharged_at, updated_at).

### 5. Merging Datasets
Merged the cleaned hospital visit data with the processed patient data using the patient_id and institution_id as keys.

### 6. Output
The final preprocessed and integrated dataset is saved as merged_data and ready for further analysis and model development.

### 7. Conclusion
The preprocessing steps ensure data consistency, handle missing values, and create a unified dataset for developing predictive models.