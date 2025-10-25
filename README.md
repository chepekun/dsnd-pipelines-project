# Fashion Forward Forecasting

This project uses a data science pipeline to predict the probability of a positive recommendation based on reviews given to a clothing article.

## Getting started

The CRISP-DM process is followed in 4 Jupyter Notebooks:
* **01_data_understanding**: Explores the data and the relation between the different review properties.
    * Requirements: data/reviews.csv
* **02_data_preparation**: Cleans the data and pre-processes text columns for more efficient modelling. .
    * Requirements: data/reviews.csv + stylesense module
    * Creates: data/reviews_processed.csv
* **03_modelling**: Creates a pipeline to fit and evaluate a prediction model for the recommendation probability.
    * Requirements: data/reviews_processed.csv + stylesense module
    * Creates: data/model.joblib
* **04_deployment**: Shows a proof of principle of how new data can be fitted with the model.
    * Requirements: data/model.joblib + stylesense module
    
The **stylesense** module contains the custom transformers used in the model. The module code and distribution packaged is part of this code base.

## Structure

The project has the following folders:
* **data**: contains the raw and pre-processed review data plus the exported trained model.
* **notebooks**: contains the CRIPS-DM workflow in separate notebooks as explained above.
* **src/stylesense**: contains the module with custom transformers used in the model
* **src/dist**: contains the packaged stylesense module
* **tests**: contains a simple test to check if the environment is correctly configured.

## Dependencies

Packages required for the analysis:
```
numpy
pandas
seaborn
scikit-learn
joblib

# spacy nlp
spacy
en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# stylesense module
./src/dist/stylesense-1.0.0.tar.gz
```

Packages required for the IDE
```
ruff
pyright
pytest
setuptools
```

## Installation

There are two ways to install this project:
* Use uv (https://docs.astral.sh/uv/) and sync with pyproject.toml
* Run pip install -r /requirements.txt

## License

[License](LICENSE.txt)