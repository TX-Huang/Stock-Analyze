import os
import shutil
import logging
import pandas as pd
from finlab import data

logging.basicConfig(filename="finlab_debug.log", level=logging.INFO, format='%(asctime)s - %(message)s')

def safe_finlab_data_get(dataset_name: str, force_download=False):
    """
    Wrapper around finlab.data.get() that actively intercepts Pickle backwards-compatibility crashes.
    """
    try:
        df = data.get(dataset_name, force_download=force_download)
        return df
    except NotImplementedError as e:
        if "CategoricalDtype" in str(e):
            logging.warning(f"Detected toxic Pickle cache for '{dataset_name}' due to Pandas version mismatch. Forcing redownload.")
            # Clear internal cache if possible
            try:
                # Some versions of FinLab have this
                data.clear_cache()
            except:
                pass

            # Delete physical cache directory locally if it exists. Usually ~/.finlab/ or ./data
            # Let's rely on force_download=True first which usually overwrites the pickle.
            return data.get(dataset_name, force_download=True)
        else:
            raise e
    except Exception as e:
        if "Categorical" in str(e) or "unpickle" in str(e).lower():
            logging.warning(f"Detected general unpickle error for '{dataset_name}'. Forcing redownload. Error: {e}")
            return data.get(dataset_name, force_download=True)
        raise e

# To be injected into data_provider.py
