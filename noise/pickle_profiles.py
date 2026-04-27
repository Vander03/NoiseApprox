import os, json, sys
from tqdm import tqdm
sys.path.append('..')
from jepa import QJEPA
import pickle

"""
Compute the CSC for each noise profile and store it in a DF
"""

CACHE_FILE = "../calibrations/profiles_cache.pkl"
def build_all_profiles_cache():
    """Run once to build the monolithic cache"""
    files = [f for f in os.listdir("../calibrations") if f.endswith(".json")]
    profiles = []
    for filename in tqdm(files, desc="Building cache"):
        filepath = os.path.join("../calibrations", filename)
        with open(filepath) as f:
            data = json.load(f)
        prof = QJEPA.build_backend(data, filename)
        if prof is not None:
            profiles.append(prof)
    
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(profiles, f)
    print(f"Cached {len(profiles)} profiles to {CACHE_FILE}")


build_all_profiles_cache()