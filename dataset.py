"""
Extracting part of the ALPACA dataset, for example extracting 500 row from the whole datset
This is to minimize compute resource used to test the fine-tuning on smaller GPU

"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from datasets import load_dataset

ds = load_dataset("tatsu-lab/alpaca")
data = ds["train"].shuffle(seed = 42)#ds.select(range(50000))

data.save_to_disk("./data/alpaca_5k")
