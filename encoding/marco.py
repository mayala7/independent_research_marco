from encoding import * 
import subprocess


save_location = "path_here"


subjects = [
    "UTS01",
    "UTS02",
    "UTS03",
    "UTS04",
    "UTS05"
]

features_stimulus = ["language", "phoneme", "word_rate"]

for subject in subjects:
    for feature in features_stimulus:
        command = ["python", "encoding/encoding.py", "--subject", subject, "--feature", feature]
        subprocess.run(command)





