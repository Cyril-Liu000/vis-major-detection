# import learn_ar_model as lm

import json
with open("./configs.json", encoding="utf-8") as f:
    configs = json.load(f)
print(configs)