import yaml
import pandas as pd


with open('../model-definition.yml') as f:
    definition = yaml.load(f, yaml.SafeLoader)

spec = definition['spec']
inputs = spec['inputs']
parameters = inputs['parameters']
dataslots = inputs['dataslots']
metadata = definition['metadata']
outputs = spec['outputs']['datasets']

s = f'## CityCAT on DAFNI\n\n{metadata["description"]}'

s += '\n\n## Parameters\n\n'

s += pd.DataFrame(parameters)[['name', 'title', 'description']].to_markdown(index=False)


s += '\n\n## Dataslots\n\n'

s += pd.DataFrame(dataslots)[['path', 'name', 'description']].to_markdown(index=False)

s += '\n\n## Outputs\n\n'

s += pd.DataFrame(outputs)[['name', 'description']].to_markdown(index=False)

with open('citycat-dafni.md', 'w') as f:
    f.write(s)
