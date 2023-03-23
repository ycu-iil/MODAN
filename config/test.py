import yaml
with open('setting.yaml') as file:
    config = yaml.safe_load(file.read())
print(config)
