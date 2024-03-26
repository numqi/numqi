import yaml

mkdocs_yaml_path = 'mkdocs.yml'

def set_jupyter_execute_true():
    with open(mkdocs_yaml_path, 'r') as fid:
        z0 = yaml.load(fid, yaml.Loader)
    # default to false for developer can see the output quickly
    z0['plugins']['mkdocs-jupyter']['execute'] = True
    with open(mkdocs_yaml_path, 'w') as fid:
        yaml.dump(z0, fid, yaml.Dumper)

if __name__=='__main__':
    set_jupyter_execute_true()
