def load_config(config_path:str):
    '''
    Loads the configuration file at config_path and output the dictionary that contains 
    all the configurations. Note that first row of the configuration file are comma separated
    configurations that has value of type string
    Params:
        config_path: str   Path of configuration file to load
    Returns:
        conf: Dictionary with key: str, value: any  Dictionary with keys as configuration name, and value as 
                                                    configuration value.
    '''
    conf = {}
    lines = []
    
    with open(config_path) as f:
        lines = f.readlines()
    str_fields = lines[0][:-2].split(",")
    for i in range(1,len(lines)):
        contents = lines[i][:-1].split(",")
        if contents[0] not in str_fields:
            conf[contents[0]] = int(float(contents[1]))
        else:
            conf[contents[0]] = contents[1][1:-1]
    return conf