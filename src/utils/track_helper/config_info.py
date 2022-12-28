import configparser
import os 

def config_reader(file_path):
    config = configparser.ConfigParser()
    configFilePath=file_path
    config.read(configFilePath, encoding='utf-8')
    return config

def config_maker(file_path):
    config = configparser.ConfigParser()

    config['upper count setting'] = {
        'searching_cnt_lim': '5',
    }
    config['lower count setting']={
        'cnt': '0',
        'cnt2': '0',
    }
    config['lower distance setting']={
        'distance_lim' : '0',
    }
    config['track type setting']={
        'type' : 'id',
    } 
    config['matching setting']={
        'matching const' : '2.0',
    }
    config['cmd setting']={
        'max_linear const' : '0.5',
        'max_angular const' : '1.0',
        'angular const' : '0.8',
        'linear const' :'0.1',
    }
    config['camera aov']={
        'angle of view' : '85',
    }
    config['orb match']={
        'ratio' : '0.4',
        'match type': 'knnMatch',
    }
    config['init distance']={
        'init_dis' : '1.0',
    }
    with open(file_path , 'w') as f:
        config.write(f)
    return config 


def config_comparer(file_path):
    saved_config=config_reader(file_path)
    new_config=config_maker(file_path)
    if new_config!=saved_config:
        print('Comparing to new and old config file , there are difference.')
        print('Replace the old ones to new ones')
        if  os.path.isfile(file_path):
            os.remove(file_path)
            config_maker(file_path)
