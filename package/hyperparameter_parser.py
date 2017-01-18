import json
import ConfigParser

HYPER_PARAMETER_PATH = './hp.ini'
UNK_SECTION = 'unk'
EVALUATION_SECTION = 'evaluation'

def read_config(path):
    config = ConfigParser.SafeConfigParser()
    config.read(path)
    return config

def parse_init_file(path, section, parameter):
    config = read_config(path)
    return json.loads(config.get(section,parameter))

def parse_item_in_unk(parameter):
    return parse_init_file(HYPER_PARAMETER_PATH,UNK_SECTION,parameter)

def parse_item_in_evaluation(parameter):
    return parse_init_file(HYPER_PARAMETER_PATH,EVALUATION_SECTION,parameter)

def parse_whole_section(section):
    config = read_config(HYPER_PARAMETER_PATH)
    dic_to_conv = dict(config.items(section))
    return {k:json.loads(v) for k,v in dic_to_conv.items()}

def parse_unk_section():
    return parse_whole_section(UNK_SECTION)

def parse_evaluation_section():
    return parse_whole_section(EVALUATION_SECTION)
