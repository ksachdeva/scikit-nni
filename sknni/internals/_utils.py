import importlib

def import_class(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c

def get_class(fqn):
    class_data = fqn.split(".")
    module_path = ".".join(class_data[:-1])
    class_name = class_data[-1]
    return import_class(module_path, class_name)
