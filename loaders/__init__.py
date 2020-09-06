def get_loader(name):
    module = __import__("loaders." + name)
    return getattr(module, name)