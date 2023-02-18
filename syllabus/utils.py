def decorate_all_functions(function_decorator):
    def decorator(cls):
        for base_cls in cls.__bases__:
            for name, obj in vars(base_cls).items():
                parent_func = getattr(base_cls, name)
                child_func = getattr(cls, name)

                # Only apply decorator to functions not overridden by subclass.
                if callable(obj) and child_func == parent_func:
                    setattr(cls, name, function_decorator(obj))
            return cls
    return decorator