from typing import Any, Dict, Optional, Type


class Registry:
    """
    A simple registry to map strings to classes.
    """

    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type] = {}

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self._name}, items={self._module_dict})"

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key: str) -> Optional[Type]:
        return self._module_dict.get(key)

    def register_module(self, name: Optional[str] = None, module: Optional[Type] = None):
        """
        Register a module.

        Args:
            name (str, optional): The name of the module. If None, use the class name.
            module (Type, optional): The module to register.
        """
        if module is not None:
            self._register_module(module=module, module_name=name)
            return module

        def _register(module_cls):
            self._register_module(module=module_cls, module_name=name)
            return module_cls

        return _register

    def _register_module(self, module: Type, module_name: Optional[str] = None):
        if not isinstance(module, type):
            raise TypeError(f"module must be a class, but got {type(module)}")

        if module_name is None:
            module_name = module.__name__

        if module_name in self._module_dict:
            raise KeyError(f"{module_name} is already registered in {self.name}")

        self._module_dict[module_name] = module

    def build(self, cfg: Dict[str, Any], default_args: Optional[Dict[str, Any]] = None) -> Any:
        """
        Build an instance from config.

        Args:
            cfg (dict): Config dict. It should contain at least a 'type' key.
            default_args (dict, optional): Default initialization arguments.
        """
        if not isinstance(cfg, dict):
            raise TypeError(f"cfg must be a dict, but got {type(cfg)}")
        if "type" not in cfg:
            raise KeyError(f"the config must contain the key 'type', but got {cfg}")

        args = cfg.copy()
        obj_type = args.pop("type")

        if isinstance(obj_type, str):
            obj_cls = self.get(obj_type)
            if obj_cls is None:
                raise KeyError(f"{obj_type} is not in the {self.name} registry")
        elif isinstance(obj_type, type):
            obj_cls = obj_type
        else:
            raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")

        if default_args is not None:
            for name, value in default_args.items():
                args.setdefault(name, value)

        return obj_cls(**args)


# Common registries
MODELS = Registry("models")
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
HOOKS = Registry("hooks")
RUNNERS = Registry("runners")
OPTIMIZERS = Registry("optimizers")
SCHEDULERS = Registry("schedulers")
