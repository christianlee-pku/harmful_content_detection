import importlib.util
import os
import sys
from typing import Any, Dict, Optional, Union


class Config(dict):
    """
    A facility for config and config file.
    It supports loading from python file and attribute access.
    """

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def __setattr__(self, name, value):
        self[name] = value

    @staticmethod
    def fromfile(filename: str) -> 'Config':
        """
        Load a config from a file.
        
        Args:
            filename (str): Path to the config file.
            
        Returns:
            Config: The loaded config.
        """
        filename = os.path.abspath(os.path.expanduser(filename))
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Config file not found: {filename}")
            
        ext = os.path.splitext(filename)[1]
        if ext not in ['.py']:
            raise NotImplementedError(f"Only .py config files are supported for now, got {ext}")
            
        cfg_dict = Config._load_py_config(filename)
        return Config(cfg_dict)
    
    @staticmethod
    def _load_py_config(filename: str) -> Dict[str, Any]:
        """Load python config file."""
        # Load the module
        module_name = os.path.splitext(os.path.basename(filename))[0]
        spec = importlib.util.spec_from_file_location(module_name, filename)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load config from {filename}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module # needed for relative imports or internal refs
        spec.loader.exec_module(module)

        # Convert module attributes to dict, excluding private ones
        cfg_dict = {k: v for k, v in module.__dict__.items() if not k.startswith("__")}
        
        # Handle inheritance
        if "_base_" in cfg_dict:
            base_paths = cfg_dict.pop("_base_")
            if isinstance(base_paths, str):
                base_paths = [base_paths]
            
            merged_cfg = {}
            config_dir = os.path.dirname(filename)
            
            for base_path in base_paths:
                # Resolve relative paths
                if not os.path.isabs(base_path):
                    base_path = os.path.join(config_dir, base_path)
                
                base_cfg = Config._load_py_config(base_path)
                Config._merge_a_into_b(base_cfg, merged_cfg)
            
            # Merge current config on top of base
            Config._merge_a_into_b(cfg_dict, merged_cfg)
            cfg_dict = merged_cfg
            
        return cfg_dict

    @staticmethod
    def _merge_a_into_b(a: Dict, b: Dict) -> None:
        """Merge dict a into dict b. Values in a will overwrite b."""
        for k, v in a.items():
            if k in b and isinstance(b[k], dict) and isinstance(v, dict):
                Config._merge_a_into_b(v, b[k])
            else:
                b[k] = v

    def merge_from_dict(self, options: Dict[str, Any]):
        """Merge a dict into the config."""
        Config._merge_a_into_b(options, self)