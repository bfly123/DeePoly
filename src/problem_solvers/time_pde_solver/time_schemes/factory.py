"""Time integration scheme factory"""

from typing import Dict, Type
from .base_time_scheme import BaseTimeScheme
from .imex_rk_222 import ImexRK222
from .onestep_predictor import OneStepPredictor


class TimeSchemeFactory:
    """Factory class for creating time integration schemes"""
    
    # Registry of available time schemes
    _schemes: Dict[str, Type[BaseTimeScheme]] = {
        "imex_rk_222": ImexRK222,
        "imex_rk_2_2_2": ImexRK222,  # Alternative naming
        "onestep_predictor": OneStepPredictor,
        "onestep": OneStepPredictor,  # Alternative naming
    }
    
    @classmethod
    def create_scheme(cls, scheme_name: str, config) -> BaseTimeScheme:
        """Create a time integration scheme instance
        
        Args:
            scheme_name: Name of the time scheme
            config: Configuration object
            
        Returns:
            BaseTimeScheme: Instance of the requested time scheme
            
        Raises:
            ValueError: If the scheme name is not supported
        """
        if scheme_name not in cls._schemes:
            available_schemes = list(cls._schemes.keys())
            raise ValueError(
                f"Unsupported time scheme: {scheme_name}. "
                f"Available schemes: {available_schemes}"
            )
        
        scheme_class = cls._schemes[scheme_name]
        return scheme_class(config)
    
    @classmethod
    def register_scheme(cls, name: str, scheme_class: Type[BaseTimeScheme]):
        """Register a new time integration scheme
        
        Args:
            name: Name to register the scheme under
            scheme_class: Class implementing BaseTimeScheme
        """
        if not issubclass(scheme_class, BaseTimeScheme):
            raise TypeError("scheme_class must inherit from BaseTimeScheme")
        
        cls._schemes[name] = scheme_class
    
    @classmethod
    def get_available_schemes(cls) -> Dict[str, str]:
        """Get information about available time schemes
        
        Returns:
            Dict[str, str]: Dictionary mapping scheme names to descriptions
        """
        scheme_info = {}
        for name, scheme_class in cls._schemes.items():
            # Create a temporary instance to get scheme info
            try:
                # This requires a minimal config - might need adjustment
                temp_instance = scheme_class(None)
                info = temp_instance.get_scheme_info()
                scheme_info[name] = info.get("method", "Unknown method")
            except:
                scheme_info[name] = scheme_class.__name__
        
        return scheme_info
    
    @classmethod
    def list_schemes(cls):
        """Print available time integration schemes"""
        schemes = cls.get_available_schemes()
        print("Available Time Integration Schemes:")
        print("=" * 40)
        for name, description in schemes.items():
            print(f"  {name}: {description}")
        print("=" * 40)


# Convenience function for direct usage
def create_time_scheme(scheme_name: str, config) -> BaseTimeScheme:
    """Convenience function to create a time scheme
    
    Args:
        scheme_name: Name of the time scheme
        config: Configuration object
        
    Returns:
        BaseTimeScheme: Instance of the requested time scheme
    """
    return TimeSchemeFactory.create_scheme(scheme_name, config)